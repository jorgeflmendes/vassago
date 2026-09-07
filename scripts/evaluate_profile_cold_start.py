"""Evaluate an onboarding-profile proxy without exposing item IDs to the ranker."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file

from vassago.config import ExperimentConfig
from vassago.contextual_ranker import ContextualEvidenceRanker
from vassago.data import Movie
from vassago.evaluation import paired_bootstrap
from vassago.hybrid_serving import CollaborativeProfileProjector
from vassago.serving import UserPreferenceProfile


def _validation_rows(interactions: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    frame = pl.read_parquet(interactions).sort(["user_id", "timestamp", "movie_id"])
    sequences = frame.group_by("user_id", maintain_order=True).agg(
        pl.col("movie_id").alias("items"),
        pl.col("timestamp").alias("timestamps"),
    )
    maximum_movie_id: Any = frame["movie_id"].max()
    if maximum_movie_id is None:
        raise ValueError("interactions contain no movie IDs")
    count = int(maximum_movie_id) + 1
    popularity = np.zeros(count, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for sequence in sequences.iter_rows(named=True):
        if len(sequence["items"]) < 3:
            continue
        history = sequence["items"][:-2]
        np.add.at(popularity, np.asarray(history, dtype=int), 1)
        rows.append(
            {
                "history": history,
                "seen": sorted(set(history)),
                "target": sequence["items"][-2],
                "timestamp": sequence["timestamps"][-2],
            }
        )
    return rows, popularity


def _profile(movies: list[Movie], history: list[int]) -> UserPreferenceProfile:
    frequencies = Counter(
        genre
        for movie_id in history
        for genre in movies[movie_id - 1].genres
        if genre != "(no genres listed)"
    )
    return UserPreferenceProfile(genres=[genre for genre, _ in frequencies.most_common(3)])


def _evaluate(
    rows: list[dict[str, Any]],
    movies: list[Movie],
    projector: CollaborativeProfileProjector,
    weights: list[float],
) -> dict[float, dict[str, Any]]:
    values: dict[float, dict[str, list[float]]] = {
        weight: {"profile": [], "popularity": []} for weight in weights
    }
    cache: dict[tuple[str, ...], np.ndarray] = {}
    years = np.array([movie.available_at for movie in movies])
    for row in rows:
        profile = _profile(movies, row["history"])
        key = tuple(sorted(profile.genres))
        if key not in cache:
            cache[key] = projector.preference_score(profile)[0]
        preference = cache[key]
        allowed = np.ones(len(movies), dtype=bool)
        allowed[np.asarray(row["seen"], dtype=int) - 1] = False
        allowed &= years <= row["timestamp"]
        eligible = np.flatnonzero(allowed)
        target = int(row["target"]) - 1
        for weight in weights:
            scores = {
                "profile": (1 - weight) * preference + weight * projector.popularity,
                "popularity": projector.popularity,
            }
            for name, score in scores.items():
                rank = 1 + int(np.sum(score[eligible] > score[target]))
                values[weight][name].append(
                    1 / np.log2(rank + 1) if rank <= 10 else 0.0
                )
    result = {}
    for weight, methods in values.items():
        profile_values = np.asarray(methods["profile"])
        popularity_values = np.asarray(methods["popularity"])
        result[weight] = {
            "profile_ndcg10": float(profile_values.mean()),
            "profile_recall10": float(
                np.count_nonzero(profile_values) / len(profile_values)
            ),
            "popularity_ndcg10": float(popularity_values.mean()),
            "popularity_recall10": float(
                np.count_nonzero(popularity_values) / len(popularity_values)
            ),
            "paired_ndcg10": paired_bootstrap(
                profile_values, popularity_values, seed=42, samples=5000
            ),
        }
    return result


@torch.inference_mode()
def _favorite_score_components(
    batch: list[dict[str, Any]],
    model: ContextualEvidenceRanker,
    max_length: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    histories = torch.zeros(len(batch), max_length, dtype=torch.long, device=device)
    for index, row in enumerate(batch):
        favorites = row["history"][-length:]
        histories[index, : len(favorites)] = torch.tensor(favorites, device=device)
    contextual = model.score(histories)[1]
    item_vectors = model.backbone.item_vectors()
    centroids = []
    for row in batch:
        favorites = torch.tensor(row["history"][-min(length, 3) :], device=device)
        centroids.append(item_vectors[favorites].mean(0))
    centroid_scores = torch.nn.functional.normalize(torch.stack(centroids), dim=-1) @ item_vectors.T
    return (
        contextual.float().cpu().numpy(),
        centroid_scores.float().cpu().numpy(),
    )


def _evaluate_favorites(
    rows: list[dict[str, Any]],
    model: ContextualEvidenceRanker,
    max_length: int,
    lengths: list[int],
    centroid_weights: list[float],
) -> dict[tuple[int, float], dict[str, float]]:
    settings = [(length, weight) for length in lengths for weight in centroid_weights]
    totals = {setting: [0.0, 0.0] for setting in settings}
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        for length in lengths:
            contextual, centroid = _favorite_score_components(
                batch, model, max_length, length
            )
            for weight in centroid_weights:
                scores = (1 - weight) * contextual + weight * centroid
                for index, row in enumerate(batch):
                    allowed = np.ones(scores.shape[1], dtype=bool)
                    allowed[0] = False
                    allowed[row["seen"]] = False
                    target = int(row["target"])
                    rank = 1 + int(np.sum(scores[index, allowed] > scores[index, target]))
                    if rank <= 10:
                        totals[(length, weight)][0] += 1 / np.log2(rank + 1)
                        totals[(length, weight)][1] += 1
    return {
        setting: {
            "ndcg10": ndcg / len(rows),
            "recall10": recall / len(rows),
        }
        for setting, (ndcg, recall) in totals.items()
    }


def _export_favorite_rankings(
    rows: list[dict[str, Any]],
    model: ContextualEvidenceRanker,
    max_length: int,
    length: int,
    centroid_weight: float,
    protocol_hash: str,
    seed: int,
    output: Path,
) -> None:
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        contextual, centroid = _favorite_score_components(batch, model, max_length, length)
        scores = (1 - centroid_weight) * contextual + centroid_weight * centroid
        for index, row in enumerate(batch):
            allowed = np.ones(scores.shape[1], dtype=bool)
            allowed[0] = False
            allowed[row["seen"]] = False
            eligible = np.flatnonzero(allowed)
            values = scores[index, eligible]
            ranking = eligible[np.argsort(-values, kind="stable")[:200]]
            predictions.extend(
                {
                    "query_id": row["query_id"],
                    "model": f"contextual-evidence-cold{length}",
                    "seed": seed,
                    "protocol_hash": protocol_hash,
                    "rank": rank,
                    "movie_id": int(movie_id),
                }
                for rank, movie_id in enumerate(ranking, 1)
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(predictions).write_parquet(output)
    manifest = {
        "schema_version": 1,
        "model": f"contextual-evidence-cold{length}",
        "protocol_hash": protocol_hash,
        "seed": seed,
        "favorite_count": length,
        "centroid_weight": centroid_weight,
        "seen_item_filter": "complete pre-query history",
        "queries": len(rows),
        "storage_format": "parquet-zstd",
        "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "status": "completed",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--cold-weights",
        type=Path,
        help="Optional validation-selected onboarding adapter checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rankings-output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    movies = [
        Movie.model_validate(row)
        for row in json.loads((args.data / "catalog.json").read_text(encoding="utf-8"))
    ]
    validation, counts = _validation_rows(args.data / "interactions.parquet")
    config = ExperimentConfig.read(args.config).model_copy(update={"device": args.device})
    model = ContextualEvidenceRanker(
        len(movies),
        config.dimension,
        config.max_length,
        config.heads,
        config.layers,
        config.dropout,
        config.contextual_dimension,
        config.contextual_memory_window,
        config.contextual_temperature,
        config.contextual_heads,
    ).to(args.device)
    model.load_state_dict(load_file(args.weights, device=args.device), strict=True)
    model.eval()
    cold_model = model
    if args.cold_weights is not None:
        cold_model = ContextualEvidenceRanker(
            len(movies),
            config.dimension,
            config.max_length,
            config.heads,
            config.layers,
            config.dropout,
            config.contextual_dimension,
            config.contextual_memory_window,
            config.contextual_temperature,
            config.contextual_heads,
        ).to(args.device)
        cold_model.load_state_dict(
            load_file(args.cold_weights, device=args.device), strict=True
        )
        cold_model.eval()
    projector = CollaborativeProfileProjector(
        movies, model.backbone.item_vectors().detach().cpu().numpy(), counts
    )
    grid = [index / 10 for index in range(11)]
    validation_results = _evaluate(validation, movies, projector, grid)
    selected = max(grid, key=lambda weight: validation_results[weight]["profile_ndcg10"])
    favorite_grid = [1, 3, 5, 10]
    centroid_weight_grid = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    favorite_validation = _evaluate_favorites(
        validation,
        cold_model,
        config.max_length,
        favorite_grid,
        centroid_weight_grid,
    )
    selected_favorites, selected_centroid_weight = max(
        favorite_validation,
        key=lambda setting: favorite_validation[setting]["ndcg10"],
    )
    test = list(pl.read_parquet(args.protocol / "queries.parquet").iter_rows(named=True))
    payload = {
        "schema_version": 1,
        "method": (
            "top-3 genre onboarding proxy derived from pre-query history; "
            "history item IDs are hidden from the cold ranker"
        ),
        "selection": {
            "split": "second-last interaction",
            "metric": "NDCG@10",
            "popularity_weight_grid": grid,
            "selected_popularity_weight": selected,
            "results": validation_results[selected],
        },
        "test": {
            "split": "final interaction",
            "queries": len(test),
            "results": _evaluate(test, movies, projector, [selected])[selected],
        },
        "favorite_onboarding": {
            "method": (
                "most recent pre-query items used as ordered favorite selections; "
                "optional cold adapter trained on ten-item windows"
            ),
            "selection": {
                "split": "second-last interaction",
                "favorite_count_grid": favorite_grid,
                "centroid_weight_grid": centroid_weight_grid,
                "selected_favorite_count": selected_favorites,
                "selected_centroid_weight": selected_centroid_weight,
                "results": favorite_validation[
                    (selected_favorites, selected_centroid_weight)
                ],
            },
            "test": _evaluate_favorites(
                test,
                cold_model,
                config.max_length,
                [selected_favorites],
                [selected_centroid_weight],
            )[(selected_favorites, selected_centroid_weight)],
        },
        "cold_adapter": {
            "weights": str(args.cold_weights) if args.cold_weights else None,
            "sha256": (
                hashlib.sha256(args.cold_weights.read_bytes()).hexdigest()
                if args.cold_weights
                else None
            ),
        },
        "warm_start_contract": "one or more positive interactions bypass the profile expert",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.rankings_output is not None:
        protocol = json.loads((args.protocol / "protocol.json").read_text(encoding="utf-8"))
        _export_favorite_rankings(
            test,
            cold_model,
            config.max_length,
            selected_favorites,
            selected_centroid_weight,
            protocol["protocol_hash"],
            config.seed,
            args.rankings_output,
        )
    print(args.output)


if __name__ == "__main__":
    main()
