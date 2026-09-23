"""Immutable sequential recommendation protocols and model-agnostic ranking evaluation."""

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from vassago.evaluation import (
    catalog_metrics,
    paired_bootstrap,
    paired_seed_bootstrap,
    ranking_metrics,
)


class FairProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[3, 4] = 3
    name: str = "ml-1m-leave-one-out-v3"
    dataset: str = "MovieLens-1M"
    split: str = "last interaction per user is test"
    tie_break: str = "timestamp ascending, canonical movie_id ascending"
    feedback: str = "all explicit ratings"
    query_time: str = "last observed interaction timestamp; held-out timestamp is not a feature"
    history_length: int = Field(default=200, ge=1)
    candidates: str = "all rated catalog items excluding the complete pre-query history"
    item_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    interactions_sha256: str
    catalog_sha256: str
    queries_sha256: str
    adapter_queries_sha256: str
    sequence_sha256: str
    training_sequence_sha256: str | None = None
    training_interactions_sha256: str | None = None
    candidate_item_ids: list[int] | None = None
    query_sample_limit: int | None = None
    query_sample_seed: int | None = None
    protocol_hash: str = ""

    def payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"protocol_hash"}, exclude_none=True)

    def expected_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weight_provenance(
    selection_weights: Path,
    selection_manifest: Path,
    final_weights: Path,
    final_manifest: Path,
    protocol_hash: str,
) -> None:
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    final = json.loads(final_manifest.read_text(encoding="utf-8"))
    selection_record = selection.get("selection_weights", {})
    final_record = final.get("weights", {})
    if selection.get("test_evaluated") is not False:
        raise ValueError("Selection manifest must certify that test was not evaluated")
    if (
        selection.get("protocol_hash") != protocol_hash
        or final.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("Weight manifests do not match the benchmark protocol")
    if selection_record.get("sha256") != sha256(selection_weights):
        raise ValueError("Selection weights do not match their manifest")
    if final_record.get("sha256") != sha256(final_weights):
        raise ValueError("Final weights do not match their manifest")


def prepare_protocol(
    data: Path,
    output: Path,
    history_length: int = 200,
    global_test_start: int | None = None,
    query_sample_limit: int | None = None,
    query_sample_seed: int = 42,
) -> Path:
    """Materialize the one shared split consumed by every model adapter."""
    if output.exists():
        raise FileExistsError(output)
    if query_sample_limit is not None and query_sample_limit < 1:
        raise ValueError("Query sample limit must be positive")
    output.mkdir(parents=True)
    interactions = pl.read_parquet(data / "interactions.parquet").sort(
        "user_id", "timestamp", "movie_id"
    )
    catalog_path = data / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if interactions.is_empty() or not catalog:
        raise ValueError("Fair benchmark requires non-empty interactions and catalog")
    item_count = len(catalog)
    catalog_sources = {str(item.get("source", "unknown")) for item in catalog}
    if len(catalog_sources) != 1:
        raise ValueError("Fair benchmark requires one catalog source")
    source = catalog_sources.pop()
    dataset_names = {
        "movielens-1m": ("MovieLens-1M", "ml-1m-leave-one-out-v3"),
        "movielens-100k": ("MovieLens-100K", "ml-100k-leave-one-out-v3"),
    }
    dataset_name, protocol_name = dataset_names.get(
        source, (source, f"{source}-leave-one-out-v3")
    )
    minimum_id = int(interactions["movie_id"].min() or 0)  # type: ignore[arg-type]
    maximum_id = int(interactions["movie_id"].max() or 0)  # type: ignore[arg-type]
    if minimum_id < 1 or maximum_id > item_count:
        raise ValueError("Interactions and catalog use incompatible canonical IDs")

    sequence_rows: list[dict[str, Any]] = []
    training_sequence_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    candidate_item_ids: list[int] | None = None
    if global_test_start is not None:
        training_interactions = interactions.filter(pl.col("timestamp") < global_test_start)
        candidate_item_ids = sorted(training_interactions["movie_id"].unique().to_list())
        candidate_set = set(candidate_item_ids)
        if not candidate_set:
            raise ValueError("Global temporal split has no catalog items before the cutoff")
    else:
        training_interactions = None
        candidate_set = set(range(1, item_count + 1))
    for user_frame in interactions.partition_by("user_id", maintain_order=True):
        rows = user_frame.iter_rows(named=True)
        ordered = list(rows)
        if global_test_start is None:
            if len(ordered) < 2:
                continue
            target = ordered[-1]
            history_rows = ordered[:-1]
        else:
            history_rows = [row for row in ordered if int(row["timestamp"]) < global_test_start]
            targets = [row for row in ordered if int(row["timestamp"]) >= global_test_start]
            if len(history_rows) >= 2:
                training_sequence_rows.append(
                    {
                        "user_id": ordered[0]["user_id"],
                        "sequence_item_ids": ",".join(
                            str(row["movie_id"]) for row in history_rows
                        ),
                        "sequence_ratings": ",".join(
                            str(int(row["rating"])) for row in history_rows
                        ),
                        "sequence_timestamps": ",".join(
                            str(row["timestamp"]) for row in history_rows
                        ),
                    }
                )
            if (
                len(history_rows) < 2
                or not targets
                or int(targets[0]["movie_id"]) not in candidate_set
            ):
                continue
            target = targets[0]
        history = [int(row["movie_id"]) for row in history_rows]
        history_timestamps = [int(row["timestamp"]) for row in history_rows]
        sequence = [*history_rows, target]
        sequence_rows.append(
            {
                "user_id": target["user_id"],
                "sequence_item_ids": ",".join(str(row["movie_id"]) for row in sequence),
                "sequence_ratings": ",".join(str(int(row["rating"])) for row in sequence),
                "sequence_timestamps": ",".join(str(row["timestamp"]) for row in sequence),
            }
        )
        query_rows.append(
            {
                "query_id": str(target["user_id"]),
                "user_id": str(target["user_id"]),
                "history": history[-history_length:],
                "history_timestamps": history_timestamps[-history_length:],
                "seen": sorted(set(history)),
                "target": int(target["movie_id"]),
                "target_rating": float(target["rating"]),
                "timestamp": history_timestamps[-1],
                "target_timestamp": int(target["timestamp"]),
                "candidate_count": len(candidate_set - set(history)),
            }
        )
    if not query_rows:
        raise ValueError("No users have enough interactions for leave-one-out evaluation")
    if query_sample_limit is not None and len(query_rows) > query_sample_limit:
        selected = sorted(
            sorted(
                range(len(query_rows)),
                key=lambda index: hashlib.sha256(
                    f"{query_sample_seed}:{query_rows[index]['query_id']}".encode()
                ).digest(),
            )[:query_sample_limit]
        )
        query_rows = [query_rows[index] for index in selected]
        sequence_rows = [sequence_rows[index] for index in selected]

    queries_path = output / "queries.parquet"
    adapter_queries_path = output / "queries.jsonl"
    sequences_path = output / "hstu_sequences.csv"
    pl.DataFrame(query_rows).write_parquet(queries_path)
    adapter_queries_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in query_rows),
        encoding="utf-8",
    )
    pl.DataFrame(sequence_rows).write_csv(sequences_path)
    training_sequences_path = output / "hstu_training_sequences.csv"
    training_interactions_path = output / "training_interactions.parquet"
    if global_test_start is not None:
        pl.DataFrame(training_sequence_rows).write_csv(training_sequences_path)
        assert training_interactions is not None
        training_interactions.write_parquet(training_interactions_path)
    protocol = FairProtocol(
        schema_version=4 if global_test_start is not None else 3,
        name=(f"{source}-global-temporal-v4" if global_test_start is not None else protocol_name),
        dataset=dataset_name,
        split=(
            "first eligible interaction at or after the global test cutoff"
            if global_test_start is not None
            else "last interaction per user is test"
        ),
        candidates=(
            "all items observed before the global test cutoff excluding complete pre-query history"
            if global_test_start is not None
            else "all rated catalog items excluding the complete pre-query history"
        ),
        history_length=history_length,
        item_count=item_count,
        query_count=len(query_rows),
        interactions_sha256=sha256(data / "interactions.parquet"),
        catalog_sha256=sha256(catalog_path),
        queries_sha256=sha256(queries_path),
        adapter_queries_sha256=sha256(adapter_queries_path),
        sequence_sha256=sha256(sequences_path),
        training_sequence_sha256=(
            sha256(training_sequences_path) if global_test_start is not None else None
        ),
        training_interactions_sha256=(
            sha256(training_interactions_path) if global_test_start is not None else None
        ),
        candidate_item_ids=candidate_item_ids,
        query_sample_limit=query_sample_limit,
        query_sample_seed=query_sample_seed if query_sample_limit is not None else None,
    )
    protocol.protocol_hash = protocol.expected_hash()
    (output / "protocol.json").write_text(
        json.dumps(protocol.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "catalog.json").write_bytes(catalog_path.read_bytes())
    return output


def _validated_predictions(
    path: Path, protocol: FairProtocol, queries: pl.DataFrame
) -> tuple[str, int, pl.DataFrame]:
    # The pinned Meta adapter emits CSV rankings even when callers retain the
    # historical `.parquet` filename. Detect the container instead of trusting
    # the suffix so every ranking is validated by the same evaluator.
    with path.open("rb") as handle:
        is_parquet = handle.read(4) == b"PAR1"
    frame = pl.read_parquet(path) if is_parquet else pl.read_csv(path)
    required = {"query_id", "model", "seed", "protocol_hash", "rank", "movie_id"}
    if required - set(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"{path} is missing prediction columns: {missing}")
    models, seeds, hashes = (
        frame["model"].unique(),
        frame["seed"].unique(),
        frame["protocol_hash"].unique(),
    )
    if len(models) != 1 or len(seeds) != 1 or hashes.to_list() != [protocol.protocol_hash]:
        raise ValueError(f"{path} must identify one model, seed and matching protocol")
    expected_queries = set(queries["query_id"].to_list())
    if set(frame["query_id"].unique().to_list()) != expected_queries:
        raise ValueError(f"{path} predictions do not cover the exact query set")
    if frame.select(pl.struct("query_id", "movie_id").is_duplicated().any()).item():
        raise ValueError(f"{path} contains duplicate recommendations")
    if frame.select(pl.struct("query_id", "rank").is_duplicated().any()).item():
        raise ValueError(f"{path} contains duplicate ranks")
    minimum_id = int(frame["movie_id"].min() or 0)  # type: ignore[arg-type]
    maximum_id = int(frame["movie_id"].max() or 0)  # type: ignore[arg-type]
    if minimum_id < 1 or maximum_id > protocol.item_count:
        raise ValueError(f"{path} recommends an item outside the shared catalog")
    counts = frame.group_by("query_id").agg(
        pl.len().alias("n"), pl.col("rank").min().alias("first"), pl.col("rank").max().alias("last")
    )
    if counts.filter((pl.col("first") != 1) | (pl.col("last") != pl.col("n"))).height:
        raise ValueError(f"{path} ranks must be contiguous and one-based")
    if "candidate_count" in queries.columns:
        expected_counts = queries.select(
            "query_id", pl.col("candidate_count").clip(upper_bound=200).alias("expected")
        )
    else:
        expected_counts = queries.select(
            "query_id",
            (protocol.item_count - pl.col("seen").list.n_unique())
            .clip(upper_bound=200)
            .alias("expected"),
        )
    if counts.join(expected_counts, on="query_id").filter(pl.col("n") != pl.col("expected")).height:
        raise ValueError(f"{path} must contain the complete shared top-200 ranking")
    seen = (
        queries.select("query_id", "seen")
        .filter(pl.col("seen").list.len() > 0)
        .explode("seen", empty_as_null=True)
        .rename({"seen": "movie_id"})
    )
    if frame.join(seen, on=["query_id", "movie_id"], how="inner").height:
        raise ValueError(f"{path} recommends items already present in query history")
    if protocol.candidate_item_ids is not None:
        allowed = set(protocol.candidate_item_ids)
        if not set(frame["movie_id"].to_list()).issubset(allowed):
            raise ValueError(f"{path} recommends an item unavailable at the global test cutoff")
    return str(models.item()), int(seeds.item()), frame.sort("query_id", "rank")


def evaluate_predictions(protocol_directory: Path, predictions: list[Path], output: Path) -> Path:
    """Evaluate external rankings only after strict protocol compatibility checks."""
    if len(predictions) < 2:
        raise ValueError("A fair comparison requires predictions from at least two models")
    protocol = FairProtocol.model_validate_json(
        (protocol_directory / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.protocol_hash != protocol.expected_hash():
        raise ValueError("Protocol manifest hash is invalid")
    queries = pl.read_parquet(protocol_directory / "queries.parquet")
    if sha256(protocol_directory / "queries.parquet") != protocol.queries_sha256:
        raise ValueError("Protocol query artifact has changed")
    if sha256(protocol_directory / "catalog.json") != protocol.catalog_sha256:
        raise ValueError("Protocol catalog artifact has changed")
    if sha256(protocol_directory / "hstu_sequences.csv") != protocol.sequence_sha256:
        raise ValueError("Protocol sequence artifact has changed")
    catalog = json.loads((protocol_directory / "catalog.json").read_text(encoding="utf-8"))
    genres = [set()] + [set(item.get("genres", [])) for item in catalog]
    counts = np.zeros(protocol.item_count + 1, dtype=np.int64)
    csv.field_size_limit(2**31 - 1)
    with (protocol_directory / "hstu_sequences.csv").open(newline="", encoding="utf-8") as handle:
        for sequence in csv.DictReader(handle):
            item_ids = [int(item) for item in sequence["sequence_item_ids"].split(",")]
            np.add.at(counts, item_ids[:-1], 1)
    eligible_items = set(protocol.candidate_item_ids or range(1, protocol.item_count + 1))
    targets = dict(queries.select("query_id", "target").iter_rows())
    target_ratings = dict(queries.select("query_id", "target_rating").iter_rows())
    per_query: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    metrics: list[dict[str, Any]] = []
    for path in predictions:
        model, seed, frame = _validated_predictions(path, protocol, queries)
        if (model, seed) in per_query:
            raise ValueError(f"Duplicate prediction run for {model} seed {seed}")
        rows = []
        ranked_lists: list[list[int]] = []
        for query_id, group in frame.partition_by("query_id", as_dict=True).items():
            key = query_id[0] if isinstance(query_id, tuple) else query_id
            ranked = group["movie_id"].to_list()
            ranked_lists.append(ranked)
            values = {
                metric: value
                for cutoff in (10, 50, 200)
                for metric, value in ranking_metrics(ranked, {targets[str(key)]}, cutoff).items()
            }
            rows.append(
                {
                    "query_id": str(key),
                    "target_rating": target_ratings[str(key)],
                    **values,
                }
            )
        result = pl.DataFrame(rows).sort("query_id")
        metric_columns = [
            column for column in result.columns if column not in {"query_id", "target_rating"}
        ]
        aggregate = result.select(metric_columns).mean().to_dicts()[0]
        positive = result.filter(pl.col("target_rating") >= 4)
        positive_aggregate = positive.select(metric_columns).mean().to_dicts()[0]
        metrics.append(
            {
                "model": model,
                "seed": seed,
                "queries": len(rows),
                "positive_queries": positive.height,
                "catalog_metrics@10": catalog_metrics(
                    [ranked[:10] for ranked in ranked_lists], counts, genres, eligible_items
                ),
                "catalog_metrics@50": catalog_metrics(
                    [ranked[:50] for ranked in ranked_lists], counts, genres, eligible_items
                ),
                **aggregate,
                **{f"{metric}_rating>=4": value for metric, value in positive_aggregate.items()},
            }
        )
        per_query[(model, seed)] = {column: result[column].to_numpy() for column in metric_columns}
    significance = []
    for a, b in combinations(sorted(per_query), 2):
        if a[1] != b[1]:
            continue
        significance.append(
            {
                "model_a": a[0],
                "model_b": b[0],
                "seed": a[1],
                "metric": "NDCG@10",
                **paired_bootstrap(per_query[a]["NDCG@10"], per_query[b]["NDCG@10"], a[1]),
            }
        )
    aggregate_significance = []
    models = sorted({model for model, _ in per_query})
    for model_a, model_b in combinations(models, 2):
        seeds_a = {seed for model, seed in per_query if model == model_a}
        seeds_b = {seed for model, seed in per_query if model == model_b}
        if seeds_a != seeds_b or len(seeds_a) < 2:
            continue
        aggregate_significance.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "seeds": sorted(seeds_a),
                "metric": "NDCG@10",
                "method": "paired crossed multiplier bootstrap over seeds and users",
                **paired_seed_bootstrap(
                    {seed: per_query[(model_a, seed)]["NDCG@10"] for seed in seeds_a},
                    {seed: per_query[(model_b, seed)]["NDCG@10"] for seed in seeds_a},
                    20260914,
                ),
            }
        )
    seed_summary = []
    metric_names = sorted(
        key
        for key, value in metrics[0].items()
        if "@" in key and isinstance(value, (int, float))
    )
    for model in sorted({row["model"] for row in metrics}):
        model_runs = [row for row in metrics if row["model"] == model]
        summary: dict[str, Any] = {
            "model": model,
            "seeds": sorted(row["seed"] for row in model_runs),
            "runs": len(model_runs),
        }
        rng = np.random.default_rng(20260908)
        for metric in metric_names:
            run_values = np.array([row[metric] for row in model_runs], dtype=np.float64)
            summary[metric] = {
                "mean": float(run_values.mean()),
                "sample_std": float(run_values.std(ddof=1)) if len(run_values) > 1 else None,
                "bootstrap_95_ci": (
                    np.quantile(
                        run_values[
                            rng.integers(0, len(run_values), size=(10_000, len(run_values)))
                        ].mean(axis=1),
                        [0.025, 0.975],
                    ).tolist()
                    if len(run_values) > 1
                    else None
                ),
            }
        seed_summary.append(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol_hash": protocol.protocol_hash,
                "metrics": metrics,
                "across_seed_summary": seed_summary,
                "paired_significance": significance,
                "across_seed_paired_significance": aggregate_significance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def popularity_predictions(data: Path, protocol_directory: Path, seed: int, output: Path) -> Path:
    """Export a deterministic train-only popularity baseline under the shared protocol."""
    protocol = FairProtocol.model_validate_json(
        (protocol_directory / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.protocol_hash != protocol.expected_hash():
        raise ValueError("Protocol manifest hash is invalid")
    queries = pl.read_parquet(protocol_directory / "queries.parquet")
    training_interactions = protocol_directory / "training_interactions.parquet"
    if protocol.training_interactions_sha256 is not None:
        if sha256(training_interactions) != protocol.training_interactions_sha256:
            raise ValueError("Protocol training interactions artifact has changed")
        counts = pl.read_parquet(training_interactions).group_by("movie_id").len()
    else:
        interactions = pl.read_parquet(data / "interactions.parquet")
        held_out = queries.group_by("target").len().rename(
            {"target": "movie_id", "len": "held_out"}
        )
        counts = (
            interactions.group_by("movie_id")
            .len()
            .join(held_out, on="movie_id", how="left")
            .with_columns((pl.col("len") - pl.col("held_out").fill_null(0)).alias("count"))
        )
    popularity = np.zeros(protocol.item_count + 1, dtype=np.int64)
    for movie_id, count in counts.select("movie_id", "count").iter_rows():
        popularity[movie_id] = count
    catalog_ids = np.arange(1, len(popularity))
    candidates = np.array(protocol.candidate_item_ids or catalog_ids.tolist())
    global_order = candidates[np.lexsort((candidates, -popularity[candidates]))]
    rows: list[dict[str, Any]] = []
    for query in queries.iter_rows(named=True):
        seen = set(query["seen"])
        ranked = [int(item) for item in global_order if item not in seen][:200]
        rows.extend(
            {
                "query_id": query["query_id"],
                "model": "popularity",
                "seed": seed,
                "protocol_hash": protocol.protocol_hash,
                "rank": rank,
                "movie_id": movie_id,
            }
            for rank, movie_id in enumerate(ranked, 1)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output)
    return output
