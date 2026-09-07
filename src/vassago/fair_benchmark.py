"""Immutable leave-one-out protocol and model-agnostic ranking evaluation."""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from vassago.evaluation import paired_bootstrap, ranking_metrics


class FairProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    name: str = "ml-1m-leave-one-out-v1"
    dataset: str = "MovieLens-1M"
    split: str = "last interaction per user is test"
    tie_break: str = "timestamp ascending, canonical movie_id ascending"
    feedback: str = "all explicit ratings"
    history_length: int = Field(default=200, ge=1)
    candidates: str = "all rated catalog items excluding the encoded history window"
    item_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    interactions_sha256: str
    catalog_sha256: str
    queries_sha256: str
    adapter_queries_sha256: str
    sequence_sha256: str
    protocol_hash: str = ""

    def payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"protocol_hash"})

    def expected_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_protocol(data: Path, output: Path, history_length: int = 200) -> Path:
    """Materialize the one shared split consumed by every model adapter."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    interactions = pl.read_parquet(data / "interactions.parquet").sort(
        "user_id", "timestamp", "movie_id"
    )
    catalog_path = data / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if interactions.is_empty() or not catalog:
        raise ValueError("Fair benchmark requires non-empty interactions and catalog")
    item_count = len(catalog)
    minimum_id = int(interactions["movie_id"].min() or 0)  # type: ignore[arg-type]
    maximum_id = int(interactions["movie_id"].max() or 0)  # type: ignore[arg-type]
    if minimum_id < 1 or maximum_id > item_count:
        raise ValueError("Interactions and catalog use incompatible canonical IDs")

    sequence_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for user_frame in interactions.partition_by("user_id", maintain_order=True):
        rows = user_frame.iter_rows(named=True)
        ordered = list(rows)
        if len(ordered) < 2:
            continue
        target = ordered[-1]
        history = [int(row["movie_id"]) for row in ordered[:-1]]
        sequence_rows.append(
            {
                "user_id": target["user_id"],
                "sequence_item_ids": ",".join(str(row["movie_id"]) for row in ordered),
                "sequence_ratings": ",".join(str(int(row["rating"])) for row in ordered),
                "sequence_timestamps": ",".join(str(row["timestamp"]) for row in ordered),
            }
        )
        query_rows.append(
            {
                "query_id": str(target["user_id"]),
                "user_id": str(target["user_id"]),
                "history": history[-history_length:],
                "seen": sorted(set(history[-history_length:])),
                "target": int(target["movie_id"]),
                "target_rating": float(target["rating"]),
                "timestamp": int(target["timestamp"]),
            }
        )
    if not query_rows:
        raise ValueError("No users have enough interactions for leave-one-out evaluation")

    queries_path = output / "queries.parquet"
    adapter_queries_path = output / "queries.jsonl"
    sequences_path = output / "hstu_sequences.csv"
    pl.DataFrame(query_rows).write_parquet(queries_path)
    adapter_queries_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in query_rows),
        encoding="utf-8",
    )
    pl.DataFrame(sequence_rows).write_csv(sequences_path)
    protocol = FairProtocol(
        history_length=history_length,
        item_count=item_count,
        query_count=len(query_rows),
        interactions_sha256=sha256(data / "interactions.parquet"),
        catalog_sha256=sha256(catalog_path),
        queries_sha256=sha256(queries_path),
        adapter_queries_sha256=sha256(adapter_queries_path),
        sequence_sha256=sha256(sequences_path),
    )
    protocol.protocol_hash = protocol.expected_hash()
    (output / "protocol.json").write_text(
        json.dumps(protocol.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return output


def _validated_predictions(
    path: Path, protocol: FairProtocol, queries: pl.DataFrame
) -> tuple[str, int, pl.DataFrame]:
    frame = pl.read_csv(path) if path.suffix.lower() == ".csv" else pl.read_parquet(path)
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
    minimum_id = int(frame["movie_id"].min() or 0)  # type: ignore[arg-type]
    maximum_id = int(frame["movie_id"].max() or 0)  # type: ignore[arg-type]
    if minimum_id < 1 or maximum_id > protocol.item_count:
        raise ValueError(f"{path} recommends an item outside the shared catalog")
    counts = frame.group_by("query_id").agg(
        pl.len().alias("n"), pl.col("rank").min().alias("first"), pl.col("rank").max().alias("last")
    )
    if counts.filter((pl.col("first") != 1) | (pl.col("last") != pl.col("n"))).height:
        raise ValueError(f"{path} ranks must be contiguous and one-based")
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
    targets = dict(queries.select("query_id", "target").iter_rows())
    target_ratings = dict(queries.select("query_id", "target_rating").iter_rows())
    per_query: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    metrics: list[dict[str, Any]] = []
    for path in predictions:
        model, seed, frame = _validated_predictions(path, protocol, queries)
        if (model, seed) in per_query:
            raise ValueError(f"Duplicate prediction run for {model} seed {seed}")
        rows = []
        for query_id, group in frame.partition_by("query_id", as_dict=True).items():
            key = query_id[0] if isinstance(query_id, tuple) else query_id
            ranked = group["movie_id"].to_list()
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
    seed_summary = []
    metric_names = sorted(key for key in metrics[0] if "@" in key)
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
    interactions = pl.read_parquet(data / "interactions.parquet")
    held_out = queries.group_by("target").len().rename({"target": "movie_id", "len": "held_out"})
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
    global_order = catalog_ids[np.lexsort((catalog_ids, -popularity[catalog_ids]))]
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
