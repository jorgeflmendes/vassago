"""Reusable preparation, embedding and saved-model evaluation APIs."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vassago.config import ExperimentConfig
from vassago.data import Movie, iter_examples
from vassago.evaluation import ranking_metrics
from vassago.experiment import write_json
from vassago.features import TextEncoder, movie_text
from vassago.serving import VassagoRanker
from vassago.sources import TMDBClient, enrich_tmdb


def build_embeddings(data: Path, config: ExperimentConfig, cutoff: int, output: Path) -> Path:
    movies = [Movie.model_validate(m) for m in json.loads((data / "catalog.json").read_text())]
    vectors = TextEncoder(config.encoder, config.dimension, config.encoder_revision).cached(
        [movie_text(movie, cutoff) for movie in movies], output
    )
    write_json(
        output / "catalog_manifest.json",
        {
            "movie_ids": [m.movie_id for m in movies],
            "cutoff": cutoff,
            "shape": list(vectors.shape),
            "encoder": config.encoder,
            "revision": config.encoder_revision,
        },
    )
    return output


def enrich_catalog(data: Path, output: Path, available_at: int, limit: int | None = None) -> None:
    """Explicit availability is mandatory; never silently backdate today's metadata."""
    if output.exists():
        raise FileExistsError(output)
    movies = [Movie.model_validate(m) for m in json.loads((data / "catalog.json").read_text())]
    client = TMDBClient(data / "tmdb_cache")
    enriched = []
    count = 0
    for movie in movies:
        if movie.tmdb_id and (limit is None or count < limit):
            movie = enrich_tmdb(movie, client.movie(movie.tmdb_id), available_at)
            count += 1
        enriched.append(movie.model_dump())
    write_json(output, enriched)


def evaluate_checkpoint(run_directory: Path, output: Path) -> dict[str, Any]:
    """Re-evaluate the selected final ensemble without refitting any component."""
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    model = VassagoRanker.load(run_directory / "model")
    cold = set(manifest["dataset_manifest"]["heldout_items"])
    examples = iter_examples(
        run_directory / "interactions.parquet",
        model.movies,
        model.config.positive_threshold,
        model.config.max_length,
        cold,
        "test",
    )
    rows = []
    for index, event in enumerate(examples):
        if model.config.evaluation_limit and index >= model.config.evaluation_limit:
            break
        result = model.retrieve(event.history, event.timestamp, event.seen)
        scores, _ = model.score(result)
        ranked = [result["ids"][i] for i in np.argsort(-scores, kind="stable")[:20]]
        rows.append(
            {
                "user_id": event.user_id,
                **{
                    key: value
                    for k in (5, 10, 20)
                    for key, value in ranking_metrics(ranked, {event.target}, k).items()
                },
            }
        )
    if not rows:
        raise ValueError("No test queries")
    frame = pl.DataFrame(rows)
    metrics = (
        frame.group_by("user_id")
        .agg(pl.exclude("user_id").mean())
        .select(pl.exclude("user_id").mean())
        .to_dicts()[0]
    )
    report = {"method": "adaptive", "queries": len(rows), "metrics": metrics}
    write_json(output, report)
    return report
