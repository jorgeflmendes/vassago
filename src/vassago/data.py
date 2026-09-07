"""Exact source identities, streaming ingestion and chronological examples."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

PARTITIONS = ("train", "base_validation", "gate_train", "validation", "test")
FORBIDDEN_FEATURES = {"popularity", "vote_average", "vote_count", "averageRating", "numVotes"}


class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movie_id: int = Field(ge=1)
    source: str = "synthetic"
    movielens_movie_id: int | None = None
    tmdb_id: int | None = None
    imdb_tconst: str | None = None
    title: str
    available_at: int = Field(ge=0)
    metadata_available_at: int = Field(default=0, ge=0)
    genres: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*:.+$")
    movie_id: int = Field(ge=1)
    timestamp: int = Field(ge=0)
    rating: float = Field(ge=0.5, le=5)


class Example(BaseModel):
    user_id: str
    history: list[int]
    target: int
    timestamp: int
    rating: float
    partition: str
    seen: list[int] = Field(default_factory=list)


def check_metadata(metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if key in FORBIDDEN_FEATURES:
            raise ValueError(f"Historical metadata cannot include current aggregate: {key}")
        if isinstance(value, dict):
            check_metadata(value)


def validate_catalog(movies: list[Movie]) -> None:
    if [m.movie_id for m in movies] != list(range(1, len(movies) + 1)):
        raise ValueError("Catalog IDs must be contiguous, ordered, one-based; zero is padding")
    for movie in movies:
        check_metadata(movie.metadata)


def synthetic(seed: int = 42) -> tuple[list[Movie], pl.DataFrame]:
    rng = np.random.default_rng(seed)
    genres = ["Science Fiction", "Drama", "Comedy", "Thriller"]
    movies = [
        Movie(
            movie_id=i,
            title=f"Synthetic film {i}",
            available_at=0,
            genres=[genres[(i - 1) % 4]],
            metadata={
                "director": f"Director {(i - 1) % 4}",
                "overview": f"A {genres[(i - 1) % 4]} adventure",
            },
        )
        for i in range(1, 49)
    ]
    rows = []
    for user in range(24):
        favorite = user % 4
        favored = np.arange(favorite + 1, 49, 4)
        other = np.setdiff1d(np.arange(1, 49), favored)
        choices = np.concatenate([rng.permutation(favored), rng.permutation(other)[:12]])
        rng.shuffle(choices)
        for t, movie in enumerate(choices):
            rows.append(
                {
                    "user_id": f"synthetic:{user}",
                    "movie_id": int(movie),
                    "timestamp": 1000 + t * 100 + user,
                    "rating": 5.0 if (movie - 1) % 4 == favorite else 4.0,
                }
            )
    return movies, pl.DataFrame(rows).sort(["user_id", "timestamp", "movie_id"])


def build_movielens(source: Path, output: Path) -> None:
    """Stream ratings to Parquet; load only the small catalog into memory."""
    if (source / "u.data").exists():
        build_movielens_100k(source, output)
        return
    if (source / "ratings.dat").exists():
        build_movielens_1m(source, output)
        return
    output.mkdir(parents=True, exist_ok=True)
    movies = pl.read_csv(source / "movies.csv")
    links = pl.read_csv(source / "links.csv", schema_overrides={"imdbId": pl.String})
    if links["movieId"].n_unique() != links.height:
        raise ValueError("Duplicate MovieLens identities in links.csv")
    catalog = movies.join(links, on="movieId", how="left", validate="1:1").sort("movieId")
    catalog = catalog.with_row_index("movie_id", offset=1)
    ratings = pl.scan_csv(source / "ratings.csv")
    # Earliest observed availability is conservative when exact release dates are absent.
    first = ratings.group_by("movieId").agg(pl.col("timestamp").min().alias("available_at"))
    catalog = catalog.join(first.collect(engine="streaming"), on="movieId", how="left")
    records = []
    for row in catalog.iter_rows(named=True):
        imdb = row.get("imdbId")
        records.append(
            Movie(
                movie_id=row["movie_id"],
                source="movielens",
                movielens_movie_id=row["movieId"],
                tmdb_id=row.get("tmdbId"),
                imdb_tconst=f"tt{imdb.zfill(7)}" if imdb else None,
                title=row["title"],
                genres=row["genres"].split("|"),
                available_at=row["available_at"] if row["available_at"] is not None else 2**62,
            )
        )
    import json

    (output / "catalog.json").write_text(
        json.dumps([m.model_dump() for m in records]), encoding="utf-8"
    )
    crosswalk = catalog.select("movie_id", "movieId", "tmdbId", "imdbId")
    crosswalk.write_parquet(output / "entity_resolution_report.parquet")
    ratings.join(crosswalk.lazy().select("movieId", "movie_id"), on="movieId").select(
        pl.concat_str(pl.lit("movielens:"), pl.col("userId")).alias("user_id"),
        "movie_id",
        "timestamp",
        "rating",
    ).sort(["user_id", "timestamp", "movie_id"]).sink_parquet(output / "interactions.parquet")
    if (source / "tags.csv").exists():
        pl.scan_csv(source / "tags.csv").sink_parquet(output / "tags.parquet")


def build_movielens_1m(source: Path, output: Path) -> None:
    """Prepare ML-1M with a reversible dense ID map shared by external adapters."""
    import json

    output.mkdir(parents=True, exist_ok=True)
    raw = pl.read_csv(
        source / "ratings.dat",
        separator=":",
        has_header=False,
        new_columns=[
            "user",
            "empty_1",
            "source_movie_id",
            "empty_2",
            "rating",
            "empty_3",
            "timestamp",
        ],
    ).select("user", "source_movie_id", "rating", "timestamp")
    source_ids = sorted(raw["source_movie_id"].unique().to_list())
    mapping = pl.DataFrame(
        {"source_movie_id": source_ids, "movie_id": range(1, len(source_ids) + 1)}
    )
    first = dict(raw.group_by("source_movie_id").agg(pl.col("timestamp").min()).iter_rows())
    movie_rows: dict[int, tuple[str, list[str]]] = {}
    for line in (source / "movies.dat").read_text(encoding="latin-1").splitlines():
        source_id, title, genres = line.split("::", 2)
        movie_rows[int(source_id)] = (title, genres.split("|"))
    missing = set(source_ids) - movie_rows.keys()
    if missing:
        raise ValueError(f"Ratings reference {len(missing)} missing ML-1M movies")
    movies = [
        Movie(
            movie_id=index,
            source="movielens-1m",
            movielens_movie_id=source_id,
            title=movie_rows[source_id][0],
            available_at=first[source_id],
            genres=movie_rows[source_id][1],
        )
        for index, source_id in enumerate(source_ids, start=1)
    ]
    (output / "catalog.json").write_text(
        json.dumps([movie.model_dump() for movie in movies]), encoding="utf-8"
    )
    mapping.write_parquet(output / "entity_resolution_report.parquet")
    raw.join(mapping, on="source_movie_id", validate="m:1").select(
        pl.concat_str(pl.lit("movielens-1m:"), pl.col("user")).alias("user_id"),
        "movie_id",
        "timestamp",
        pl.col("rating").cast(pl.Float64),
    ).sort("user_id", "timestamp", "movie_id").write_parquet(output / "interactions.parquet")


def build_movielens_100k(source: Path, output: Path) -> None:
    """Stable legal development corpus; no title-based external identity guesses."""
    import json

    output.mkdir(parents=True, exist_ok=True)
    names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    ratings = pl.read_csv(
        source / "u.data",
        separator="\t",
        has_header=False,
        new_columns=["user", "movie_id", "rating", "timestamp"],
    )
    first = dict(ratings.group_by("movie_id").agg(pl.col("timestamp").min()).iter_rows())
    movies = []
    for line in (source / "u.item").read_text(encoding="latin-1").splitlines():
        row = line.split("|")
        movie_id = int(row[0])
        release = (
            int(datetime.strptime(row[2], "%d-%b-%Y").replace(tzinfo=UTC).timestamp())
            if row[2]
            else first[movie_id]
        )
        movies.append(
            Movie(
                movie_id=movie_id,
                source="movielens-100k",
                movielens_movie_id=movie_id,
                title=row[1],
                available_at=max(0, min(release, first[movie_id])),
                genres=[
                    genre for genre, active in zip(names, row[5:], strict=True) if active == "1"
                ],
            )
        )
    (output / "catalog.json").write_text(
        json.dumps([m.model_dump() for m in movies]), encoding="utf-8"
    )
    ratings.select(
        pl.concat_str(pl.lit("movielens-100k:"), pl.col("user")).alias("user_id"),
        "movie_id",
        "timestamp",
        pl.col("rating").cast(pl.Float64),
    ).sort("user_id", "timestamp", "movie_id").write_parquet(output / "interactions.parquet")
    pl.DataFrame(
        {"movie_id": [m.movie_id for m in movies], "status": ["unresolved"] * len(movies)}
    ).write_parquet(output / "entity_resolution_report.parquet")


def partition_data(
    frame: pl.DataFrame, quantiles: tuple[float, ...], cutoffs: tuple[int, ...] | None = None
) -> tuple[pl.DataFrame, list[int]]:
    if frame.is_empty():
        raise ValueError("Cannot split an empty interaction table")
    boundaries = (
        list(cutoffs)
        if cutoffs
        else [
            int(float(frame["timestamp"].quantile(q, interpolation="nearest") or 0))
            for q in quantiles
        ]
    )
    if len(set(boundaries)) != 4:
        raise ValueError("Insufficient distinct timestamps for five temporal windows")
    partition = pl.lit("test")
    for boundary, name in reversed(list(zip(boundaries, PARTITIONS, strict=False))):
        partition = pl.when(pl.col("timestamp") <= boundary).then(pl.lit(name)).otherwise(partition)
    return frame.with_columns(partition.alias("partition")), boundaries


def prepare_parquet(
    path: Path, output: Path, quantiles: tuple[float, ...], cutoffs: tuple[int, ...] | None = None
) -> list[int]:
    scan = pl.scan_parquet(path)
    bounds = (
        list(cutoffs)
        if cutoffs
        else [
            int(x)
            for x in scan.select(
                [
                    pl.col("timestamp").quantile(q, interpolation="nearest").alias(str(i))
                    for i, q in enumerate(quantiles)
                ]
            )
            .collect()
            .row(0)
        ]
    )
    if len(set(bounds)) != 4:
        raise ValueError("Insufficient distinct split boundaries")
    partition = pl.lit("test")
    for boundary, name in reversed(list(zip(bounds, PARTITIONS, strict=False))):
        partition = pl.when(pl.col("timestamp") <= boundary).then(pl.lit(name)).otherwise(partition)
    scan.with_columns(partition.alias("partition")).sort(
        ["user_id", "timestamp", "movie_id"]
    ).sink_parquet(output)
    return bounds


def iter_examples(
    path: Path,
    movies: list[Movie],
    threshold: float,
    max_length: int,
    cold_items: set[int],
    partition: str | None = None,
) -> Iterator[Example]:
    """User-sorted Parquet, bounded memory; simultaneous events never enter each other's history."""
    previous_user = ""
    history: list[int] = []
    seen: set[int] = set()
    pending_seen: list[int] = []
    pending: list[int] = []
    previous_time = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        for row in batch.to_pylist():
            event = Interaction.model_validate({k: row[k] for k in Interaction.model_fields})
            if event.user_id != previous_user:
                history, pending, previous_time = [], [], -1
                seen, pending_seen = set(), []
                previous_user = event.user_id
            if event.timestamp < previous_time:
                raise ValueError("Interactions must be sorted chronologically per user")
            if event.timestamp > previous_time:
                seen.update(pending_seen)
                pending_seen = []
                history = (history + pending)[-max_length:]
                pending = []
            previous_time = event.timestamp
            if event.movie_id > len(movies):
                raise ValueError("Interaction references unknown movie")
            if event.timestamp < movies[event.movie_id - 1].available_at:
                raise ValueError("Interaction predates movie availability")
            if row["partition"] == "train" and event.movie_id in cold_items:
                continue
            pending_seen.append(event.movie_id)
            if event.rating < threshold:
                continue
            if (partition is None or row["partition"] == partition) and event.movie_id not in seen:
                yield Example(
                    user_id=event.user_id,
                    history=history.copy(),
                    target=event.movie_id,
                    timestamp=event.timestamp,
                    rating=event.rating,
                    partition=row["partition"],
                    seen=sorted(seen),
                )
            pending.append(event.movie_id)


def train_statistics(
    path: Path, count: int, threshold: float, cold_items: set[int]
) -> tuple[np.ndarray, dict[str, int]]:
    scan = pl.scan_parquet(path).filter(
        (pl.col("partition") == "train")
        & (pl.col("rating") >= threshold)
        & ~pl.col("movie_id").is_in(sorted(cold_items))
    )
    counts = np.zeros(count + 1, dtype=np.float32)
    for movie_id, n in scan.group_by("movie_id").len().collect().iter_rows():
        counts[movie_id] = n
    users = dict(scan.group_by("user_id").len().collect().iter_rows())
    return counts, users


def exact_resolve(records: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    """Resolve exact IDs, explicitly reject inconsistent bridges; never match titles."""
    tmdb: dict[int, int] = {}
    imdb: dict[str, int] = {}
    result: list[int] = []
    report: list[dict[str, Any]] = []
    for record in records:
        a, b = record.get("tmdb_id"), record.get("imdb_tconst")
        matches = {
            v
            for v in (
                tmdb.get(a) if a is not None else None,
                imdb.get(b) if b is not None else None,
            )
            if v is not None
        }
        if len(matches) > 1:
            raise ValueError("Conflicting exact identifiers require manual resolution")
        identity = next(iter(matches)) if matches else len(set(result)) + 1
        if a is not None:
            tmdb[a] = identity
        if b is not None:
            imdb[b] = identity
        result.append(identity)
        report.append(
            {
                "internal_movie_id": identity,
                "tmdb_id": a,
                "imdb_tconst": b,
                "status": "exact" if a or b else "unresolved",
            }
        )
    return result, report
