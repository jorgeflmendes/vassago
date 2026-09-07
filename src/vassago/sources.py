"""User-triggered official downloads and isolated optional source adapters."""

import csv
import gzip
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from vassago.data import Movie


def download_movielens(release: str, output: Path) -> Path:
    if release not in {"ml-32m", "ml-1m", "ml-100k"}:
        raise ValueError("Only stable MovieLens releases are supported")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{release}.zip"
    if archive.exists():
        raise FileExistsError("Archive exists; verify its manifest before reusing it")
    url = f"https://files.grouplens.org/datasets/movielens/{release}.zip"
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("xb") as stream:
        while block := response.read(1024 * 1024):
            digest.update(block)
            stream.write(block)
    root = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or member.file_size > 10_000_000_000:
                raise ValueError("Unsafe archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not permitted")
        bundle.extractall(root)
    (output / f"{release}.manifest.json").write_text(
        json.dumps(
            {
                "source": url,
                "release": release,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "sha256": digest.hexdigest(),
                "terms": "https://grouplens.org/datasets/movielens/",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / release


class TMDBClient:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("TMDB_READ_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("Set TMDB_READ_ACCESS_TOKEN in the environment")

    def movie(self, tmdb_id: int) -> dict[str, Any]:
        if isinstance(tmdb_id, bool) or not isinstance(tmdb_id, int) or tmdb_id < 1:
            raise ValueError("tmdb_id must be a positive integer")
        path = self.cache / f"{tmdb_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=credits,keywords"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                if payload.get("id") != tmdb_id:
                    raise ValueError("TMDB response identifier mismatch")
                # Deliberate allowlist excludes live vote/popularity aggregates and image downloads.
                selected = {
                    key: payload.get(key)
                    for key in (
                        "id",
                        "title",
                        "original_title",
                        "overview",
                        "genres",
                        "release_date",
                        "credits",
                        "keywords",
                        "production_countries",
                        "original_language",
                        "belongs_to_collection",
                    )
                }
                selected["retrieved_at"] = datetime.now(UTC).isoformat()
                path.write_text(json.dumps(selected), encoding="utf-8")
                return selected
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise RuntimeError(f"TMDB request failed (HTTP {error.code})") from None
                time.sleep(2**attempt)
        raise RuntimeError("TMDB retries exhausted")


def enrich_tmdb(movie: Movie, payload: dict[str, Any], metadata_available_at: int) -> Movie:
    if movie.tmdb_id != payload.get("id"):
        raise ValueError("Exact TMDB identifier required for enrichment")
    credits = payload.get("credits") or {}
    crew = credits.get("crew", [])
    metadata = {
        key: payload[key]
        for key in ("original_title", "overview", "original_language")
        if payload.get(key)
    }
    for name, jobs in {
        "director": {"Director"},
        "writers": {"Writer", "Screenplay", "Story"},
        "cinematographer": {"Director of Photography"},
        "composer": {"Original Music Composer"},
    }.items():
        metadata[name] = sorted({person["name"] for person in crew if person.get("job") in jobs})
    metadata["cast"] = [person["name"] for person in credits.get("cast", [])[:10]]
    metadata["keywords"] = [
        item["name"] for item in (payload.get("keywords") or {}).get("keywords", [])
    ]
    metadata["production_countries"] = [
        item["name"] for item in payload.get("production_countries") or []
    ]
    return Movie.model_validate(
        {
            **movie.model_dump(),
            "metadata": metadata,
            "metadata_available_at": metadata_available_at,
            "genres": [item["name"] for item in payload.get("genres") or []],
        }
    )


def imdb_rows(path: Path) -> Iterator[dict[str, str | None]]:
    """Stream official TSV(.gz); caller selects static fields, never current ratings."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            yield {key: None if value == r"\N" else value for key, value in row.items()}


def download_imdb(table: str, output: Path) -> Path:
    if table not in {
        "title.basics",
        "title.crew",
        "title.principals",
        "title.ratings",
        "name.basics",
        "title.akas",
    }:
        raise ValueError("Unsupported official IMDb table")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{table}.tsv.gz"
    with urllib.request.urlopen(f"https://datasets.imdbws.com/{path.name}", timeout=60) as response:
        with path.open("xb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    return path


def genome_scores(path: Path) -> pl.LazyFrame:
    """Normalized score-table adapter; preserve all relevance values and exact IDs."""
    scan = pl.scan_csv(path)
    schema = scan.collect_schema()
    renames = {
        old: new for old, new in (("movieId", "movie_id"), ("tagId", "tag_id")) if old in schema
    }
    scan = scan.rename(renames)
    if not {"movie_id", "tag_id", "relevance"}.issubset(scan.collect_schema().names()):
        raise ValueError("Genome input requires movie_id, tag_id, relevance")
    return scan.select("movie_id", "tag_id", "relevance")


def cutoff_tags(path: Path, cutoff: int) -> pl.LazyFrame:
    return pl.scan_parquet(path).filter(pl.col("timestamp") <= cutoff)


def convenience_kaggle_pipeline(path: Path) -> pl.LazyFrame:
    scan = pl.scan_csv(path, infer_schema_length=10000)
    columns = scan.collect_schema().names()
    if "id" not in columns:
        raise ValueError("Expected exact TMDB id column; title matching is prohibited")
    allowed = [
        c
        for c in (
            "id",
            "imdb_id",
            "title",
            "overview",
            "genres",
            "release_date",
            "original_language",
        )
        if c in columns
    ]
    return scan.select(allowed).with_columns(pl.lit("convenience_kaggle_pipeline").alias("source"))


def amazon_reviews(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield {
                "user_id": f"amazon:{row['user_id']}",
                "product_id": row["parent_asin"],
                "timestamp": int(row["timestamp"]) // 1000,
                "rating": float(row["rating"]),
                "source": "amazon-reviews-2023:Movies_and_TV",
            }


def beliefs_table(path: Path, user_column: str) -> pl.LazyFrame:
    """Keep experimental belief tables separate; no implicit core benchmark joins."""
    scan = pl.scan_csv(path)
    if user_column not in scan.collect_schema():
        raise ValueError("Specify the source table's exact user identifier column")
    return scan.with_columns(
        pl.concat_str(pl.lit("movielens-beliefs:"), pl.col(user_column)).alias("user_id")
    )
