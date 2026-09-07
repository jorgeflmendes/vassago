import json
from pathlib import Path

import polars as pl
import pytest

from vassago.data import build_movielens
from vassago.sources import amazon_reviews, convenience_kaggle_pipeline, cutoff_tags, imdb_rows


def test_movielens_csv_mapping_and_original_ratings(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "movies.csv").write_text("movieId,title,genres\n10,Film A,Drama\n20,Film B,Comedy\n")
    (source / "links.csv").write_text("movieId,imdbId,tmdbId\n10,0000123,456\n20,,\n")
    (source / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n3,20,2.5,200\n3,10,5,100\n"
    )
    output = tmp_path / "processed"
    build_movielens(source, output)
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog[0]["tmdb_id"] == 456 and catalog[0]["imdb_tconst"] == "tt0000123"
    assert catalog[1]["imdb_tconst"] is None
    frame = pl.read_parquet(output / "interactions.parquet")
    assert frame["timestamp"].to_list() == [100, 200]
    assert frame["rating"].to_list() == [5, 2.5]
    assert frame["user_id"].to_list() == ["movielens:3", "movielens:3"]


def test_source_adapters_preserve_identity_and_cutoffs(tmp_path: Path) -> None:
    imdb = tmp_path / "title.basics.tsv"
    imdb.write_text("tconst\tprimaryTitle\tstartYear\ntt0000001\tFilm\t\\N\n")
    assert list(imdb_rows(imdb))[0]["startYear"] is None
    tags = tmp_path / "tags.parquet"
    pl.DataFrame({"timestamp": [10, 20], "tag": ["past", "future"]}).write_parquet(tags)
    assert cutoff_tags(tags, 10).collect()["tag"].to_list() == ["past"]
    amazon = tmp_path / "amazon.jsonl"
    amazon.write_text(
        json.dumps({"user_id": "3", "parent_asin": "PRODUCT", "timestamp": 10000, "rating": 4})
    )
    assert list(amazon_reviews(amazon))[0]["user_id"] == "amazon:3"
    kaggle = tmp_path / "movies.csv"
    kaggle.write_text("id,title,vote_average\n7,Film,9\n")
    assert "vote_average" not in convenience_kaggle_pipeline(kaggle).collect_schema()


def test_kaggle_never_falls_back_to_title_matching(tmp_path: Path) -> None:
    path = tmp_path / "movies.csv"
    path.write_text("title\nFilm\n")
    with pytest.raises(ValueError, match="exact"):
        convenience_kaggle_pipeline(path)
