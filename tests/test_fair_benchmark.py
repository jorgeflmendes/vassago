import json
from pathlib import Path

import polars as pl
import pytest

from vassago.data import build_movielens
from vassago.fair_benchmark import evaluate_predictions, prepare_protocol


def _dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    raw.joinpath("movies.dat").write_text(
        "1::One (2000)::Drama\n"
        "2::Two (2000)::Comedy\n"
        "3::Three (2000)::Drama\n"
        "4::Four (2000)::Action\n"
        "5::Five (2000)::Drama\n"
        "6::Six (2000)::Comedy\n"
        "7::Seven (2000)::Action\n"
        "8::Eight (2000)::Drama\n",
        encoding="latin-1",
    )
    raw.joinpath("ratings.dat").write_text(
        "1::1::5::1\n1::2::4::2\n1::3::3::3\n1::4::5::4\n1::5::4::5\n1::6::5::6\n"
        "2::3::5::1\n2::4::4::2\n2::5::5::3\n2::6::3::4\n2::7::4::5\n2::8::5::6\n",
        encoding="ascii",
    )
    prepared = tmp_path / "prepared"
    build_movielens(raw, prepared)
    return prepared


def test_ml1m_protocol_and_strict_shared_evaluator(tmp_path: Path) -> None:
    prepared = _dataset(tmp_path)
    protocol_dir = prepare_protocol(prepared, tmp_path / "protocol", history_length=2)
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    assert protocol["query_count"] == 2
    assert protocol["item_count"] == 8
    queries = pl.read_parquet(protocol_dir / "queries.parquet")
    assert queries["history"].to_list() == [[4, 5], [6, 7]]

    def predictions(model: str, rankings: list[list[int]], path: Path) -> Path:
        rows = []
        for query_id, ranked in zip(queries["query_id"], rankings, strict=True):
            rows.extend(
                {
                    "query_id": query_id,
                    "model": model,
                    "seed": 42,
                    "protocol_hash": protocol["protocol_hash"],
                    "rank": rank,
                    "movie_id": movie_id,
                }
                for rank, movie_id in enumerate(ranked, 1)
            )
        pl.DataFrame(rows).write_parquet(path)
        return path

    a = predictions("a", [[6, 1, 2, 3, 7, 8], [8, 1, 2, 3, 4, 5]], tmp_path / "a.parquet")
    b = predictions("b", [[1, 6, 2, 3, 7, 8], [1, 8, 2, 3, 4, 5]], tmp_path / "b.parquet")
    report = evaluate_predictions(protocol_dir, [a, b], tmp_path / "comparison.json")
    result = json.loads(report.read_text())
    assert result["metrics"][0]["NDCG@10"] == 1
    assert result["metrics"][1]["NDCG@10"] < 1
    assert len(result["paired_significance"]) == 1

    invalid = predictions(
        "invalid", [[4, 1, 2, 3, 7, 8], [8, 1, 2, 3, 4, 5]], tmp_path / "invalid.parquet"
    )
    with pytest.raises(ValueError, match="already present"):
        evaluate_predictions(protocol_dir, [a, invalid], tmp_path / "invalid.json")
