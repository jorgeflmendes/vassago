import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch

from vassago.data import build_movielens
from vassago.fair_benchmark import (
    evaluate_predictions,
    prepare_protocol,
    validate_weight_provenance,
)
from vassago.fair_vassago import _sequences


def _causal_payloads(
    torch: Any, past_lengths: Any, past_payloads: dict[str, Any]
) -> dict[str, Any]:
    timestamps = past_payloads.get("timestamps")
    if timestamps is None:
        return past_payloads
    payloads = dict(past_payloads)
    causal_timestamps = timestamps.clone()
    rows = torch.arange(len(causal_timestamps), device=causal_timestamps.device)
    target_positions = past_lengths.clamp(min=1, max=causal_timestamps.shape[1] - 1)
    history_positions = (target_positions - 1).clamp_min(0)
    causal_timestamps[rows, target_positions] = causal_timestamps[rows, history_positions]
    payloads["timestamps"] = causal_timestamps
    return payloads


def test_hstu_adapter_censors_the_held_out_timestamp() -> None:
    payloads = {"timestamps": torch.tensor([[10, 20, 30, 99, 0]])}
    result = _causal_payloads(torch, torch.tensor([3]), payloads)
    assert result["timestamps"].tolist() == [[10, 20, 30, 30, 0]]
    assert payloads["timestamps"].tolist() == [[10, 20, 30, 99, 0]]


def test_sequence_reader_accepts_large_historical_rows(tmp_path: Path) -> None:
    values = ",".join(["1"] * 70_000)
    path = tmp_path / "sequences.csv"
    path.write_text(
        "user_id,sequence_item_ids,sequence_ratings,sequence_timestamps\n"
        f'user,"{values}","{values}","{values}"\n',
        encoding="utf-8",
    )
    assert len(_sequences(path)[0]["items"]) == 70_000


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
        "2::3::5::1\n2::4::4::2\n2::5::5::3\n2::1::3::4\n2::7::4::5\n2::8::5::6\n",
        encoding="ascii",
    )
    prepared = tmp_path / "prepared"
    build_movielens(raw, prepared)
    return prepared


def test_ml1m_protocol_and_strict_shared_evaluator(tmp_path: Path) -> None:
    prepared = _dataset(tmp_path)
    protocol_dir = prepare_protocol(prepared, tmp_path / "protocol", history_length=2)
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    assert protocol["schema_version"] == 3
    assert protocol["dataset"] == "MovieLens-1M"
    assert protocol["query_count"] == 2
    assert protocol["item_count"] == 8
    queries = pl.read_parquet(protocol_dir / "queries.parquet")
    assert queries["history"].to_list() == [[4, 5], [1, 7]]
    assert queries["history_timestamps"].to_list() == [[4, 5], [4, 5]]
    assert queries["seen"].to_list() == [[1, 2, 3, 4, 5], [1, 3, 4, 5, 7]]
    assert queries["timestamp"].to_list() == [5, 5]
    assert queries["target_timestamp"].to_list() == [6, 6]

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

    a = predictions("a", [[6, 7, 8], [8, 6, 2]], tmp_path / "a.parquet")
    b = predictions("b", [[7, 6, 8], [6, 8, 2]], tmp_path / "b.parquet")
    report = evaluate_predictions(protocol_dir, [a, b], tmp_path / "comparison.json")
    result = json.loads(report.read_text())
    assert result["metrics"][0]["NDCG@10"] == 1
    assert result["metrics"][1]["NDCG@10"] < 1
    assert set(result["metrics"][0]["catalog_metrics@10"]) == {
        "coverage",
        "long_tail_coverage",
        "average_popularity",
        "novelty",
        "genre_diversity",
    }
    assert len(result["paired_significance"]) == 1

    adapter_csv = tmp_path / "adapter-output.parquet"
    pl.read_parquet(b).write_csv(adapter_csv)
    adapter_report = evaluate_predictions(protocol_dir, [a, adapter_csv], tmp_path / "adapter.json")
    assert json.loads(adapter_report.read_text())["metrics"][1]["model"] == "b"

    invalid = predictions("invalid", [[4, 7, 8], [8, 1, 2]], tmp_path / "invalid.parquet")
    with pytest.raises(ValueError, match="already present"):
        evaluate_predictions(protocol_dir, [a, invalid], tmp_path / "invalid.json")

    duplicate_ranks = pl.read_parquet(b).with_columns(
        pl.when(pl.col("rank") == 2).then(1).otherwise(pl.col("rank")).alias("rank")
    )
    duplicate_ranks.write_parquet(tmp_path / "duplicate-ranks.parquet")
    with pytest.raises(ValueError, match="duplicate ranks"):
        evaluate_predictions(
            protocol_dir, [a, tmp_path / "duplicate-ranks.parquet"], tmp_path / "ranks.json"
        )


def test_global_temporal_protocol_excludes_future_training_events(tmp_path: Path) -> None:
    prepared = _dataset(tmp_path)
    protocol_dir = prepare_protocol(
        prepared, tmp_path / "global-protocol", history_length=2, global_test_start=4
    )
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    assert protocol["schema_version"] == 4
    assert protocol["candidate_item_ids"] == [1, 2, 3, 4, 5]
    assert protocol["training_sequence_sha256"]
    assert protocol["training_interactions_sha256"]

    queries = pl.read_parquet(protocol_dir / "queries.parquet")
    assert queries.select("query_id", "history", "target", "timestamp").to_dicts() == [
        {"query_id": "movielens-1m:1", "history": [2, 3], "target": 4, "timestamp": 3},
        {"query_id": "movielens-1m:2", "history": [4, 5], "target": 1, "timestamp": 3},
    ]
    assert pl.read_parquet(protocol_dir / "training_interactions.parquet")["timestamp"].max() < 4
    training_rows = (protocol_dir / "hstu_training_sequences.csv").read_text().splitlines()
    assert len(training_rows) == 3


def test_global_temporal_query_sampling_is_stable_and_preserves_training(tmp_path: Path) -> None:
    prepared = _dataset(tmp_path)
    first = prepare_protocol(
        prepared,
        tmp_path / "first",
        history_length=2,
        global_test_start=4,
        query_sample_limit=1,
        query_sample_seed=7,
    )
    second = prepare_protocol(
        prepared,
        tmp_path / "second",
        history_length=2,
        global_test_start=4,
        query_sample_limit=1,
        query_sample_seed=7,
    )
    first_protocol = json.loads((first / "protocol.json").read_text())
    second_protocol = json.loads((second / "protocol.json").read_text())
    assert first_protocol["query_sample_limit"] == 1
    assert first_protocol["query_sample_seed"] == 7
    assert first_protocol["queries_sha256"] == second_protocol["queries_sha256"]
    assert first_protocol["training_sequence_sha256"] == second_protocol["training_sequence_sha256"]


def test_weight_provenance_requires_test_blind_selection(tmp_path: Path) -> None:
    selection_weights = tmp_path / "selection.safetensors"
    final_weights = tmp_path / "final.safetensors"
    selection_weights.write_bytes(b"selection")
    final_weights.write_bytes(b"final")
    selection_manifest = tmp_path / "selection.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "test_evaluated": False,
                "protocol_hash": "protocol",
                "selection_weights": {
                    "sha256": hashlib.sha256(selection_weights.read_bytes()).hexdigest()
                },
            }
        )
    )
    final_manifest = tmp_path / "final.json"
    final_manifest.write_text(
        json.dumps(
            {
                "protocol_hash": "protocol",
                "weights": {"sha256": hashlib.sha256(final_weights.read_bytes()).hexdigest()},
            }
        )
    )
    validate_weight_provenance(
        selection_weights,
        selection_manifest,
        final_weights,
        final_manifest,
        "protocol",
    )
    payload = json.loads(selection_manifest.read_text())
    payload["test_evaluated"] = True
    selection_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="test was not evaluated"):
        validate_weight_provenance(
            selection_weights,
            selection_manifest,
            final_weights,
            final_manifest,
            "protocol",
        )
