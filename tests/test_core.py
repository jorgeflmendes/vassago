from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from vassago.config import ExperimentConfig
from vassago.data import (
    Interaction,
    Movie,
    check_metadata,
    exact_resolve,
    iter_examples,
    partition_data,
    synthetic,
    train_statistics,
)
from vassago.evaluation import (
    deterministic_rank,
    paired_bootstrap,
    paired_seed_bootstrap,
    ranking_metrics,
)
from vassago.models import LightGCN, SASRec
from vassago.retrieval import (
    AdaptiveGate,
    Calibrator,
    DenseIndex,
    fusion_features,
    union_candidates,
)
from vassago.semantic_ids import SIDExpert
from vassago.serving import history_tensor


def test_recommendation_gradient_reaches_sid_tokenizer() -> None:
    torch.manual_seed(42)
    model = SIDExpert(torch.randn(12, 16), 16, 2, 4)
    history = torch.tensor([[1, 2, 3], [3, 4, 5]])
    loss = model.recommendation_loss(history, torch.tensor([7, 8]), differentiable=True)
    loss.backward()
    for parameter in (model.tokenizer.codebooks, model.tokenizer.encoder.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 1e-8
    # No reconstruction loss was added: decoder gradients cannot satisfy this test.
    assert model.tokenizer.decoder.weight.grad is None


def test_sasrec_mask_padding_and_eval_determinism() -> None:
    torch.manual_seed(42)
    model = SASRec(10, 16, 8).eval()
    short = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[1, 2, 3, 0, 0]])
    torch.testing.assert_close(model(short), model(padded))
    torch.testing.assert_close(model(padded), model(padded))
    assert torch.isfinite(model(torch.zeros(2, 5, dtype=torch.long))).all()
    assert model.score(padded).shape == (1, 11)


def test_sid_codes_ranges_collisions_and_valid_generation() -> None:
    torch.manual_seed(2)
    model = SIDExpert(torch.ones(10, 8), 8, 2, 4).eval()
    codes = model.codes()
    assert codes.min() >= 0 and codes.max() < 4
    torch.testing.assert_close(codes, model.codes())
    result = model.decoder.generate(torch.zeros(8), codes, 10, 20)
    assert {r["movie_id"] for r in result} == set(range(1, 10))
    assert model.tokenizer.diagnostics(model.embeddings[1:])["colliding_items"] == 9


def test_data_aware_tokenizer_initialization_avoids_code_collapse() -> None:
    torch.manual_seed(7)
    vectors = torch.nn.functional.normalize(torch.randn(256, 32))
    model = SIDExpert(torch.cat([torch.zeros(1, 32), vectors]), 16, 2, 16)
    model.tokenizer.initialize(vectors, iterations=5)
    diagnostics = model.tokenizer.diagnostics(vectors)
    assert diagnostics["unique_codes"] > 128
    assert min(diagnostics["utilization"]) > 0.75


def test_metrics_hand_computed() -> None:
    values = ranking_metrics([3, 2, 1], {1, 2}, 3)
    assert values["Recall@3"] == 1
    assert values["MRR@3"] == 0.5
    assert values["MAP@3"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert values["NDCG@3"] == pytest.approx((1 / np.log2(3) + 0.5) / (1 + 1 / np.log2(3)))
    assert ranking_metrics([], {1}, 5)["NDCG@5"] == 0


def test_deterministic_rank_breaks_equal_scores_by_item_id() -> None:
    scores = np.array([0.0, 0.5, 0.5, 0.5])
    eligible = np.array([False, True, True, True])
    assert deterministic_rank(scores, 1, eligible) == 1
    assert deterministic_rank(scores, 3, eligible) == 3


def test_exact_crosswalk_and_namespaces() -> None:
    ids, report = exact_resolve(
        [
            {"tmdb_id": 5, "imdb_tconst": "tt0000001"},
            {"imdb_tconst": "tt0000001"},
            {"title": "Same"},
            {"title": "Same"},
        ]
    )
    assert ids == [1, 1, 2, 3]
    assert report[-1]["status"] == "unresolved"
    a = Interaction(user_id="movielens:123", movie_id=1, timestamp=1, rating=4)
    b = a.model_copy(update={"user_id": "amazon:123"})
    assert a.user_id != b.user_id
    with pytest.raises(ValueError):
        Interaction(user_id="123", movie_id=1, timestamp=1, rating=6)


def test_temporal_train_statistics_and_cold_holdout(tmp_path: Path) -> None:
    movies, frame = synthetic()
    split, bounds = partition_data(frame, (0.55, 0.65, 0.78, 0.88))
    path = tmp_path / "events.parquet"
    split.write_parquet(path)
    counts, _ = train_statistics(path, len(movies), 4, {1})
    assert counts[1] == 0
    expected = split.filter((pl.col("partition") == "train") & (pl.col("movie_id") != 1)).height
    assert counts.sum() == expected
    examples = list(iter_examples(path, movies, 4, 20, {1}))
    assert all(e.timestamp <= bounds[0] for e in examples if e.partition == "train")
    assert all(1 not in e.history and e.target != 1 for e in examples if e.partition == "train")
    assert examples == list(iter_examples(path, movies, 4, 20, {1}))


def test_availability_and_forbidden_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_metadata({"details": {"vote_count": 15}})
    pl.DataFrame(
        [{"user_id": "test:1", "movie_id": 1, "timestamp": 2, "rating": 4.0, "partition": "train"}]
    ).write_parquet(tmp_path / "data.parquet")
    with pytest.raises(ValueError, match="predates"):
        list(
            iter_examples(
                tmp_path / "data.parquet",
                [Movie(movie_id=1, title="x", available_at=3)],
                4,
                20,
                set(),
            )
        )


def test_gate_calibration_and_provenance() -> None:
    candidates = union_candidates({"id": [(1, 0.9), (1, 0.8)], "semantic": [(1, 0.5), (2, 0.4)]})
    assert len(candidates) == 2 and candidates[0].sources == ["id", "semantic"]
    assert candidates[0].ranks["id"] == 1
    features = fusion_features(np.zeros((2, 9), dtype=np.float32), np.zeros((2, 3)))
    assert features.shape == (2, 15)
    weights = AdaptiveGate()(torch.tensor(features), torch.tensor([[True, True, False]] * 2))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    assert (weights[:, 2] == 0).all()
    calibrator = Calibrator()
    with pytest.raises(ValueError):
        calibrator.fit([], [], "test")
    calibrator.fit([], [], "base_validation")
    normalized = calibrator.transform(np.ones((4, 3)))
    assert np.isfinite(normalized).all() and (normalized == normalized[0]).all()


def test_exact_index_and_paired_statistics() -> None:
    scores, ids = DenseIndex(np.eye(3, dtype=np.float32)).search(np.array([[0.0, 1, 0]]), 2)
    assert ids[0, 0] == 1 and scores[0, 0] == 1
    result = paired_bootstrap(np.ones(5), np.zeros(5), samples=100)
    assert result["mean_difference"] == result["ci_low"] == result["ci_high"] == 1

    seeded = paired_seed_bootstrap(
        {42: np.ones(5), 43: np.ones(5)},
        {42: np.zeros(5), 43: np.zeros(5)},
        samples=100,
    )
    assert seeded["mean_difference"] == seeded["ci_low"] == seeded["ci_high"] == 1
    with pytest.raises(ValueError, match="same users"):
        paired_seed_bootstrap(
            {42: np.ones(5), 43: np.ones(4)},
            {42: np.zeros(5), 43: np.zeros(4)},
            samples=100,
        )


def test_config_and_history_contract() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(dimension=9, heads=2)
    with pytest.raises(ValueError):
        ExperimentConfig(contextual_persistence_scales=9)
    with pytest.raises(ValueError):
        ExperimentConfig(contextual_velocity_scales=9)
    with pytest.raises(ValueError):
        ExperimentConfig(base_batch_size=0)
    assert history_tensor([[1, 2, 3]], 2, "cpu").tolist() == [[2, 3]]


def test_chunked_lightgcn_gradient_matches_full_objective() -> None:
    import copy

    from torch.nn import functional as F

    torch.manual_seed(4)
    model = LightGCN(2, 4, 8, torch.tensor([[1, 1, 2], [1, 2, 3]]))
    reference = copy.deepcopy(model)
    users, positives, negatives = torch.tensor([1, 2]), torch.tensor([1, 3]), torch.tensor([4, 2])
    reference.loss(users, positives, negatives).backward()
    representations = model.representations()
    leaves = tuple(v.detach().requires_grad_(True) for v in representations)
    for index in range(2):
        u, items = leaves
        loss = -F.logsigmoid(
            (u[users[index]] * (items[positives[index]] - items[negatives[index]])).sum()
        )
        loss.backward()
    torch.autograd.backward(representations, tuple(v.grad / 2 for v in leaves))
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
