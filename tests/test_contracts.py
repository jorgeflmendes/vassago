from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from vassago.config import ExperimentConfig
from vassago.data import Movie, iter_examples, synthetic
from vassago.features import TextEncoder, movie_text
from vassago.retrieval import Calibrator, mmr
from vassago.serving import RecommendationContext, UserPreferenceProfile, VassagoRanker


def small_model() -> VassagoRanker:
    config = ExperimentConfig()
    movies, _ = synthetic()
    encoded = TextEncoder("hash", 32).encode([movie_text(m, 0) for m in movies])
    embeddings = np.vstack([np.zeros((1, 32)), encoded]).astype(np.float32)
    model = VassagoRanker(config, movies, embeddings, np.array([0] + [1] * len(movies)))
    model.calibrator.fit([], [], "base_validation")
    model.eval()
    return model


def test_semantic_finetuning_cannot_mutate_sid_embeddings() -> None:
    model = small_model()
    before = model.sid_expert.embeddings.clone()
    with torch.no_grad():
        model.semantic_expert.items.weight.add_(1)
    torch.testing.assert_close(model.sid_expert.embeddings, before)
    assert model.id_expert.items.weight.data_ptr() != model.semantic_expert.items.weight.data_ptr()


def test_new_movie_semantic_sid_and_checkpoint(tmp_path: Path) -> None:
    model = small_model()
    model.add_movie(
        Movie(movie_id=49, title="New film", available_at=0, genres=["Drama"]),
        model.embeddings[1].copy(),
    )
    assert model.counts[49] == 0
    assert model.eligible([], 10)[49]
    assert model.sid_expert.codes().shape[0] == 49
    model.weights = np.array([1.0, 0, 0])
    result = {"ids": [49], "scores": np.ones((1, 3))}
    _, weights = model.score(result, "weighted")
    np.testing.assert_equal(weights, [[0, 1, 0]])
    model.save(tmp_path / "model")
    restored = VassagoRanker.load(tmp_path / "model")
    assert len(restored.movies) == 49 and restored.counts[49] == 0


def test_simultaneous_events_and_seen_set_survive_truncation(tmp_path: Path) -> None:
    events = [
        {
            "user_id": "test:1",
            "movie_id": i,
            "rating": 4.0,
            "timestamp": timestamp,
            "partition": "train",
        }
        for i, timestamp in [(1, 10), (2, 10), (3, 20), (4, 30), (1, 40)]
    ]
    path = tmp_path / "events.parquet"
    pl.DataFrame(events).write_parquet(path)
    movies = [Movie(movie_id=i, title=str(i), available_at=0) for i in range(1, 5)]
    examples = list(iter_examples(path, movies, 4, 1, set()))
    assert examples[0].history == examples[1].history == []
    assert examples[2].history == [2] and examples[2].seen == [1, 2]
    assert len(examples) == 4


def test_temporal_history_contract_rejects_future() -> None:
    from vassago.data import Interaction

    model = small_model()
    with pytest.raises(ValueError, match="strictly precede"):
        model.recommend(
            [Interaction(user_id="test:1", movie_id=1, timestamp=20, rating=5)],
            context=RecommendationContext(timestamp=20),
        )
    with pytest.raises(ValueError, match="ordered"):
        UserPreferenceProfile(years=(2026, 1900))
    with pytest.raises(ValueError, match="unique"):
        UserPreferenceProfile(favorite_movie_ids=[1, 1])


def test_calibration_variants_and_mmr() -> None:
    scores = [np.array([[1.0, 2, 3], [3.0, 2, 1]])]
    for method in ("rank", "zscore", "temperature"):
        calibrator = Calibrator(method)
        calibrator.fit(scores, [0], "base_validation")
        assert np.isfinite(calibrator.transform(scores[0])).all()
        restored = Calibrator.from_state(calibrator.state())
        np.testing.assert_equal(restored.transform(scores[0]), calibrator.transform(scores[0]))
    vectors = np.array([[0.0, 0], [1, 0], [1, 0], [0, 1]])
    assert mmr([1, 2, 3], np.array([1.0, 0.9, 0.8]), vectors, 2, 0.1) == [1, 3]


def test_no_preference_user_uses_train_popularity() -> None:
    model = small_model()
    model.counts[7] = 100
    rows = model.recommend([], 3, context=RecommendationContext(timestamp=20))
    assert rows[0].movie_id == 7
    assert rows[0].candidate_sources == ["popularity"]
