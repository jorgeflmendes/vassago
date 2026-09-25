from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from vassago.config import ExperimentConfig
from vassago.contextual_ranker import ContextualEvidenceRanker, _query_timestamp_tensor
from vassago.data import Interaction, synthetic
from vassago.hybrid_serving import ParetoHybridRecommender
from vassago.serving import RecommendationContext, UserPreferenceProfile


def _hybrid(counts: np.ndarray | None = None) -> ParetoHybridRecommender:
    torch.manual_seed(9)
    movies, _ = synthetic()
    config = ExperimentConfig(
        dimension=8,
        heads=2,
        layers=1,
        max_length=6,
        dropout=0.0,
        contextual_dimension=4,
        contextual_memory_window=4,
        contextual_groups=2,
    )
    model = ContextualEvidenceRanker(48, 8, 6, 2, 1, 0.0, 4, 4, 0.2)
    return ParetoHybridRecommender(
        model,
        config,
        movies,
        np.ones(49) if counts is None else counts,
    )


def test_profile_cold_start_prefers_requested_metadata() -> None:
    recommender = _hybrid()
    rows = recommender.recommend(
        [],
        8,
        UserPreferenceProfile(genres=["Science Fiction"]),
        RecommendationContext(timestamp=10_000),
    )
    assert rows
    assert all(row.source == "profile" for row in rows)
    assert all("genre:science fiction" in row.matched_features for row in rows)


def test_warm_scores_are_invariant_to_profile() -> None:
    recommender = _hybrid()
    history = [
        Interaction(user_id="synthetic:1", movie_id=2, timestamp=100, rating=5),
        Interaction(user_id="synthetic:1", movie_id=6, timestamp=200, rating=5),
    ]
    context = RecommendationContext(timestamp=300)
    first = recommender.recommend(
        history,
        12,
        UserPreferenceProfile(genres=["Drama"], favorite_movie_ids=[3, 4]),
        context,
    )
    second = recommender.recommend(history, 12, UserPreferenceProfile(genres=["Comedy"]), context)
    assert [row.movie_id for row in first] == [row.movie_id for row in second]
    assert [row.score for row in first] == [row.score for row in second]
    assert all(row.source == "contextual" for row in first)
    ids = torch.tensor([[2, 6]])
    timestamps = torch.tensor([[100, 200]])
    query_timestamps = _query_timestamp_tensor(timestamps, torch.tensor([2]), torch.tensor([300]))
    _, expected = recommender.model.score(
        ids,
        timestamps=timestamps,
        query_timestamps=query_timestamps,
        inference_chunk_size=recommender.inference_attention_chunk_size,
    )
    assert [row.score for row in first] == [float(expected[0, row.movie_id]) for row in first]


def test_onboarding_favorites_use_contextual_checkpoint() -> None:
    recommender = _hybrid()
    rows = recommender.recommend(
        [],
        5,
        UserPreferenceProfile(favorite_movie_ids=[1, 5, 9]),
        RecommendationContext(timestamp=10_000),
    )
    assert rows
    assert all(row.source == "onboarding" for row in rows)
    assert not ({1, 5, 9} & {row.movie_id for row in rows})


def test_onboarding_can_use_a_separate_cold_adapter() -> None:
    warm = _hybrid()
    cold = _hybrid().model
    recommender = ParetoHybridRecommender(
        warm.model,
        warm.config,
        warm.movies,
        np.ones(49),
        cold_model=cold,
        onboarding_centroid_weight=0.0,
    )
    rows = recommender.recommend(
        [],
        5,
        UserPreferenceProfile(favorite_movie_ids=[1, 5, 9]),
        RecommendationContext(timestamp=10_000),
    )
    ids = torch.tensor([[1, 5, 9]])
    expected = cold.score(ids, inference_chunk_size=recommender.inference_attention_chunk_size)[1][
        0
    ]
    assert [row.score for row in rows] == [float(expected[row.movie_id]) for row in rows]


def test_unknown_profile_uses_popularity_fallback() -> None:
    rows = _hybrid(np.arange(49)).recommend(
        [],
        3,
        UserPreferenceProfile(directors=["Unknown Person"]),
        RecommendationContext(timestamp=10_000),
    )
    assert [row.movie_id for row in rows] == [48, 47, 46]
    assert all(row.source == "popularity" for row in rows)


def test_final_contextual_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    original = _hybrid()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(original.config.model_dump()), encoding="utf-8")
    weights_path = tmp_path / "model.safetensors"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in original.model.state_dict().items()
        },
        weights_path,
    )
    restored = ParetoHybridRecommender.load(
        config_path,
        weights_path,
        original.movies,
        np.ones(49),
    )
    assert restored.model.backbone.items.num_embeddings == 49
