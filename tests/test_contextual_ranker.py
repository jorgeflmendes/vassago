import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import vassago.contextual_ranker as contextual_ranker
from vassago.config import ExperimentConfig
from vassago.contextual_ranker import (
    ContextualEvidenceRanker,
    _apply_evidence_scale,
    _config_fingerprint,
    _exclude_seen_all_positions,
    _exclude_seen_selected_positions,
    _fit_context,
    _interpolate_state,
    _query_timestamp_tensor,
    _read_selection_recipe,
    _select_evidence_scale,
    _sequence_tensors_with_timestamps,
    freeze_selection_recipe,
)


def _model() -> ContextualEvidenceRanker:
    torch.manual_seed(7)
    return ContextualEvidenceRanker(12, 8, 6, 2, 1, 0.0, 4, 4, 0.2)


def _persistence_model() -> ContextualEvidenceRanker:
    torch.manual_seed(7)
    return ContextualEvidenceRanker(12, 8, 6, 2, 1, 0.0, 4, 4, 0.2, 1, 3)


def _velocity_model() -> ContextualEvidenceRanker:
    torch.manual_seed(7)
    return ContextualEvidenceRanker(12, 8, 6, 2, 1, 0.0, 4, 4, 0.2, 1, 0, 3)


def test_fixed_selection_recipe_requires_a_separate_development_protocol(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    config = ExperimentConfig()
    recipe_path.write_text(
        f"""{{
  "schema_version": 1,
  "source_protocol_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "selected_base_epoch": 2,
  "selected_context_epoch": 1,
  "selected_joint_epoch": 0,
  "selected_evidence_scale": 1.0,
  "selected_original_weight": 1.0,
  "config_fingerprint": "{_config_fingerprint(config)}"
}}
"""
    )
    assert _read_selection_recipe(recipe_path, "b" * 64, config).selected_base_epoch == 2
    assert (
        _read_selection_recipe(
            recipe_path, "b" * 64, config.model_copy(update={"seed": 43})
        ).selected_base_epoch
        == 2
    )
    with pytest.raises(ValueError, match="separate development"):
        _read_selection_recipe(recipe_path, "a" * 64, config)
    with pytest.raises(ValueError, match="configured architecture"):
        _read_selection_recipe(recipe_path, "b" * 64, ExperimentConfig(dimension=16))


def test_freeze_selection_recipe_requires_test_blind_matching_validation(tmp_path: Path) -> None:
    config = ExperimentConfig()
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "test_evaluated": False,
                "protocol_hash": "a" * 64,
                "config": config.model_dump(),
                "selected_base_epoch": 2,
                "selected_context_epoch": 1,
                "selected_joint_epoch": 0,
                "selected_evidence_scale": 1.0,
                "selected_original_weight": 1.0,
            }
        ),
        encoding="utf-8",
    )
    recipe_path = freeze_selection_recipe(manifest_path, config, tmp_path / "recipe.json")
    assert _read_selection_recipe(recipe_path, "b" * 64, config).selected_base_epoch == 2
    payload = json.loads(manifest_path.read_text())
    payload["test_evaluated"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test-blind"):
        freeze_selection_recipe(manifest_path, config, tmp_path / "invalid.json")


def test_sampled_scores_match_catalog_scores_for_last_position() -> None:
    model = _model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    targets = torch.tensor([4])
    negatives = torch.tensor([[5, 6]])
    base, contextual = model.sampled_logits(history, targets, negatives, torch.tensor([2]))
    full_base, full_contextual = model.score(history)
    torch.testing.assert_close(base[0] * 0.05, full_base[0, [4, 5, 6]])
    torch.testing.assert_close(contextual[0] * 0.05, full_contextual[0, [4, 5, 6]])
    assert not full_base.requires_grad
    assert not full_contextual.requires_grad


def test_contextual_loss_reaches_memory_and_candidate_projections() -> None:
    model = _model().train()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    _, contextual = model.sampled_logits(
        history, torch.tensor([4]), torch.tensor([[5, 6]]), torch.tensor([2])
    )
    F.cross_entropy(contextual, torch.zeros(1, dtype=torch.long)).backward()
    for parameter in (
        model.key_projections[0].weight,
        model.query_projections[0].weight,
        model.salience_heads[0].weight,
        model.gamma_unconstrained,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_temporal_partition_bounds_contain_exact_evidence() -> None:
    model = _model().eval()
    history = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]])
    candidates = torch.tensor([[7, 8, 9], [7, 8, 9]])
    exact, lower, upper = model.evidence_bounds(history, candidates, groups=2)
    assert torch.all(lower <= exact + 1e-6)
    assert torch.all(exact <= upper + 1e-6)


def test_singleton_groups_recover_exact_evidence() -> None:
    model = _model().eval()
    history = torch.tensor([[1, 2, 3, 4, 0, 0]])
    candidates = torch.tensor([[7, 8]])
    exact, lower, upper = model.evidence_bounds(history, candidates, groups=4)
    torch.testing.assert_close(lower, exact, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(upper, exact, atol=1e-6, rtol=1e-6)


def test_multiple_evidence_heads_preserve_sampled_catalog_contract() -> None:
    model = ContextualEvidenceRanker(12, 8, 6, 2, 1, 0.0, 4, 4, 0.2, 3).eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    targets = torch.tensor([4])
    negatives = torch.tensor([[5, 6]])
    _, sampled = model.sampled_logits(history, targets, negatives, torch.tensor([2]))
    _, catalog = model.score(history)
    torch.testing.assert_close(sampled[0] * 0.05, catalog[0, [4, 5, 6]])


def test_persistence_scores_match_catalog_scores() -> None:
    model = _persistence_model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 400, 1_000, 0, 0, 0]])
    targets = torch.tensor([4])
    negatives = torch.tensor([[5, 6]])
    _, sampled = model.sampled_logits(
        history,
        targets,
        negatives,
        torch.tensor([2]),
        timestamps=timestamps,
        query_timestamps=timestamps,
    )
    _, catalog = model.score(history, timestamps=timestamps, query_timestamps=timestamps)
    torch.testing.assert_close(sampled[0] * 0.05, catalog[0, [4, 5, 6]])


def test_persistence_gradient_reaches_routing_decay_and_decoder() -> None:
    model = _persistence_model().train()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 400, 1_000, 0, 0, 0]])
    _, contextual = model.sampled_logits(
        history,
        torch.tensor([4]),
        torch.tensor([[5, 6]]),
        torch.tensor([2]),
        timestamps=timestamps,
        query_timestamps=timestamps,
    )
    F.cross_entropy(contextual, torch.zeros(1, dtype=torch.long)).backward()
    for parameter in (
        model.persistence_router.weight,
        model.persistence_candidate_gate.weight,
        model.persistence_history_gate.weight,
        model.persistence_half_life_increments,
        model.persistence_gamma_unconstrained,
        model.persistence_disagreement_unconstrained,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_persistence_is_invariant_to_an_absolute_timestamp_offset() -> None:
    model = _persistence_model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 400, 1_000, 0, 0, 0]])
    shifted = torch.tensor([[1_100, 1_400, 2_000, 0, 0, 0]])
    _, original = model.score(history, timestamps=timestamps, query_timestamps=timestamps)
    _, translated = model.score(history, timestamps=shifted, query_timestamps=shifted)
    torch.testing.assert_close(original, translated)


def test_persistence_disables_evidence_bounds() -> None:
    with pytest.raises(ValueError, match="persistence scales"):
        _persistence_model().evidence_bounds(
            torch.tensor([[1, 2, 3, 0, 0, 0]]), torch.tensor([[4, 5]]), groups=2
        )


def test_persistence_empty_history_has_finite_base_scores() -> None:
    base, contextual = _persistence_model().eval().score(torch.zeros(1, 6, dtype=torch.long))
    assert torch.isfinite(contextual).all()
    torch.testing.assert_close(contextual, base)


def test_velocity_scores_match_catalog_scores() -> None:
    model = _velocity_model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 400, 1_000, 0, 0, 0]])
    _, sampled = model.sampled_logits(
        history,
        torch.tensor([4]),
        torch.tensor([[5, 6]]),
        torch.tensor([2]),
        timestamps=timestamps,
        query_timestamps=timestamps,
    )
    _, catalog = model.score(history, timestamps=timestamps, query_timestamps=timestamps)
    torch.testing.assert_close(sampled[0] * 0.05, catalog[0, [4, 5, 6]])


def test_velocity_gradient_reaches_track_parameters() -> None:
    model = _velocity_model().train()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 400, 1_000, 0, 0, 0]])
    _, contextual = model.sampled_logits(
        history,
        torch.tensor([4]),
        torch.tensor([[5, 6]]),
        torch.tensor([2]),
        timestamps=timestamps,
        query_timestamps=timestamps,
    )
    F.cross_entropy(contextual, torch.zeros(1, dtype=torch.long)).backward()
    for parameter in (
        model.velocity_router.weight,
        model.velocity_candidate_gate.weight,
        model.velocity_history_gate.weight,
        model.velocity_half_life_increments,
        model.velocity_gamma_unconstrained,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_velocity_single_event_history_has_finite_scores() -> None:
    model = _velocity_model().eval()
    base, contextual = model.score(
        torch.tensor([[1, 0, 0, 0, 0, 0]]),
        timestamps=torch.tensor([[100, 0, 0, 0, 0, 0]]),
    )
    assert torch.isfinite(contextual).all()
    torch.testing.assert_close(contextual, base)


def test_evidence_scale_selection_uses_registered_ndcg10(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_metrics(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        scale = float(_args[-1])
        return {
            "NDCG@10": {0.5: 0.2, 0.75: 0.3, 1.0: 0.25, 1.25: 0.1, 1.5: 0.1, 2.0: 0.1}[scale],
            "Recall@50": float(scale),
        }

    monkeypatch.setattr(contextual_ranker, "_validation_cutoff_metrics", fake_metrics)
    selected, _ = _select_evidence_scale(_model(), [], {}, ExperimentConfig())
    assert selected == 0.75


def test_evidence_scale_folds_into_gamma_without_new_parameters() -> None:
    model = _model()
    before = model.gamma.detach().clone()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _apply_evidence_scale(model, 1.5)
    torch.testing.assert_close(model.gamma, before * 1.5)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_weight_soup_interpolates_one_checkpoint_without_extra_state() -> None:
    original = {"weight": torch.tensor([1.0, 3.0])}
    adapted = {"weight": torch.tensor([5.0, 7.0])}
    result = _interpolate_state(original, adapted, 0.25)
    torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))


def test_negative_resampling_excludes_only_causally_seen_items() -> None:
    torch.manual_seed(3)
    history = torch.tensor([[2, 4, 6, 0]])
    candidates = torch.tensor([[[2, 4], [2, 4], [2, 4], [2, 4]]])
    result = _exclude_seen_all_positions(candidates, history, torch.tensor([[7, 7, 7, 7]]), 8)
    assert 2 not in result[0, 0].tolist()
    assert not ({2, 4} & set(result[0, 1].tolist()))
    assert not ({2, 4, 6} & set(result[0, 2].tolist()))


def test_selected_negative_resampling_excludes_prefix() -> None:
    torch.manual_seed(5)
    history = torch.tensor([[2, 4, 6, 0], [1, 3, 5, 7]])
    candidates = torch.tensor([[2, 4, 8], [1, 3, 8]])
    result = _exclude_seen_selected_positions(
        candidates,
        history,
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([7, 7]),
        8,
    )
    assert not ({2, 4} & set(result[0].tolist()))
    assert not ({1, 3, 5} & set(result[1].tolist()))


def test_temporal_conditioning_changes_states_and_remains_causal() -> None:
    model = _model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    dense = torch.tensor([[100, 200, 300, 0, 0, 0]])
    sparse = torch.tensor([[100, 20_000, 40_000, 0, 0, 0]])
    dense_states = model.backbone.sequence_states(history, dense)
    sparse_states = model.backbone.sequence_states(history, sparse)
    assert not torch.allclose(dense_states[:, 1:3], sparse_states[:, 1:3])
    torch.testing.assert_close(dense_states[:, 0], sparse_states[:, 0])


def test_query_timestamp_tensor_keeps_causal_times_and_appends_request_time() -> None:
    history_timestamps = torch.tensor([[10, 20, 30, 0], [40, 50, 0, 0]])
    result = _query_timestamp_tensor(
        history_timestamps, torch.tensor([3, 2]), torch.tensor([100, 200])
    )
    expected = torch.tensor([[10, 20, 100, 0], [40, 200, 0, 0]])
    torch.testing.assert_close(result, expected)


def test_training_query_times_do_not_use_future_event_timestamps() -> None:
    rows = [
        {
            "items": [1, 2, 3, 4],
            "timestamps": [10, 20, 30, 40],
        }
    ]
    history, targets, timestamps, query_timestamps = _sequence_tensors_with_timestamps(
        rows, 4, "cpu"
    )
    torch.testing.assert_close(history, torch.tensor([[1, 2, 0, 0]]))
    torch.testing.assert_close(targets, torch.tensor([[2, 3, 0, 0]]))
    torch.testing.assert_close(timestamps, torch.tensor([[10, 20, 0, 0]]))
    torch.testing.assert_close(query_timestamps, timestamps)


def test_target_query_time_changes_final_state_without_breaking_causality() -> None:
    model = _model().eval()
    with torch.no_grad():
        bucket_bias = torch.arange(32, dtype=torch.float32).unsqueeze(1)
        model.backbone.temporal_bias.weight.copy_(
            bucket_bias.expand(-1, model.backbone.temporal_heads)
        )
    history = torch.tensor([[1, 2, 3, 0, 0, 0]])
    timestamps = torch.tensor([[100, 200, 300, 0, 0, 0]])
    early_query = torch.tensor([[200, 300, 400, 0, 0, 0]])
    late_query = torch.tensor([[200, 300, 1_000, 0, 0, 0]])
    early_states = model.backbone.sequence_states(history, timestamps, early_query)
    late_states = model.backbone.sequence_states(history, timestamps, late_query)
    torch.testing.assert_close(early_states[:, :2], late_states[:, :2])
    assert not torch.allclose(early_states[:, 2], late_states[:, 2])


def test_chunked_inference_preserves_temporal_catalog_ranking() -> None:
    model = _model().eval()
    history = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]])
    timestamps = torch.tensor([[100, 200, 300, 0, 0, 0], [50, 100, 0, 0, 0, 0]])
    query_timestamps = _query_timestamp_tensor(
        timestamps, torch.tensor([3, 2]), torch.tensor([400, 200])
    )
    _, standard = model.score(history, timestamps=timestamps, query_timestamps=query_timestamps)
    _, chunked = model.score(
        history,
        timestamps=timestamps,
        query_timestamps=query_timestamps,
        inference_chunk_size=2,
    )
    torch.testing.assert_close(standard, chunked, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        torch.argsort(standard, dim=1, descending=True),
        torch.argsort(chunked, dim=1, descending=True),
    )


def test_joint_phase_updates_backbone_after_frozen_context_training() -> None:
    model = _model()
    rows = [
        {
            "user_id": "1",
            "items": [1, 2, 3, 4, 5],
            "timestamps": [100, 200, 300, 400, 500],
        }
    ]
    config = ExperimentConfig(
        seed=3,
        dimension=8,
        heads=2,
        layers=1,
        max_length=6,
        dropout=0.0,
        batch_size=1,
        sampled_negatives=2,
        learning_rate=0.01,
        contextual_positions_per_user=4,
    )
    counts = np.ones(13)
    initial = model.backbone.items.weight.detach().clone()
    _fit_context(model, rows, counts, config, epochs=1)
    torch.testing.assert_close(model.backbone.items.weight, initial)
    _fit_context(model, rows, counts, config, epochs=1, joint=True)
    assert not torch.allclose(model.backbone.items.weight, initial)
