import json
from pathlib import Path

import numpy as np
import pytest

from vassago.config import ExperimentConfig
from vassago.experiment import run
from vassago.serving import RecommendationContext, UserPreferenceProfile, VassagoRanker


def test_end_to_end_training_save_load_and_cold_profile(tmp_path: Path) -> None:
    config = ExperimentConfig(epochs=1, tokenizer_epochs=2, evaluation_limit=8)
    output = run(config, tmp_path)
    model = VassagoRanker.load(output / "model")
    context = RecommendationContext(timestamp=9999)
    before = model.recommend([], 5, context=context)
    model.save(tmp_path / "reload")
    after = VassagoRanker.load(tmp_path / "reload").recommend([], 5, context=context)
    assert before == after
    profile = model.recommend([], 5, UserPreferenceProfile(genres=["Drama"]), context)
    assert len(profile) == 5 and all(r.weights == [0, 1, 0] for r in profile)
    assert all(np.isfinite(r.final_score) for r in before)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert (output / "cold_item_metrics.json").exists()
    assert (output / "complementarity.json").exists()
    from vassago.pipeline import evaluate_checkpoint

    reevaluated = evaluate_checkpoint(output, tmp_path / "reevaluated.json")
    metrics = json.loads((output / "metrics.json").read_text())
    adaptive = next(row for row in metrics if row["method"] == "adaptive")
    assert reevaluated["metrics"]["NDCG@10"] == pytest.approx(adaptive["NDCG@10"], abs=1e-12)
