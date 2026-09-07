import json
from pathlib import Path

from vassago.config import ExperimentConfig
from vassago.experiment import run, sha256


def test_repeated_cpu_run_has_identical_metrics_and_weights(tmp_path: Path) -> None:
    config = ExperimentConfig(epochs=1, tokenizer_epochs=1, evaluation_limit=4)
    a = run(config, tmp_path / "a")
    b = run(config, tmp_path / "b")
    assert json.loads((a / "metrics.json").read_text()) == json.loads(
        (b / "metrics.json").read_text()
    )
    assert sha256(a / "model" / "model.safetensors") == sha256(b / "model" / "model.safetensors")
