import json
from pathlib import Path

from vassago.config import ExperimentConfig
from vassago.search import search


def test_search_registers_selection_before_final_test(tmp_path: Path) -> None:
    output = search(
        ExperimentConfig(epochs=1, tokenizer_epochs=1, evaluation_limit=2),
        [{"learning_rate": 0.003}],
        tmp_path / "search",
    )
    choices = json.loads((output / "selected_trials.json").read_text())
    rows = json.loads((output / "selected_test_metrics.json").read_text())
    assert len(choices) == len(rows) == 12
    validation = next((output / "validation").glob("*/*/run_manifest.json"))
    assert json.loads(validation.read_text())["evaluation_partition"] == "validation"
    assert all(row["selected_trial"] == 0 for row in rows)
