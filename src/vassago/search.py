"""Equal-trial-budget tuning; select every method before any final-test run."""

import json
from pathlib import Path
from typing import Any

from vassago.config import ExperimentConfig
from vassago.experiment import run, write_json


def search(
    base: ExperimentConfig, trials: list[dict[str, Any]], output: Path, data: Path | None = None
) -> Path:
    if not trials:
        raise ValueError("Provide at least one registered hyperparameter trial")
    allowed = {
        "learning_rate",
        "weight_decay",
        "dimension",
        "heads",
        "layers",
        "dropout",
        "codebooks",
        "codebook_size",
        "mmr_lambda",
        "refinement_weight",
    }
    if any(set(trial) - allowed for trial in trials):
        raise ValueError(
            "Search must preserve dataset, split, seed, feedback and evaluation protocol"
        )
    output.mkdir(parents=True, exist_ok=False)
    configs = [ExperimentConfig.model_validate({**base.model_dump(), **trial}) for trial in trials]
    # Register the complete search space before optimization; no adaptive test feedback.
    write_json(
        output / "search_space.json",
        {
            "selection_metric": "NDCG@10",
            "selection_partition": "validation",
            "configs": [c.model_dump() for c in configs],
        },
    )
    best: dict[str, dict[str, Any]] = {}
    for index, config in enumerate(configs):
        path = run(config, output / "validation" / str(index), data, "validation")
        for row in json.loads((path / "metrics.json").read_text()):
            method = row["method"]
            if method not in best or row["NDCG@10"] > best[method]["validation_ndcg10"]:
                best[method] = {"trial": index, "validation_ndcg10": row["NDCG@10"]}
    selection = output / "selected_trials.json"
    write_json(selection, best)
    # Selection is persisted before opening any final test targets.
    final: dict[int, dict[str, dict[str, Any]]] = {}
    for index in sorted({choice["trial"] for choice in best.values()}):
        path = run(configs[index], output / "test" / str(index), data, "test")
        final[index] = {r["method"]: r for r in json.loads((path / "metrics.json").read_text())}
    write_json(
        output / "selected_test_metrics.json",
        [
            {**final[choice["trial"]][method], "selected_trial": choice["trial"]}
            for method, choice in sorted(best.items())
        ],
    )
    return output
