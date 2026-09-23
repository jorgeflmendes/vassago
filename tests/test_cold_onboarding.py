import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "train_cold_onboarding", Path(__file__).parents[1] / "scripts" / "train_cold_onboarding.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_metrics = _MODULE._metrics
_rows = _MODULE._rows


class _TiedModel:
    def __init__(self) -> None:
        self.history: torch.Tensor | None = None
        self.timestamps: torch.Tensor | None = None
        self.query_timestamps: torch.Tensor | None = None

    def score(
        self,
        history: torch.Tensor,
        *,
        timestamps: torch.Tensor,
        query_timestamps: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        self.history = history.detach().cpu()
        self.timestamps = timestamps.detach().cpu()
        self.query_timestamps = query_timestamps.detach().cpu()
        return None, torch.zeros(len(history), 31)


def test_cold_rows_exclude_the_complete_pre_query_history() -> None:
    sequence = {
        "items": [1, 2, 3, 4, 5],
        "ratings": [5.0] * 5,
        "timestamps": [1, 2, 3, 4, 5],
    }
    assert _rows([sequence], "test", 2)[0]["seen"] == {1, 2, 3, 4}


def test_cold_metrics_use_configured_window_and_deterministic_ties() -> None:
    model = _TiedModel()
    metrics = _metrics(
        model,
        [
            {
                "history": [3, 4, 5, 6],
                "history_timestamps": [30, 40, 50, 60],
                "timestamp": 60,
                "seen": {3, 4, 5, 6},
                "target": 30,
            }
        ],
        max_length=6,
        window=2,
        device="cpu",
    )
    assert model.history is not None
    assert model.history[0].tolist() == [5, 6, 0, 0, 0, 0]
    assert model.timestamps is not None
    assert model.timestamps[0].tolist() == [50, 60, 0, 0, 0, 0]
    assert model.query_timestamps is not None
    assert model.query_timestamps[0].tolist() == [50, 60, 0, 0, 0, 0]
    assert metrics["Recall@10"] == 0
    assert metrics["Recall@50"] == 1
