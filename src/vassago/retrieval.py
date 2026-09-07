"""Candidate provenance, calibration, transparent gating and MMR."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class Candidate:
    movie_id: int
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    sid: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def union_candidates(outputs: dict[str, list[tuple[int, float]]]) -> list[Candidate]:
    candidates: dict[int, Candidate] = {}
    for expert, rows in outputs.items():
        for rank, (movie_id, score) in enumerate(rows, start=1):
            candidate = candidates.setdefault(movie_id, Candidate(movie_id))
            if expert not in candidate.sources:
                candidate.scores[expert] = float(score)
                candidate.ranks[expert] = rank
                candidate.sources.append(expert)
    return [candidates[i] for i in sorted(candidates)]


class DenseIndex:
    def __init__(self, vectors: np.ndarray, ann: bool = False) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.index: Any = None
        if ann:
            import faiss

            self.index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 128
            self.index.add(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        queries = np.asarray(queries, dtype=np.float32)
        if self.index is not None:
            scores, ids = self.index.search(queries, k)
            return scores, ids
        scores = queries @ self.vectors.T
        ids = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids


class Calibrator:
    def __init__(self, method: str = "rank") -> None:
        self.method = method
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        self.temperature = np.ones(3)
        self.fitted_on: str | None = None

    def fit(self, scores: list[np.ndarray], targets: list[int], partition: str) -> None:
        if partition != "base_validation":
            raise ValueError("Calibration fits only on base_validation")
        if scores:
            values = np.concatenate(scores)
            self.mean = values.mean(0)
            self.std = np.maximum(values.std(0), 1e-6)
            for expert in range(3):
                losses = []
                for temperature in (0.1, 0.3, 1, 3, 10):
                    loss = 0.0
                    for matrix, target in zip(scores, targets, strict=True):
                        logits = matrix[:, expert] / temperature
                        logits = logits - logits.max()
                        loss += float(np.log(np.exp(logits).sum()) - logits[target])
                    losses.append(loss)
                self.temperature[expert] = (0.1, 0.3, 1, 3, 10)[int(np.argmin(losses))]
        self.fitted_on = partition

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.fitted_on is None:
            raise ValueError("Calibrator has not been fitted")
        if self.method == "zscore":
            return (scores - self.mean) / self.std
        if self.method == "temperature":
            z = scores / self.temperature
            z -= z.max(0)
            p = np.exp(z)
            return p / p.sum(0).clip(1e-12)
        if self.method != "rank":
            raise ValueError(f"Unknown calibration method: {self.method}")
        # Tied values receive equal ranks (including absent SID candidates).
        result = np.empty_like(scores)
        for column in range(scores.shape[1]):
            _, inverse = np.unique(scores[:, column], return_inverse=True)
            result[:, column] = inverse / max(int(inverse.max()), 1)
        return result

    def state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "temperature": self.temperature.tolist(),
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Calibrator":
        obj = cls(state["method"])
        for name in ("mean", "std", "temperature"):
            setattr(obj, name, np.asarray(state[name]))
        obj.fitted_on = state["fitted_on"]
        return obj


class AdaptiveGate(nn.Module):
    def __init__(self, features: int = 15) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, features: Tensor, enabled: Tensor | None = None) -> Tensor:
        logits = self.network(features)
        if enabled is not None:
            if not enabled.any(-1).all():
                raise ValueError("At least one expert must be enabled")
            logits = logits.masked_fill(~enabled, -torch.inf)
        return logits.softmax(-1)


def gate_features(
    history_length: int,
    ids: list[int],
    counts: np.ndarray,
    sources: np.ndarray,
    completeness: np.ndarray,
) -> np.ndarray:
    popularity = np.log1p(counts[ids])
    return np.column_stack(
        [
            np.full(len(ids), np.log1p(history_length)),
            popularity,
            counts[ids] == 0,
            completeness[ids],
            sources,
            np.full(len(ids), history_length == 0),
            np.full(len(ids), history_length <= 3),
        ]
    ).astype(np.float32)


def fusion_features(base: np.ndarray, calibrated_scores: np.ndarray) -> np.ndarray:
    """Add candidate evidence and per-expert confidence to contextual gate inputs."""
    if base.ndim != 2 or calibrated_scores.ndim != 2 or len(base) != len(calibrated_scores):
        raise ValueError("Fusion feature matrices must be aligned and two-dimensional")
    if calibrated_scores.shape[1] != 3:
        raise ValueError("Fusion expects exactly three expert score columns")
    if not len(calibrated_scores):
        return np.empty((0, base.shape[1] + 6), dtype=np.float32)
    ordered = np.sort(calibrated_scores, axis=0)
    margins = ordered[-1] - ordered[-2] if len(ordered) > 1 else np.zeros(3)
    confidence = np.broadcast_to(margins, calibrated_scores.shape)
    return np.column_stack([base, calibrated_scores, confidence]).astype(np.float32)


def mmr(
    ids: list[int], scores: np.ndarray, vectors: np.ndarray, k: int, relevance_weight: float
) -> list[int]:
    if not 0 <= relevance_weight <= 1:
        raise ValueError("MMR weight must be in [0, 1]")
    remaining = list(range(len(ids)))
    selected: list[int] = []
    scaled = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-12) if len(scores) else scores
    while remaining and len(selected) < k:

        def objective(i: int) -> tuple[float, int]:
            redundancy = max(
                (float(vectors[ids[i]] @ vectors[ids[j]]) for j in selected), default=0
            )
            return relevance_weight * float(scaled[i]) - (1 - relevance_weight) * redundancy, -ids[
                i
            ]

        best = max(remaining, key=objective)
        selected.append(best)
        remaining.remove(best)
    return [ids[i] for i in selected]
