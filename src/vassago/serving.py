"""Canonical Python inference contract, independent of HTTP and explanation models."""

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.torch import load_file, save_file

from vassago.config import ExperimentConfig
from vassago.data import Interaction, Movie, validate_catalog
from vassago.features import TextEncoder
from vassago.models import SASRec
from vassago.retrieval import (
    AdaptiveGate,
    Calibrator,
    fusion_features,
    gate_features,
    mmr,
    union_candidates,
)
from vassago.semantic_ids import SIDExpert


class UserPreferenceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    favorite_movie_ids: list[int] = Field(
        default_factory=list,
        max_length=10,
        description="Up to ten favorites ordered from least to most representative",
    )
    genres: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    years: tuple[int, int] | None = None
    languages: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_year_range(self) -> "UserPreferenceProfile":
        if self.years and not 1800 <= self.years[0] <= self.years[1] <= 3000:
            raise ValueError("years must be an ordered plausible inclusive range")
        if any(movie_id < 1 for movie_id in self.favorite_movie_ids):
            raise ValueError("favorite movie IDs must be positive")
        if len(set(self.favorite_movie_ids)) != len(self.favorite_movie_ids):
            raise ValueError("favorite movie IDs must be unique")
        return self


class RecommendationContext(BaseModel):
    timestamp: int = Field(ge=0)
    strategy: str = "adaptive"
    diversify: bool = False


class Recommendation(BaseModel):
    movie_id: int
    final_score: float
    expert_scores: dict[str, float]
    candidate_sources: list[str]
    expert_ranks: dict[str, int]
    weights: list[float]
    sid_evidence: dict[str, Any]
    matched_features: list[str]


class Recommender(Protocol):
    def recommend(
        self,
        history: list[Interaction],
        k: int = 20,
        profile: UserPreferenceProfile | None = None,
        context: RecommendationContext | None = None,
    ) -> list[Recommendation]: ...

    def save(self, path: Path) -> None: ...


def history_tensor(histories: list[list[int]], length: int, device: str) -> torch.Tensor:
    result = torch.zeros(len(histories), length, dtype=torch.long, device=device)
    for i, history in enumerate(histories):
        clipped = history[-length:]
        if clipped:
            result[i, : len(clipped)] = torch.tensor(clipped, device=device)
    return result


class VassagoRanker:
    def __init__(
        self,
        config: ExperimentConfig,
        movies: list[Movie],
        embeddings: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        validate_catalog(movies)
        self.config, self.movies = config, movies
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.float32)
        tensor = torch.tensor(self.embeddings)
        kwargs: dict[str, Any] = {
            "n_items": len(movies),
            "dimension": config.dimension,
            "max_length": config.max_length,
            "heads": config.heads,
            "layers": config.layers,
            "dropout": config.dropout,
        }
        self.id_expert = SASRec(**kwargs).to(config.device)
        self.semantic_expert = SASRec(
            **kwargs, semantic=tensor.clone(), finetune=config.finetune_semantic
        ).to(config.device)
        self.sid_expert = SIDExpert(
            tensor, config.dimension, config.codebooks, config.codebook_size
        ).to(config.device)
        self.gate = AdaptiveGate().to(config.device)
        self.calibrator = Calibrator(config.calibration)
        self.weights = np.ones(3) / 3
        self.pair_weights = np.array([0.5, 0.5, 0.0])
        self._sid_codes: torch.Tensor | None = None
        self.completeness = np.array(
            [0] + [(bool(m.genres) + bool(m.metadata)) / 2 for m in movies]
        )

    def eval(self) -> None:
        for module in (self.id_expert, self.semantic_expert, self.sid_expert, self.gate):
            module.eval()

    def refresh_indexes(self) -> None:
        """Call after externally fine-tuning weights; retrieval caches are inference-only."""
        self._sid_codes = None

    def add_movie(self, movie: Movie, vector: np.ndarray) -> None:
        """Append a metadata-only item without inventing collaborative evidence."""
        if movie.movie_id != len(self.movies) + 1:
            raise ValueError("New movies must append the next canonical ID")
        validate_catalog([*self.movies, movie])
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("New embedding must match the catalog encoder and be finite")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("New movie requires a nonzero semantic embedding")
        vector = vector / norm
        self.movies = [*self.movies, movie]
        self.embeddings = np.vstack([self.embeddings, vector])
        self.counts = np.append(self.counts, np.float32(0))
        self.completeness = np.append(
            self.completeness, (bool(movie.genres) + bool(movie.metadata)) / 2
        )
        semantic = torch.tensor(vector[None], device=self.config.device)
        old_semantic = self.semantic_expert.items.weight.detach()
        self.semantic_expert.items = torch.nn.Embedding.from_pretrained(
            torch.cat([old_semantic, semantic]),
            freeze=not self.config.finetune_semantic,
            padding_idx=0,
        )
        old_id = self.id_expert.items.weight.detach()
        self.id_expert.items = torch.nn.Embedding.from_pretrained(
            torch.cat([old_id, torch.zeros_like(old_id[:1])]), freeze=False, padding_idx=0
        )
        self.sid_expert.embeddings = torch.cat([self.sid_expert.embeddings, semantic])
        self.refresh_indexes()
        self.eval()

    def eligible(self, history: list[int], timestamp: int) -> np.ndarray:
        seen = set(history)
        return np.array(
            [False]
            + [
                m.movie_id not in seen
                and m.available_at <= timestamp
                and m.metadata_available_at <= timestamp
                for m in self.movies
            ]
        )

    @torch.no_grad()
    def retrieve(
        self, history: list[int], timestamp: int, seen: list[int] | None = None
    ) -> dict[str, Any]:
        self.eval()
        tensor = history_tensor([history], self.config.max_length, self.config.device)
        allowed = self.eligible(seen if seen is not None else history, timestamp)
        raw = {
            "id": self.id_expert.score(tensor)[0].cpu().numpy(),
            "semantic": self.semantic_expert.score(tensor)[0].cpu().numpy(),
            "popularity": np.log1p(self.counts),
        }
        # Untrained atomic rows never compete with metadata-supported cold items.
        id_allowed = allowed & (self.counts > 0)
        if self._sid_codes is None:
            self._sid_codes = self.sid_expert.codes()
        sid_rows = self.sid_expert.decoder.generate(
            self.sid_expert.context(tensor)[0],
            self._sid_codes,
            self.config.beam_width,
            len(self.movies),
        )
        sid_rows = [r for r in sid_rows if allowed[int(r["movie_id"])]]
        raw["sid"] = np.full(len(self.movies) + 1, -100.0)
        sid_info = {}
        for row in sid_rows:
            item = int(row["movie_id"])
            raw["sid"][item] = row["log_probability"]
            sid_info[item] = row
        outputs = {}
        for expert in ("id", "semantic", "sid", "popularity"):
            mask = id_allowed if expert == "id" else allowed.copy()
            if expert == "sid":
                mask &= raw["sid"] > -100
            eligible_ids = np.flatnonzero(mask)
            order = eligible_ids[np.argsort(-raw[expert][eligible_ids], kind="stable")]
            limit = self.config.popularity_k if expert == "popularity" else self.config.candidate_k
            outputs[expert] = [(int(i), float(raw[expert][i])) for i in order[:limit]]
        candidates = union_candidates(outputs)
        ids = [c.movie_id for c in candidates]
        matrix = np.array(
            [[raw[e][i] for e in ("id", "semantic", "sid")] for i in ids], dtype=np.float32
        ).reshape(-1, 3)
        sources = np.array(
            [[e in c.sources for e in ("id", "semantic", "sid")] for c in candidates],
            dtype=np.float32,
        ).reshape(-1, 3)
        features = gate_features(len(history), ids, self.counts, sources, self.completeness)
        for candidate in candidates:
            candidate.sid = dict(sid_info.get(candidate.movie_id, {}))
            candidate.scores.update(
                {e: float(raw[e][candidate.movie_id]) for e in ("id", "semantic", "sid")}
            )
        return {
            "ids": ids,
            "raw": raw,
            "scores": matrix,
            "features": features,
            "candidates": candidates,
            "outputs": outputs,
            "allowed": allowed,
        }

    @torch.no_grad()
    def score(
        self, retrieved: dict[str, Any], strategy: str = "adaptive"
    ) -> tuple[np.ndarray, np.ndarray]:
        if not retrieved["ids"]:
            return np.zeros(0), np.zeros((0, 3))
        calibrated = self.calibrator.transform(retrieved["scores"])
        if strategy in ("adaptive", "refined", "diverse"):
            features = torch.tensor(
                fusion_features(retrieved["features"], calibrated), device=self.config.device
            )
            enabled = torch.ones(len(features), 3, dtype=torch.bool, device=self.config.device)
            enabled[:, 0] = torch.tensor(
                self.counts[retrieved["ids"]] > 0, device=self.config.device
            )
            enabled[:, 2] = torch.tensor(retrieved["features"][:, 6] > 0, device=self.config.device)
            weights = self.gate(features, enabled).cpu().numpy()
        elif strategy in ("weighted", "pair"):
            weights = np.tile(
                self.pair_weights if strategy == "pair" else self.weights, (len(calibrated), 1)
            )
            weights[self.counts[retrieved["ids"]] == 0, 0] = 0
            empty = weights.sum(1) == 0
            weights[empty, 1] = 1
            weights /= weights.sum(1, keepdims=True).clip(1e-12)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
        score = (weights * calibrated).sum(1)
        if strategy in ("refined", "diverse"):
            score += self.config.refinement_weight * retrieved["scores"][:, 1]
        return score, weights

    def recommend(
        self,
        history: list[Interaction],
        k: int = 20,
        profile: UserPreferenceProfile | None = None,
        context: RecommendationContext | None = None,
    ) -> list[Recommendation]:
        if not 1 <= k <= 1000:
            raise ValueError("k must lie between 1 and 1000")
        if context is None:
            raise ValueError("An explicit recommendation timestamp is required")
        if any(e.timestamp >= context.timestamp for e in history):
            raise ValueError("History must strictly precede recommendation timestamp")
        if any(e.movie_id > len(self.movies) for e in history):
            raise ValueError("History contains unknown movies")
        if len({e.user_id for e in history}) > 1:
            raise ValueError("A recommendation history must belong to one source-specific user")
        if any(e.timestamp < self.movies[e.movie_id - 1].available_at for e in history):
            raise ValueError("History contains an interaction before movie availability")
        ids = [
            e.movie_id
            for e in sorted(history, key=lambda e: (e.timestamp, e.movie_id))
            if e.rating >= self.config.positive_threshold
        ]
        if not ids and profile is None:
            eligible = np.flatnonzero(
                self.eligible([e.movie_id for e in history], context.timestamp)
            )
            ordered = eligible[np.argsort(-self.counts[eligible], kind="stable")][:k]
            return [
                Recommendation(
                    movie_id=int(i),
                    final_score=float(np.log1p(self.counts[i])),
                    expert_scores={"popularity": float(self.counts[i])},
                    candidate_sources=["popularity"],
                    expert_ranks={"popularity": rank},
                    weights=[0, 0, 0],
                    sid_evidence={},
                    matched_features=["training popularity fallback"],
                )
                for rank, i in enumerate(ordered, 1)
            ]
        if not ids and profile is not None:
            document = "\n".join(
                f"{key}: {value}" for key, value in profile.model_dump().items() if value
            )
            encoder = TextEncoder(
                self.config.encoder, self.config.dimension, self.config.encoder_revision
            )
            scores = self.embeddings @ encoder.encode([document])[0]
            eligible = np.flatnonzero(
                self.eligible([e.movie_id for e in history], context.timestamp)
            )
            ordered = eligible[np.argsort(-scores[eligible], kind="stable")][:k]
            return [
                Recommendation(
                    movie_id=int(i),
                    final_score=float(scores[i]),
                    expert_scores={"semantic": float(scores[i])},
                    candidate_sources=["profile"],
                    expert_ranks={"semantic": rank},
                    weights=[0, 1, 0],
                    sid_evidence={},
                    matched_features=sorted(set(profile.genres) & set(self.movies[i - 1].genres)),
                )
                for rank, i in enumerate(ordered, 1)
            ]
        result = self.retrieve(ids, context.timestamp, [e.movie_id for e in history])
        scores, weights = self.score(result, context.strategy)
        if context.diversify or context.strategy == "diverse":
            chosen = mmr(result["ids"], scores, self.embeddings, k, self.config.mmr_lambda)
        else:
            chosen = [result["ids"][i] for i in np.argsort(-scores, kind="stable")[:k]]
        positions = {item: position for position, item in enumerate(result["ids"])}
        output = []
        for item in chosen:
            position = positions[item]
            candidate = result["candidates"][position]
            evidence = (
                ["long-tail discovery"] if self.counts[item] <= np.median(self.counts[1:]) else []
            )
            output.append(
                Recommendation(
                    movie_id=item,
                    final_score=float(scores[position]),
                    expert_scores=candidate.scores,
                    candidate_sources=candidate.sources,
                    expert_ranks=candidate.ranks,
                    weights=weights[position].tolist(),
                    sid_evidence=candidate.sid,
                    matched_features=evidence,
                )
            )
        return output

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "config": self.config.model_dump(),
            "movies": [m.model_dump() for m in self.movies],
            "counts": self.counts.tolist(),
            "calibrator": self.calibrator.state(),
            "weights": self.weights.tolist(),
            "pair_weights": self.pair_weights.tolist(),
        }
        (path / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.save(path / "embeddings.npy", self.embeddings, allow_pickle=False)
        state = {}
        for name in ("id_expert", "semantic_expert", "sid_expert", "gate"):
            for key, tensor in getattr(self, name).state_dict().items():
                state[f"{name}.{key}"] = tensor.detach().cpu().contiguous()
        save_file(state, path / "model.safetensors")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "VassagoRanker":
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported checkpoint schema")
        config = ExperimentConfig.model_validate({**metadata["config"], "device": device})
        obj = cls(
            config,
            [Movie.model_validate(m) for m in metadata["movies"]],
            np.load(path / "embeddings.npy", allow_pickle=False),
            np.array(metadata["counts"]),
        )
        state = load_file(path / "model.safetensors", device=device)
        for name in ("id_expert", "semantic_expert", "sid_expert", "gate"):
            getattr(obj, name).load_state_dict(
                {k[len(name) + 1 :]: v for k, v in state.items() if k.startswith(name + ".")}
            )
        obj.calibrator = Calibrator.from_state(metadata["calibrator"])
        obj.weights = np.array(metadata["weights"])
        obj.pair_weights = np.array(metadata["pair_weights"])
        obj.eval()
        return obj
