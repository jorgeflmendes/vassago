"""Pareto-safe profile cold start for the final contextual ranker.

The router keeps the trained sequential ranker authoritative whenever at least
one positive interaction exists.  Consequently, adding or changing a profile
cannot perturb an established user's scores.  Empty-history requests use a
metadata expert aligned to the collaborative item space.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from safetensors.torch import load_file

from vassago.config import ExperimentConfig
from vassago.contextual_ranker import (
    ContextualEvidenceRanker,
    _query_timestamp_tensor,
)
from vassago.data import Interaction, Movie, validate_catalog
from vassago.serving import RecommendationContext, UserPreferenceProfile

_YEAR = re.compile(r"\((\d{4})\)\s*$")


class HybridRecommendation(BaseModel):
    movie_id: int
    score: float
    source: Literal["contextual", "onboarding", "profile", "popularity"]
    matched_features: list[str] = Field(default_factory=list)


def _values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if isinstance(item, (str, int))]
    return []


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _movie_tokens(movie: Movie) -> set[str]:
    tokens = {f"genre:{_normalise(value)}" for value in movie.genres}
    fields = {
        "director": ("director", "directors"),
        "actor": ("actor", "actors", "cast"),
        "language": ("language", "languages", "original_language"),
    }
    for prefix, keys in fields.items():
        for key in keys:
            for value in _values(movie.metadata.get(key)):
                tokens.add(f"{prefix}:{_normalise(value)}")
    return tokens


def _movie_year(movie: Movie) -> int | None:
    raw = movie.metadata.get("year")
    if isinstance(raw, int) and 1800 <= raw <= 3000:
        return raw
    match = _YEAR.search(movie.title)
    return int(match.group(1)) if match else None


class CollaborativeProfileProjector:
    """Project explicit metadata preferences into trained collaborative geometry."""

    def __init__(
        self,
        movies: list[Movie],
        item_vectors: np.ndarray,
        counts: np.ndarray,
        ridge: float = 0.1,
        popularity_weight: float = 0.8,
    ) -> None:
        validate_catalog(movies)
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if not 0 <= popularity_weight < 1:
            raise ValueError("popularity_weight must lie in [0, 1)")
        vectors = np.asarray(item_vectors, dtype=np.float32)
        popularity = np.asarray(counts, dtype=np.float32)
        if vectors.shape[0] != len(movies) + 1:
            raise ValueError("item vectors must include the zero padding row")
        if not np.isfinite(vectors).all():
            raise ValueError("item vectors must be finite")
        if (
            popularity.shape != (len(movies) + 1,)
            or not np.isfinite(popularity).all()
            or (popularity < 0).any()
        ):
            raise ValueError("counts must be finite, nonnegative and include the zero padding row")

        self.movies = movies
        self.popularity_weight = popularity_weight
        self.tokens = [_movie_tokens(movie) for movie in movies]
        vocabulary = sorted(set().union(*self.tokens))
        self.feature_index = {feature: index for index, feature in enumerate(vocabulary)}
        features = np.zeros((len(movies), len(vocabulary)), dtype=np.float32)
        for row, tokens in enumerate(self.tokens):
            for token in tokens:
                features[row, self.feature_index[token]] = 1
        if vocabulary:
            document_frequency = (features > 0).sum(axis=0)
            self.idf = (np.log((1 + len(movies)) / (1 + document_frequency)) + 1).astype(
                np.float32
            )
            weighted = features * self.idf
            self.normalised_features = weighted / np.linalg.norm(
                weighted, axis=1, keepdims=True
            ).clip(1e-12)
            gram = weighted.T @ weighted
            self.projection = np.linalg.solve(
                gram + ridge * np.eye(len(vocabulary), dtype=np.float32),
                weighted.T @ vectors[1:],
            ).astype(np.float32)
        else:
            self.idf = np.zeros(0, dtype=np.float32)
            self.normalised_features = features
            self.projection = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        self.item_vectors = vectors[1:] / np.linalg.norm(
            vectors[1:], axis=1, keepdims=True
        ).clip(1e-12)
        scaled = np.log1p(popularity[1:])
        self.popularity = scaled / max(float(scaled.max()), 1.0)

    @staticmethod
    def _profile_tokens(profile: UserPreferenceProfile) -> set[str]:
        fields = {
            "genre": profile.genres,
            "director": profile.directors,
            "actor": profile.actors,
            "language": profile.languages,
        }
        return {
            f"{prefix}:{_normalise(value)}"
            for prefix, values in fields.items()
            for value in values
        }

    def preference_score(
        self, profile: UserPreferenceProfile | None
    ) -> tuple[np.ndarray, list[set[str]], bool]:
        requested = self._profile_tokens(profile) if profile is not None else set()
        vector = np.zeros(len(self.feature_index), dtype=np.float32)
        for token in requested:
            index = self.feature_index.get(token)
            if index is not None:
                vector[index] = self.idf[index]
        known = np.linalg.norm(vector) > 0
        if known:
            vector /= np.linalg.norm(vector)
            direct = self.normalised_features @ vector
            projected = vector @ self.projection
            projected /= max(float(np.linalg.norm(projected)), 1e-12)
            collaborative = self.item_vectors @ projected
            scores = 0.70 * direct + 0.30 * collaborative
        else:
            scores = self.popularity.copy()

        if profile is not None and profile.years is not None:
            start, end = profile.years
            years = np.array(
                [
                    float(start <= year <= end) if year is not None else 0.0
                    for year in map(_movie_year, self.movies)
                ],
                dtype=np.float32,
            )
            scores = 0.9 * scores + 0.1 * years
        matches = [tokens & requested for tokens in self.tokens]
        return scores.astype(np.float32, copy=False), matches, bool(known)

    def score(self, profile: UserPreferenceProfile | None) -> tuple[np.ndarray, list[set[str]]]:
        preference, matches, known = self.preference_score(profile)
        if not known and not (profile is not None and profile.years is not None):
            return self.popularity.copy(), matches
        scores = (
            (1 - self.popularity_weight) * preference
            + self.popularity_weight * self.popularity
        )
        return scores.astype(np.float32, copy=False), matches


class ParetoHybridRecommender:
    """Route empty histories to profiles and preserve every warm score exactly."""

    def __init__(
        self,
        model: ContextualEvidenceRanker,
        config: ExperimentConfig,
        movies: list[Movie],
        counts: np.ndarray,
        cold_model: ContextualEvidenceRanker | None = None,
        onboarding_centroid_weight: float = 0.1,
        inference_attention_chunk_size: int | None = 16,
    ) -> None:
        if len(movies) + 1 != model.backbone.items.num_embeddings:
            raise ValueError("catalog and checkpoint item counts differ")
        self.model = model.eval()
        if cold_model is not None and len(movies) + 1 != cold_model.backbone.items.num_embeddings:
            raise ValueError("catalog and cold checkpoint item counts differ")
        if not 0 <= onboarding_centroid_weight <= 1:
            raise ValueError("onboarding_centroid_weight must lie in [0, 1]")
        if inference_attention_chunk_size is not None and inference_attention_chunk_size < 1:
            raise ValueError("inference_attention_chunk_size must be positive or None")
        self.cold_model = (cold_model or model).eval()
        self.onboarding_centroid_weight = onboarding_centroid_weight
        self.inference_attention_chunk_size = inference_attention_chunk_size
        self.config = config
        self.device = str(next(model.parameters()).device)
        self.movies = movies
        vectors = model.backbone.item_vectors().detach().cpu().numpy()
        self.profile_projector = CollaborativeProfileProjector(movies, vectors, counts)

    @classmethod
    def load(
        cls,
        config_path: Path,
        weights_path: Path,
        movies: list[Movie],
        counts: np.ndarray,
        device: str = "cpu",
        cold_weights_path: Path | None = None,
        onboarding_centroid_weight: float = 0.1,
        inference_attention_chunk_size: int | None = 16,
    ) -> ParetoHybridRecommender:
        config = ExperimentConfig.read(config_path).model_copy(update={"device": device})
        model = ContextualEvidenceRanker(
            len(movies),
            config.dimension,
            config.max_length,
            config.heads,
            config.layers,
            config.dropout,
            config.contextual_dimension,
            config.contextual_memory_window,
            config.contextual_temperature,
            config.contextual_heads,
            config.contextual_persistence_scales,
            config.contextual_velocity_scales,
        ).to(device)
        model.load_state_dict(load_file(weights_path, device=device), strict=True)
        cold_model = None
        if cold_weights_path is not None:
            cold_model = ContextualEvidenceRanker(
                len(movies),
                config.dimension,
                config.max_length,
                config.heads,
                config.layers,
                config.dropout,
                config.contextual_dimension,
                config.contextual_memory_window,
                config.contextual_temperature,
                config.contextual_heads,
                config.contextual_persistence_scales,
                config.contextual_velocity_scales,
            ).to(device)
            cold_model.load_state_dict(
                load_file(cold_weights_path, device=device), strict=True
            )
        return cls(
            model,
            config,
            movies,
            counts,
            cold_model=cold_model,
            onboarding_centroid_weight=onboarding_centroid_weight,
            inference_attention_chunk_size=inference_attention_chunk_size,
        )

    @torch.inference_mode()
    def recommend(
        self,
        history: list[Interaction],
        k: int,
        profile: UserPreferenceProfile | None,
        context: RecommendationContext,
    ) -> list[HybridRecommendation]:
        if not 1 <= k <= 1000:
            raise ValueError("k must lie between 1 and 1000")
        if any(event.timestamp >= context.timestamp for event in history):
            raise ValueError("History must strictly precede recommendation timestamp")
        if any(event.movie_id > len(self.movies) for event in history):
            raise ValueError("History contains unknown movies")
        if any(
            event.timestamp < self.movies[event.movie_id - 1].available_at
            for event in history
        ):
            raise ValueError("History contains an interaction before movie availability")
        if len({event.user_id for event in history}) > 1:
            raise ValueError("A recommendation history must belong to one user")

        positive = [
            event
            for event in sorted(history, key=lambda event: (event.timestamp, event.movie_id))
            if event.rating >= self.config.positive_threshold
        ]
        favorites = profile.favorite_movie_ids if profile is not None else []
        if any(movie_id > len(self.movies) for movie_id in favorites):
            raise ValueError("Profile contains unknown favorite movies")
        if any(
            self.movies[movie_id - 1].available_at > context.timestamp
            or self.movies[movie_id - 1].metadata_available_at > context.timestamp
            for movie_id in favorites
        ):
            raise ValueError("Profile contains a favorite unavailable at query time")
        seen = {event.movie_id for event in history} | set(favorites)
        eligible = np.array(
            [
                movie.movie_id not in seen
                and movie.available_at <= context.timestamp
                and movie.metadata_available_at <= context.timestamp
                for movie in self.movies
            ],
            dtype=bool,
        )

        if not positive and favorites:
            cold_favorites = favorites[-self.config.max_length :]
            sequence_length = max(len(cold_favorites), 1)
            ids = torch.zeros(
                1, sequence_length, dtype=torch.long, device=self.device
            )
            ids[0, : len(cold_favorites)] = torch.tensor(cold_favorites, device=self.device)
            _, contextual = self.cold_model.score(
                ids, inference_chunk_size=self.inference_attention_chunk_size
            )
            item_vectors = self.cold_model.backbone.item_vectors()
            favorite_vectors = item_vectors[
                torch.tensor(cold_favorites[-3:], device=self.device)
            ]
            centroid = torch.nn.functional.normalize(
                favorite_vectors.mean(0), dim=0
            ) @ item_vectors.T
            weight = self.onboarding_centroid_weight
            scores = (
                (1 - weight) * contextual[0] + weight * centroid
            ).detach().cpu().numpy()
            allowed_ids = np.flatnonzero(np.concatenate(([False], eligible)))
            order = allowed_ids[np.argsort(-scores[allowed_ids], kind="stable")[:k]]
            return [
                HybridRecommendation(
                    movie_id=int(movie_id),
                    score=float(scores[movie_id]),
                    source="onboarding",
                )
                for movie_id in order
            ]

        if not positive:
            scores, matches = self.profile_projector.score(profile)
            source: Literal["profile", "popularity"] = (
                "profile"
                if profile is not None and (any(matches) or profile.years is not None)
                else "popularity"
            )
            order = np.flatnonzero(eligible)
            order = order[np.argsort(-scores[order], kind="stable")[:k]]
            return [
                HybridRecommendation(
                    movie_id=int(index + 1),
                    score=float(scores[index]),
                    source=source,
                    matched_features=sorted(matches[index]),
                )
                for index in order
            ]

        events = positive[-self.config.max_length :]
        device = self.device
        sequence_length = max(len(events), 1)
        ids = torch.zeros(1, sequence_length, dtype=torch.long, device=device)
        timestamps = torch.zeros_like(ids)
        ids[0, : len(events)] = torch.tensor([event.movie_id for event in events], device=device)
        timestamps[0, : len(events)] = torch.tensor(
            [event.timestamp for event in events], device=device
        )
        query_timestamps = _query_timestamp_tensor(
            timestamps,
            torch.tensor([len(events)], device=device),
            torch.tensor([context.timestamp], device=device),
        )
        _, contextual = self.model.score(
            ids,
            timestamps=timestamps,
            query_timestamps=query_timestamps,
            inference_chunk_size=self.inference_attention_chunk_size,
        )
        scores = contextual[0].detach().cpu().numpy()
        allowed_ids = np.flatnonzero(np.concatenate(([False], eligible)))
        order = allowed_ids[np.argsort(-scores[allowed_ids], kind="stable")[:k]]
        return [
            HybridRecommendation(
                movie_id=int(movie_id),
                score=float(scores[movie_id]),
                source="contextual",
            )
            for movie_id in order
        ]
