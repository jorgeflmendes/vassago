"""Candidate-conditioned contextual evidence ranker for the immutable fair protocol."""

import copy
import hashlib
import json
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import functional as F

from vassago.config import ExperimentConfig
from vassago.data import Example
from vassago.fair_benchmark import FairProtocol, sha256
from vassago.fair_vassago import (
    _counts,
    _event_queries,
    _sequence_tensors,
    _sequences,
    _truncate_rows,
)
from vassago.models import SASRec
from vassago.training import seed_everything


class FixedSelectionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    source_protocol_hash: str = Field(min_length=64, max_length=64)
    selected_base_epoch: int = Field(ge=1)
    selected_context_epoch: int = Field(ge=1)
    selected_joint_epoch: int = Field(ge=0)
    selected_evidence_scale: float = Field(gt=0)
    selected_original_weight: float = Field(ge=0, le=1)
    config_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)


def _config_fingerprint(config: ExperimentConfig) -> str:
    payload = json.dumps(
        config.model_dump(exclude={"seed"}), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_selection_recipe(
    path: Path, target_protocol_hash: str, config: ExperimentConfig
) -> FixedSelectionRecipe:
    recipe = FixedSelectionRecipe.model_validate_json(path.read_text(encoding="utf-8"))
    if recipe.source_protocol_hash == target_protocol_hash:
        raise ValueError("Selection recipe must originate from a separate development protocol")
    if (
        recipe.config_fingerprint is not None
        and recipe.config_fingerprint != _config_fingerprint(config)
    ):
        raise ValueError("Selection recipe does not match the configured architecture")
    return recipe


def freeze_selection_recipe(
    selection_manifest: Path, config: ExperimentConfig, output: Path
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    required = {
        "protocol_hash",
        "selected_base_epoch",
        "selected_context_epoch",
        "selected_joint_epoch",
        "selected_evidence_scale",
        "selected_original_weight",
    }
    if selection.get("status") != "completed" or selection.get("test_evaluated") is not False:
        raise ValueError("Selection manifest must be a completed test-blind validation run")
    if required - selection.keys():
        raise ValueError("Selection manifest is missing frozen-recipe fields")
    manifest_config = json.dumps(selection.get("config"), sort_keys=True, separators=(",", ":"))
    configured = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    if manifest_config != configured:
        raise ValueError("Selection manifest does not match the configured architecture")
    recipe = FixedSelectionRecipe(
        schema_version=1,
        source_protocol_hash=selection["protocol_hash"],
        selected_base_epoch=selection["selected_base_epoch"],
        selected_context_epoch=selection["selected_context_epoch"],
        selected_joint_epoch=selection["selected_joint_epoch"],
        selected_evidence_scale=selection["selected_evidence_scale"],
        selected_original_weight=selection["selected_original_weight"],
        config_fingerprint=_config_fingerprint(config),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recipe.model_dump(), indent=2) + "\n", encoding="utf-8")
    return output


def _sequence_tensors_with_timestamps(
    rows: list[dict[str, Any]], max_length: int, device: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    history, targets = _sequence_tensors(rows, max_length, device)
    timestamps = torch.zeros_like(history)
    query_timestamps = torch.zeros_like(history)
    for index, row in enumerate(rows):
        training = row["timestamps"][:-1][-max_length - 1 :]
        inputs = training[:-1]
        if inputs:
            timestamps[index, : len(inputs)] = torch.tensor(inputs, device=device)
            query_timestamps[index, : len(inputs)] = torch.tensor(inputs, device=device)
    return history, targets, timestamps, query_timestamps


def _timestamp_tensor(
    user_ids: list[str], histories: dict[str, list[int]], max_length: int, device: str
) -> Tensor:
    result = torch.zeros(len(user_ids), max_length, dtype=torch.long, device=device)
    for row, user_id in enumerate(user_ids):
        values = histories[user_id][-max_length:]
        result[row, : len(values)] = torch.tensor(values, device=device)
    return result


def _query_timestamp_tensor(
    history_timestamps: Tensor, lengths: Tensor, query_timestamps: Tensor
) -> Tensor:
    result = history_timestamps.clone()
    rows = torch.arange(len(result), device=result.device)
    result[rows, (lengths - 1).clamp_min(0)] = query_timestamps
    return result


class ContextualBackbone(SASRec):
    """Repository SASRec with stable small initialization and scaled item inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        dimension = self.items.embedding_dim
        heads = self.encoder.layers[0].self_attn.num_heads
        self.temporal_heads = heads
        self.gap_embeddings = nn.Embedding(32, dimension)
        self.temporal_bias = nn.Embedding(32, heads)
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.items.weight[0].zero_()
            self.temporal_bias.weight.zero_()

    @staticmethod
    def _time_buckets(seconds: Tensor) -> Tensor:
        return torch.log2(seconds.clamp_min(0).float() / 60 + 1).long().clamp_max(31)

    def sequence_states(
        self,
        history: Tensor,
        timestamps: Tensor | None = None,
        query_timestamps: Tensor | None = None,
        inference_chunk_size: int | None = None,
    ) -> Tensor:
        valid = history.ne(0)
        positions = torch.arange(history.shape[1], device=history.device)
        x = self.projection(self.items(history)) * self.items.embedding_dim**0.5
        x = x + self.positions(positions)
        attention_mask: Tensor = torch.ones(
            history.shape[1], history.shape[1], device=history.device, dtype=torch.bool
        ).triu(1)
        padding = ~valid.clone()
        padding[:, 0] = False
        key_padding: Tensor | None = padding
        if timestamps is not None:
            gaps = timestamps - torch.roll(timestamps, 1, dims=1)
            gaps[:, 0] = 0
            x = x + self.gap_embeddings(self._time_buckets(gaps))
            query_times = timestamps if query_timestamps is None else query_timestamps
            elapsed = query_times[:, :, None] - timestamps[:, None, :]
            bias = self.temporal_bias(self._time_buckets(elapsed)).permute(0, 3, 1, 2)
            attention_mask = bias.reshape(
                len(history) * self.temporal_heads, history.shape[1], history.shape[1]
            )
            allowed_keys = valid.clone()
            allowed_keys[:, 0] = True
            invalid = positions[:, None].lt(positions[None, :])[None, None]
            invalid = invalid | ~allowed_keys[:, None, None, :]
            expanded_invalid = invalid.expand_as(
                attention_mask.view(
                    len(history), self.temporal_heads, history.shape[1], history.shape[1]
                )
            ).reshape_as(attention_mask)
            # A finite sentinel avoids NaN gradients in mixed-precision SDPA kernels.
            attention_mask = attention_mask.masked_fill(expanded_invalid, -10_000.0)
            key_padding = torch.zeros_like(timestamps, dtype=x.dtype)
        x = self.dropout(x) * valid.unsqueeze(-1)
        if inference_chunk_size is not None:
            if self.training:
                raise ValueError("inference_chunk_size is only available in evaluation mode")
            if inference_chunk_size < 1:
                raise ValueError("inference_chunk_size must be positive")
            return self._chunked_sequence_states(
                x, valid, timestamps, query_timestamps, inference_chunk_size
            )
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        if timestamps is not None and not self.training:
            # PyTorch's inference fast path is unstable with batch-specific 3-D masks.
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            x = self.encoder(x, mask=attention_mask, src_key_padding_mask=key_padding)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
        return F.normalize(self.norm(x), dim=-1) * valid.unsqueeze(-1)

    def _chunked_sequence_states(
        self,
        x: Tensor,
        valid: Tensor,
        timestamps: Tensor | None,
        query_timestamps: Tensor | None,
        chunk_size: int,
    ) -> Tensor:
        """Evaluate causal attention in bounded query blocks for serving.

        The training and benchmark paths retain PyTorch's standard encoder. This
        inference-only path keeps the exact learned projections while bounding the
        temporary attention tensor to ``batch × heads × chunk × sequence``.
        """
        batch_size, sequence_length, dimension = x.shape
        heads = self.temporal_heads
        head_dimension = dimension // heads
        positions = torch.arange(sequence_length, device=x.device)
        allowed_keys = valid.clone()
        allowed_keys[:, 0] = True
        query_times = timestamps if query_timestamps is None else query_timestamps
        result = x
        for layer in self.encoder.layers:
            source = result
            normalised = layer.norm1(source)
            query_weight, key_weight, value_weight = layer.self_attn.in_proj_weight.split(
                dimension
            )
            query_bias, key_bias, value_bias = layer.self_attn.in_proj_bias.split(dimension)
            key = F.linear(normalised, key_weight, key_bias).reshape(
                batch_size, sequence_length, heads, head_dimension
            )
            key = key.transpose(1, 2)
            value = F.linear(normalised, value_weight, value_bias).reshape(
                batch_size, sequence_length, heads, head_dimension
            )
            value = value.transpose(1, 2)
            updated = torch.empty_like(source)
            for start in range(0, sequence_length, chunk_size):
                end = min(start + chunk_size, sequence_length)
                query = F.linear(
                    normalised[:, start:end], query_weight, query_bias
                ).reshape(batch_size, end - start, heads, head_dimension)
                query = query.transpose(1, 2)
                logits = torch.matmul(
                    query, key.transpose(-2, -1)
                ) / head_dimension**0.5
                if timestamps is not None:
                    assert query_times is not None
                    elapsed = query_times[:, start:end, None] - timestamps[:, None, :]
                    bias = self.temporal_bias(self._time_buckets(elapsed)).permute(0, 3, 1, 2)
                    logits = logits + bias
                    invalid_value = -10_000.0
                else:
                    invalid_value = -torch.inf
                future = positions[start:end, None].lt(positions[None, :])
                invalid = future[None, None] | ~allowed_keys[:, None, None, :]
                logits = logits.masked_fill(invalid, invalid_value)
                weights = torch.softmax(logits, dim=-1)
                attention = torch.matmul(weights, value).transpose(1, 2)
                attention = attention.reshape(batch_size, end - start, dimension)
                attention = layer.self_attn.out_proj(attention)
                block = source[:, start:end] + layer.dropout1(attention)
                feedforward = layer.norm2(block)
                feedforward = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(feedforward)))
                )
                updated[:, start:end] = block + layer.dropout2(feedforward)
            result = updated
        return F.normalize(self.norm(result), dim=-1) * valid.unsqueeze(-1)


class ContextualEvidenceRanker(nn.Module):
    """Augment a collaborative score with candidate-specific historical evidence."""

    def __init__(
        self,
        n_items: int,
        dimension: int,
        max_length: int,
        heads: int,
        layers: int,
        dropout: float,
        contextual_dimension: int,
        memory_window: int,
        temperature: float,
        contextual_heads: int = 1,
        persistence_scales: int = 0,
        velocity_scales: int = 0,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = ContextualBackbone(
            n_items, dimension, max_length, heads, layers, dropout, ffn_dim=ffn_dim
        )
        self.memory_window = memory_window
        self.temperature = temperature
        self.key_projections = nn.ModuleList(
            nn.Linear(dimension, contextual_dimension, bias=False) for _ in range(contextual_heads)
        )
        self.query_projections = nn.ModuleList(
            nn.Linear(dimension, contextual_dimension, bias=False) for _ in range(contextual_heads)
        )
        self.salience_heads = nn.ModuleList(
            nn.Linear(dimension, 1, bias=False) for _ in range(contextual_heads)
        )
        self.decay_unconstrained = nn.Parameter(torch.zeros(contextual_heads))
        self.gamma_unconstrained = nn.Parameter(torch.full((contextual_heads,), -3.0))
        self.persistence_scales = persistence_scales
        if persistence_scales:
            self.persistence_router = nn.Linear(dimension, persistence_scales, bias=False)
            self.persistence_candidate_gate = nn.Linear(dimension, persistence_scales, bias=False)
            self.persistence_history_gate = nn.Linear(dimension, 1)
            start = torch.linspace(
                float(np.log(6 * 60 * 60)), float(np.log(90 * 24 * 60 * 60)), persistence_scales
            )
            increments = start - float(np.log(60 * 60))
            increments[1:] = start[1:] - start[:-1]
            self.persistence_half_life_increments = nn.Parameter(
                torch.log(torch.expm1(increments))
            )
            self.persistence_gamma_unconstrained = nn.Parameter(torch.tensor(-4.0))
            self.persistence_disagreement_unconstrained = nn.Parameter(torch.tensor(-2.0))
        self.velocity_scales = velocity_scales
        if velocity_scales:
            self.velocity_router = nn.Linear(dimension, velocity_scales, bias=False)
            self.velocity_candidate_gate = nn.Linear(dimension, velocity_scales, bias=False)
            self.velocity_history_gate = nn.Linear(dimension, 1)
            start = torch.linspace(
                float(np.log(30 * 60)), float(np.log(30 * 24 * 60 * 60)), velocity_scales
            )
            increments = start - float(np.log(5 * 60))
            increments[1:] = start[1:] - start[:-1]
            self.velocity_half_life_increments = nn.Parameter(torch.log(torch.expm1(increments)))
            self.velocity_gamma_unconstrained = nn.Parameter(torch.tensor(-4.0))

    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.gamma_unconstrained)

    def _memory(
        self, states: Tensor, valid: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather the causal memory ending at each selected sequence position."""
        offsets = torch.arange(self.memory_window, device=states.device)
        indices = positions[:, None] - (self.memory_window - 1 - offsets)[None, :]
        mask = indices.ge(0)
        indices = indices.clamp_min(0)
        batch = torch.arange(len(states), device=states.device)[:, None]
        memory = states[batch, indices]
        mask &= valid[batch, indices]
        return memory, mask, offsets.expand(len(states), -1)

    def _head_evidence(
        self,
        memory: Tensor,
        mask: Tensor,
        candidates: Tensor,
        head: int,
    ) -> Tensor:
        keys = F.normalize(self.key_projections[head](memory), dim=-1)
        item_vectors = self.backbone.item_vectors()[candidates]
        queries = F.normalize(self.query_projections[head](item_vectors), dim=-1)
        relative_age = torch.arange(
            memory.shape[1] - 1, -1, -1, device=memory.device, dtype=memory.dtype
        ) / max(memory.shape[1], 1)
        salience = self.salience_heads[head](memory).squeeze(-1)
        salience = salience - F.softplus(self.decay_unconstrained[head]) * relative_age
        salience = salience.masked_fill(~mask, -torch.inf)
        log_weights = F.log_softmax(salience, dim=-1)
        weights = log_weights.exp()
        similarities = torch.einsum("ncr,nwr->ncw", queries, keys)
        contextual = self.temperature * torch.logsumexp(
            log_weights[:, None, :] + similarities / self.temperature, dim=-1
        )
        mean_key = torch.einsum("nw,nwr->nr", weights, keys)
        linear = torch.einsum("ncr,nr->nc", queries, mean_key)
        return contextual - linear

    def _evidence(
        self,
        memory: Tensor,
        mask: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        components = torch.stack(
            [
                self._head_evidence(memory, mask, candidates, head)
                for head in range(len(self.key_projections))
            ],
            dim=-1,
        )
        return (components * self.gamma).sum(-1)

    def _persistence_tracks(
        self,
        states: Tensor,
        valid: Tensor,
        positions: Tensor,
        timestamps: Tensor | None,
    ) -> tuple[Tensor, Tensor] | None:
        if not self.persistence_scales:
            return None
        memory, mask, offsets = self._memory(states, valid, positions)
        log_half_lives = float(np.log(60 * 60)) + torch.cumsum(
            F.softplus(self.persistence_half_life_increments), dim=0
        )
        half_lives = log_half_lives.exp().to(dtype=states.dtype)
        if timestamps is None:
            ages = (self.memory_window - 1 - offsets).to(dtype=states.dtype)
        else:
            indices = positions[:, None] - (self.memory_window - 1 - offsets)
            indices = indices.clamp_min(0)
            batch = torch.arange(len(states), device=states.device)[:, None]
            memory_times = timestamps[batch, indices]
            query_times = timestamps[
                torch.arange(len(states), device=states.device), positions
            ][:, None]
            ages = (query_times - memory_times).clamp_min(0).to(dtype=states.dtype)
        decay = torch.exp(-ages[:, :, None] / half_lives[None, None, :])
        weights = F.softmax(self.persistence_router(memory), dim=-1) * decay
        weights = weights * mask[:, :, None]
        numerator = torch.einsum("nwk,nwd->nkd", weights, memory)
        mass = weights.sum(dim=1)
        normalized = numerator / mass.clamp_min(torch.finfo(states.dtype).tiny)[:, :, None]
        tracks = F.normalize(normalized, dim=-1)
        return tracks, mass

    def _persistence_evidence(
        self, tracks: tuple[Tensor, Tensor] | None, candidates: Tensor
    ) -> Tensor:
        if tracks is None:
            return self.backbone.item_vectors()[candidates].new_zeros(candidates.shape)
        states, mass = tracks
        item_vectors = self.backbone.item_vectors()[candidates]
        similarities = torch.einsum("ncd,nkd->nck", item_vectors, states)
        candidate_gate = self.persistence_candidate_gate(item_vectors)
        history_gate = self.persistence_history_gate(states).squeeze(-1)
        available = mass.gt(torch.finfo(mass.dtype).tiny)[:, None, :]
        gate_logits = (candidate_gate + history_gate[:, None, :]).masked_fill(
            ~available, -torch.inf
        )
        gates = F.softmax(gate_logits, dim=-1)
        signal = (gates * similarities).sum(-1)
        disagreement = similarities.var(dim=-1, unbiased=False)
        confidence = torch.exp(
            -F.softplus(self.persistence_disagreement_unconstrained) * disagreement
        )
        return F.softplus(self.persistence_gamma_unconstrained) * confidence * signal

    def _velocity_tracks(
        self,
        states: Tensor,
        valid: Tensor,
        positions: Tensor,
        timestamps: Tensor | None,
        query_timestamps: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        if not self.velocity_scales:
            return None
        memory, mask, offsets = self._memory(states, valid, positions)
        transitions = memory[:, 1:] - memory[:, :-1]
        transition_mask = mask[:, 1:] & mask[:, :-1]
        log_half_lives = float(np.log(5 * 60)) + torch.cumsum(
            F.softplus(self.velocity_half_life_increments), dim=0
        )
        half_lives = log_half_lives.exp().to(dtype=states.dtype)
        if timestamps is None:
            ages = (self.memory_window - 1 - offsets[:, 1:]).to(dtype=states.dtype)
        else:
            indices = positions[:, None] - (self.memory_window - 1 - offsets[:, 1:])
            indices = indices.clamp_min(0)
            batch = torch.arange(len(states), device=states.device)[:, None]
            end_times = timestamps[batch, indices]
            if query_timestamps is None:
                query_times = timestamps[
                    torch.arange(len(states), device=states.device), positions
                ][:, None]
            else:
                query_times = query_timestamps[
                    torch.arange(len(states), device=states.device), positions
                ][:, None]
            ages = (query_times - end_times).clamp_min(0).to(dtype=states.dtype)
        decay = torch.exp(-ages[:, :, None] / half_lives[None, None, :])
        weights = F.softmax(self.velocity_router(memory[:, 1:]), dim=-1) * decay
        weights = weights * transition_mask[:, :, None]
        numerator = torch.einsum("nmk,nmd->nkd", weights, transitions)
        mass = weights.sum(dim=1)
        momentum = numerator / mass.clamp_min(torch.finfo(states.dtype).tiny)[:, :, None]
        energy = momentum.norm(dim=-1)
        return F.normalize(momentum, dim=-1), mass, energy

    def _velocity_evidence(
        self, tracks: tuple[Tensor, Tensor, Tensor] | None, candidates: Tensor
    ) -> Tensor:
        if tracks is None:
            return self.backbone.item_vectors()[candidates].new_zeros(candidates.shape)
        momentum, mass, energy = tracks
        item_vectors = self.backbone.item_vectors()[candidates]
        similarities = torch.einsum("ncd,nkd->nck", item_vectors, momentum)
        candidate_gate = self.velocity_candidate_gate(item_vectors)
        history_gate = self.velocity_history_gate(momentum).squeeze(-1)
        available = mass.gt(torch.finfo(mass.dtype).tiny)[:, None, :]
        gate_logits = (candidate_gate + history_gate[:, None, :]).masked_fill(
            ~available, -torch.inf
        )
        gates = F.softmax(gate_logits, dim=-1)
        gates = torch.where(available.any(-1, keepdim=True), gates, torch.zeros_like(gates))
        signal = (gates * similarities * energy[:, None, :]).sum(dim=-1)
        return F.softplus(self.velocity_gamma_unconstrained) * signal

    def sampled_logits(
        self,
        history: Tensor,
        targets: Tensor,
        negatives: Tensor,
        positions: Tensor,
        sequence_rows: Tensor | None = None,
        timestamps: Tensor | None = None,
        query_timestamps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        valid = history.ne(0)
        states = self.backbone.sequence_states(history, timestamps, query_timestamps)
        if sequence_rows is None:
            sequence_rows = torch.arange(len(history), device=history.device)
        selected_states = states[sequence_rows]
        selected_valid = valid[sequence_rows]
        rows = torch.arange(len(sequence_rows), device=history.device)
        context = selected_states[rows, positions]
        candidates = torch.cat([targets[:, None], negatives], dim=-1)
        item_vectors = self.backbone.item_vectors()[candidates]
        base = torch.einsum("nd,ncd->nc", context, item_vectors)
        memory, memory_mask, _ = self._memory(selected_states, selected_valid, positions)
        track_timestamps = None if timestamps is None else timestamps[sequence_rows]
        tracks = self._persistence_tracks(
            selected_states, selected_valid, positions, track_timestamps
        )
        velocity_tracks = self._velocity_tracks(
            selected_states,
            selected_valid,
            positions,
            track_timestamps,
            None if query_timestamps is None else query_timestamps[sequence_rows],
        )
        contextual = base + self._evidence(memory, memory_mask, candidates)
        contextual = contextual + self._persistence_evidence(tracks, candidates)
        contextual = contextual + self._velocity_evidence(velocity_tracks, candidates)
        return base / 0.05, contextual / 0.05

    @torch.no_grad()
    def score(
        self,
        history: Tensor,
        chunk_size: int = 512,
        timestamps: Tensor | None = None,
        query_timestamps: Tensor | None = None,
        inference_chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        valid = history.ne(0)
        lengths = valid.sum(-1)
        positions = (lengths - 1).clamp_min(0)
        states = self.backbone.sequence_states(
            history, timestamps, query_timestamps, inference_chunk_size
        )
        rows = torch.arange(len(history), device=history.device)
        context = states[rows, positions] * lengths.gt(0).unsqueeze(-1)
        all_items = self.backbone.item_vectors()
        base = context @ all_items.T
        memory, memory_mask, _ = self._memory(states, valid, positions)
        tracks = self._persistence_tracks(states, valid, positions, timestamps)
        velocity_tracks = self._velocity_tracks(
            states, valid, positions, timestamps, query_timestamps
        )
        output = []
        for start in range(0, len(all_items), chunk_size):
            candidates = torch.arange(
                start, min(start + chunk_size, len(all_items)), device=history.device
            ).expand(len(history), -1)
            output.append(
                self._evidence(memory, memory_mask, candidates)
                + self._persistence_evidence(tracks, candidates)
                + self._velocity_evidence(velocity_tracks, candidates)
            )
        evidence = torch.cat(output, dim=-1)
        evidence = torch.where(lengths.gt(0)[:, None], evidence, torch.zeros_like(evidence))
        return base, base + evidence

    def evidence_bounds(
        self,
        history: Tensor,
        candidates: Tensor,
        groups: int,
        timestamps: Tensor | None = None,
        query_timestamps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return exact evidence and valid temporal-partition lower/upper bounds."""
        if groups < 1 or groups > self.memory_window:
            raise ValueError("groups must be within [1, memory_window]")
        if len(self.key_projections) != 1:
            raise ValueError("evidence bounds currently require one contextual head")
        if self.persistence_scales:
            raise ValueError("evidence bounds require persistence scales disabled")
        valid = history.ne(0)
        lengths = valid.sum(-1)
        positions = (lengths - 1).clamp_min(0)
        states = self.backbone.sequence_states(history, timestamps, query_timestamps)
        memory, mask, _ = self._memory(states, valid, positions)
        exact = self._head_evidence(memory, mask, candidates, 0) * self.gamma[0]
        keys = F.normalize(self.key_projections[0](memory), dim=-1)
        queries = F.normalize(
            self.query_projections[0](self.backbone.item_vectors()[candidates]), dim=-1
        )
        relative_age = torch.arange(
            memory.shape[1] - 1, -1, -1, device=memory.device, dtype=memory.dtype
        ) / max(memory.shape[1], 1)
        logits = self.salience_heads[0](memory).squeeze(-1)
        logits = logits - F.softplus(self.decay_unconstrained[0]) * relative_age
        logits = logits.masked_fill(~mask, -torch.inf)
        weights = F.softmax(logits, dim=-1)
        mean_key = torch.einsum("nw,nwr->nr", weights, keys)
        linear = torch.einsum("ncr,nr->nc", queries, mean_key)
        lower_terms, upper_terms = [], []
        for group in torch.tensor_split(
            torch.arange(self.memory_window, device=history.device), groups
        ):
            group_weights = weights[:, group]
            mass = group_weights.sum(-1)
            safe_mass = mass.clamp_min(torch.finfo(weights.dtype).tiny)
            center = torch.einsum("nw,nwr->nr", group_weights, keys[:, group]) / safe_mass[:, None]
            distances = (keys[:, group] - center[:, None]).norm(dim=-1)
            radius = distances.masked_fill(~mask[:, group], 0).amax(-1)
            center_score = torch.einsum("ncr,nr->nc", queries, center)
            log_mass = safe_mass.log()[:, None]
            lower_terms.append(log_mass + center_score / self.temperature)
            upper_terms.append(log_mass + (center_score + radius[:, None]) / self.temperature)
        lower = self.gamma[0] * (
            self.temperature * torch.logsumexp(torch.stack(lower_terms), dim=0) - linear
        )
        upper = self.gamma[0] * (
            self.temperature * torch.logsumexp(torch.stack(upper_terms), dim=0) - linear
        )
        return exact, lower, upper


def _selected_positions(targets: Tensor, maximum: int) -> tuple[Tensor, Tensor, Tensor]:
    rows, positions, selected_targets = [], [], []
    for row in range(len(targets)):
        valid = targets[row].nonzero().flatten()
        if len(valid) > maximum:
            # Evenly cover early, middle and recent decisions deterministically.
            chosen = torch.linspace(0, len(valid) - 1, maximum, device=targets.device).long()
            valid = valid[chosen]
        rows.extend([row] * len(valid))
        positions.extend(valid.tolist())
        selected_targets.extend(targets[row, valid].tolist())
    device = targets.device
    return (
        torch.tensor(rows, device=device),
        torch.tensor(positions, device=device),
        torch.tensor(selected_targets, device=device),
    )


@torch.no_grad()
def _validation_ndcg10(
    model: ContextualEvidenceRanker,
    examples: list[Example],
    timestamp_histories: dict[str, list[int]],
    config: ExperimentConfig,
) -> tuple[float, float]:
    model.eval()
    base_values: list[float] = []
    contextual_values: list[float] = []
    for start in range(0, len(examples), 128):
        batch = examples[start : start + 128]
        history = torch.zeros(len(batch), config.max_length, dtype=torch.long, device=config.device)
        for row, example in enumerate(batch):
            history_values = example.history[-config.max_length :]
            history[row, : len(history_values)] = torch.tensor(history_values, device=config.device)
        timestamps = _timestamp_tensor(
            [example.user_id for example in batch],
            timestamp_histories,
            config.max_length,
            config.device,
        )
        query_timestamps = _query_timestamp_tensor(
            timestamps,
            history.ne(0).sum(1),
            torch.tensor([example.timestamp for example in batch], device=config.device),
        )
        score_sets = model.score(history, timestamps=timestamps, query_timestamps=query_timestamps)
        targets = torch.tensor([example.target for example in batch], device=config.device)
        for values, scores in zip((base_values, contextual_values), score_sets, strict=True):
            if not torch.isfinite(scores).all():
                raise FloatingPointError("non-finite validation scores")
            for row, example in enumerate(batch):
                scores[row, 0] = -torch.inf
                if example.seen:
                    scores[row, example.seen] = -torch.inf
            target_scores = scores.gather(1, targets[:, None])
            ranks = (scores > target_scores).sum(1) + 1
            values.extend(
                torch.where(
                    ranks <= 10, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks.float())
                )
                .cpu()
                .tolist()
            )
    return float(np.mean(base_values)), float(np.mean(contextual_values))


@torch.no_grad()
def _validation_cutoff_metrics(
    model: ContextualEvidenceRanker,
    examples: list[Example],
    timestamp_histories: dict[str, list[int]],
    config: ExperimentConfig,
    evidence_scale: float,
) -> dict[str, float]:
    """Measure one inference-only evidence scale on temporal validation queries."""
    model.eval()
    ranks: list[int] = []
    for start in range(0, len(examples), 128):
        batch = examples[start : start + 128]
        history = torch.zeros(len(batch), config.max_length, dtype=torch.long, device=config.device)
        for row, example in enumerate(batch):
            values = example.history[-config.max_length :]
            history[row, : len(values)] = torch.tensor(values, device=config.device)
        timestamps = _timestamp_tensor(
            [example.user_id for example in batch],
            timestamp_histories,
            config.max_length,
            config.device,
        )
        query_timestamps = _query_timestamp_tensor(
            timestamps,
            history.ne(0).sum(1),
            torch.tensor([example.timestamp for example in batch], device=config.device),
        )
        base, contextual = model.score(
            history, timestamps=timestamps, query_timestamps=query_timestamps
        )
        scores = base + evidence_scale * (contextual - base)
        targets = torch.tensor([example.target for example in batch], device=config.device)
        for row, example in enumerate(batch):
            scores[row, 0] = -torch.inf
            if example.seen:
                scores[row, example.seen] = -torch.inf
        target_scores = scores.gather(1, targets[:, None])
        ranks.extend(((scores > target_scores).sum(1) + 1).cpu().tolist())
    rank_array = np.asarray(ranks, dtype=np.float64)
    result: dict[str, float] = {}
    for cutoff in (10, 50, 200):
        hits = rank_array <= cutoff
        result[f"Recall@{cutoff}"] = float(hits.mean())
        result[f"NDCG@{cutoff}"] = float(
            np.where(hits, 1 / np.log2(rank_array + 1), 0).mean()
        )
        result[f"MRR@{cutoff}"] = float(np.where(hits, 1 / rank_array, 0).mean())
    return result


def _select_evidence_scale(
    model: ContextualEvidenceRanker,
    examples: list[Example],
    timestamp_histories: dict[str, list[int]],
    config: ExperimentConfig,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select the evidence scale by the registered primary validation metric."""
    candidates = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    evaluations = {
        str(scale): _validation_cutoff_metrics(
            model, examples, timestamp_histories, config, scale
        )
        for scale in candidates
    }
    selected = max(candidates, key=lambda scale: evaluations[str(scale)]["NDCG@10"])
    return selected, evaluations


def _apply_evidence_scale(model: ContextualEvidenceRanker, scale: float) -> None:
    """Fold a positive inference scale into gamma without changing the architecture."""
    if scale <= 0:
        raise ValueError("evidence scale must be positive")
    with torch.no_grad():
        target = model.gamma * scale
        model.gamma_unconstrained.copy_(torch.log(torch.expm1(target)))


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _interpolate_state(
    original: dict[str, Tensor], adapted: dict[str, Tensor], original_weight: float
) -> dict[str, Tensor]:
    if original.keys() != adapted.keys() or not 0 <= original_weight <= 1:
        raise ValueError("weight-soup states or interpolation weight are invalid")
    return {
        name: original_weight * original[name] + (1 - original_weight) * adapted[name]
        for name in original
    }


def _select_weight_soup(
    model: ContextualEvidenceRanker,
    original: dict[str, Tensor],
    adapted: dict[str, Tensor],
    examples: list[Example],
    timestamp_histories: dict[str, list[int]],
    config: ExperimentConfig,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Select the interpolation with the best aggregate relative validation metrics."""
    candidates = (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0)
    evaluations = {}
    for weight in candidates:
        model.load_state_dict(_interpolate_state(original, adapted, weight))
        evaluations[str(weight)] = _validation_cutoff_metrics(
            model, examples, timestamp_histories, config, 1.0
        )
    metrics = tuple(evaluations["1.0"])
    endpoint_best = {
        metric: max(evaluations["0.0"][metric], evaluations["1.0"][metric])
        for metric in metrics
    }
    selected = max(
        candidates,
        key=lambda weight: (
            sum(evaluations[str(weight)][metric] / endpoint_best[metric] for metric in metrics),
            min(
                evaluations[str(weight)][metric] / endpoint_best[metric] - 1
                for metric in metrics
            ),
        ),
    )
    model.load_state_dict(_interpolate_state(original, adapted, selected))
    return selected, evaluations


def _candidate_pool(targets: Tensor, count: int, n_items: int, popularity: Tensor) -> Tensor:
    uniform_count = count // 2
    uniform = torch.randint(1, n_items + 1, (*targets.shape, uniform_count), device=targets.device)
    popular = torch.multinomial(
        popularity, targets.numel() * (count - uniform_count), replacement=True
    ).reshape(*targets.shape, count - uniform_count)
    result = torch.cat([uniform, popular], dim=-1)
    collisions = result.eq(targets.unsqueeze(-1))
    while collisions.any():
        result[collisions] = torch.randint(
            1, n_items + 1, (int(collisions.sum()),), device=targets.device
        )
        collisions = result.eq(targets.unsqueeze(-1))
    return result


def _uniform_candidates(targets: Tensor, count: int, n_items: int) -> Tensor:
    result = torch.randint(1, n_items + 1, (*targets.shape, count), device=targets.device)
    collisions = result.eq(targets.unsqueeze(-1))
    while collisions.any():
        result[collisions] = torch.randint(
            1, n_items + 1, (int(collisions.sum()),), device=targets.device
        )
        collisions = result.eq(targets.unsqueeze(-1))
    return result


def _first_occurrence(history: Tensor, n_items: int) -> Tensor:
    """Return the first position of every item without a sequence/catalog expansion."""
    length = history.shape[1]
    first = torch.full((len(history), n_items + 1), length, dtype=torch.long, device=history.device)
    positions = torch.arange(length, device=history.device).expand_as(history)
    positions = positions.masked_fill(history.eq(0), length)
    first.scatter_reduce_(1, history, positions, reduce="amin", include_self=True)
    first[:, 0] = 0
    return first


def _exclude_seen_all_positions(
    candidates: Tensor, history: Tensor, targets: Tensor, n_items: int
) -> Tensor:
    """Resample negatives exposed at or before each causal training position."""
    first = _first_occurrence(history, n_items)
    positions = torch.arange(history.shape[1], device=history.device)[None, :, None]
    candidate_first = first[:, None, :].expand(-1, history.shape[1], -1).gather(2, candidates)
    collisions = candidate_first.le(positions) | candidates.eq(targets.unsqueeze(-1))
    while collisions.any():
        candidates[collisions] = torch.randint(
            1, n_items + 1, (int(collisions.sum()),), device=history.device
        )
        candidate_first = first[:, None, :].expand(-1, history.shape[1], -1).gather(2, candidates)
        collisions = candidate_first.le(positions) | candidates.eq(targets.unsqueeze(-1))
    return candidates


def _exclude_seen_selected_positions(
    candidates: Tensor,
    history: Tensor,
    sequence_rows: Tensor,
    positions: Tensor,
    targets: Tensor,
    n_items: int,
) -> Tensor:
    first = _first_occurrence(history, n_items)[sequence_rows]
    collisions = first.gather(1, candidates).le(positions[:, None]) | candidates.eq(
        targets[:, None]
    )
    while collisions.any():
        candidates[collisions] = torch.randint(
            1, n_items + 1, (int(collisions.sum()),), device=history.device
        )
        collisions = first.gather(1, candidates).le(positions[:, None]) | candidates.eq(
            targets[:, None]
        )
    return candidates


@torch.no_grad()
def _mine_candidates(
    model: ContextualBackbone,
    history: Tensor,
    targets: Tensor,
    candidates: Tensor,
    count: int,
    timestamps: Tensor | None = None,
    query_timestamps: Tensor | None = None,
) -> Tensor:
    hard_count = count * 3 // 4
    states = model.sequence_states(history, timestamps, query_timestamps)
    item_vectors = model.item_vectors()[candidates]
    scores = torch.einsum("bld,blkd->blk", states, item_vectors)
    scores.masked_fill_(candidates.eq(targets.unsqueeze(-1)), -torch.inf)
    hard = candidates.gather(-1, scores.topk(hard_count, dim=-1).indices)
    exploration = candidates[..., : count - hard_count]
    return torch.cat([hard, exploration], dim=-1)


@torch.no_grad()
def _mine_selected_candidates(
    model: ContextualBackbone,
    history: Tensor,
    sequence_rows: Tensor,
    positions: Tensor,
    targets: Tensor,
    candidates: Tensor,
    count: int,
    timestamps: Tensor | None = None,
    query_timestamps: Tensor | None = None,
) -> Tensor:
    hard_count = count * 3 // 4
    states = model.sequence_states(history, timestamps, query_timestamps)[sequence_rows, positions]
    item_vectors = model.item_vectors()[candidates]
    scores = torch.einsum("nd,nkd->nk", states, item_vectors)
    scores.masked_fill_(candidates.eq(targets.unsqueeze(-1)), -torch.inf)
    hard = candidates.gather(-1, scores.topk(hard_count, dim=-1).indices)
    exploration = candidates[:, : count - hard_count]
    return torch.cat([hard, exploration], dim=-1)


def _fit_base(
    model: ContextualEvidenceRanker,
    rows: list[dict[str, Any]],
    counts: np.ndarray,
    config: ExperimentConfig,
    epochs: int,
    validation: list[Example] | None = None,
    validation_timestamps: dict[str, list[int]] | None = None,
) -> tuple[list[dict[str, float]], int]:
    contextual_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    }
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) not in contextual_ids)
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    popularity = torch.tensor(counts, device=config.device).float().clamp_min(0)
    popularity[0] = 0
    popularity /= popularity.sum().clamp_min(1)
    best_metric, best_epoch, best = -1.0, epochs, None
    training_log: list[dict[str, float]] = []
    batch_size = config.base_batch_size or config.batch_size
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            batch = [rows[int(index)] for index in order[start : start + batch_size]]
            history, all_targets, timestamps, query_timestamps = _sequence_tensors_with_timestamps(
                batch, config.max_length, config.device
            )
            valid = all_targets.ne(0)
            if not valid.any():
                continue
            pool = _candidate_pool(
                all_targets,
                (config.sampled_negatives or 128) * 2,
                model.backbone.items.num_embeddings - 1,
                popularity,
            )
            pool = _exclude_seen_all_positions(
                pool, history, all_targets, model.backbone.items.num_embeddings - 1
            )
            negatives = _mine_candidates(
                model.backbone,
                history,
                all_targets,
                pool,
                config.sampled_negatives or 128,
                timestamps,
                query_timestamps,
            )
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if (config.mixed_precision and config.device.startswith("cuda"))
                else nullcontext()
            )
            with amp:
                states = model.backbone.sequence_states(history, timestamps, query_timestamps)[
                    valid
                ]
                targets = all_targets[valid]
                item_vectors = model.backbone.item_vectors()
                positive = (states * item_vectors[targets]).sum(-1, keepdim=True)
                negative = torch.einsum("nd,nkd->nk", states, item_vectors[negatives[valid]])
                logits = torch.cat([positive, negative], dim=-1) / 0.05
                loss = F.cross_entropy(
                    logits,
                    torch.zeros(len(targets), dtype=torch.long, device=config.device),
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        entry = {"epoch": float(epoch + 1), "base_loss": float(np.mean(losses))}
        if validation is not None and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            assert validation_timestamps is not None
            base_ndcg, _ = _validation_ndcg10(model, validation, validation_timestamps, config)
            entry["validation_base_ndcg10"] = base_ndcg
            if base_ndcg > best_metric:
                best_metric, best_epoch = base_ndcg, epoch + 1
                best = copy.deepcopy(model.backbone.state_dict())
        training_log.append(entry)
        print(json.dumps({"stage": "base", **entry}), flush=True)
    if best is not None:
        model.backbone.load_state_dict(best)
    model.eval()
    return training_log, best_epoch


def _fit_context(
    model: ContextualEvidenceRanker,
    rows: list[dict[str, Any]],
    counts: np.ndarray,
    config: ExperimentConfig,
    epochs: int,
    validation: list[Example] | None = None,
    validation_timestamps: dict[str, list[int]] | None = None,
    *,
    joint: bool = False,
) -> tuple[list[dict[str, float]], int]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(joint or not name.startswith("backbone."))
    contextual_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimized_parameters = list(model.parameters()) if joint else contextual_parameters
    optimizer_groups: list[dict[str, Any]]
    if joint:
        optimizer_groups = [
            {
                "params": list(model.backbone.parameters()),
                "lr": config.learning_rate * config.contextual_joint_backbone_lr_scale,
            },
            {"params": contextual_parameters, "lr": config.learning_rate},
        ]
    else:
        optimizer_groups = [{"params": contextual_parameters, "lr": config.learning_rate}]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed + (2 if joint else 1))
    popularity = torch.tensor(counts, device=config.device).float().clamp_min(0)
    popularity[0] = 0
    popularity /= popularity.sum().clamp_min(1)
    best_metric, best_epoch, best = -1.0, epochs, None
    training_log: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        if not joint:
            model.backbone.eval()
        losses = []
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), config.batch_size):
            batch = [rows[int(index)] for index in order[start : start + config.batch_size]]
            history, all_targets, timestamps, query_timestamps = _sequence_tensors_with_timestamps(
                batch, config.max_length, config.device
            )
            selected_rows, positions, targets = _selected_positions(
                all_targets, config.contextual_positions_per_user
            )
            if not len(targets):
                continue
            negative_count = config.sampled_negatives or 128
            if config.contextual_hard_negatives:
                pool = _candidate_pool(
                    targets,
                    negative_count * 2,
                    model.backbone.items.num_embeddings - 1,
                    popularity,
                )
                pool = _exclude_seen_selected_positions(
                    pool,
                    history,
                    selected_rows,
                    positions,
                    targets,
                    model.backbone.items.num_embeddings - 1,
                )
                negatives = _mine_selected_candidates(
                    model.backbone,
                    history,
                    selected_rows,
                    positions,
                    targets,
                    pool,
                    negative_count,
                    timestamps,
                    query_timestamps,
                )
            else:
                negatives = _uniform_candidates(
                    targets,
                    negative_count,
                    model.backbone.items.num_embeddings - 1,
                )
                negatives = _exclude_seen_selected_positions(
                    negatives,
                    history,
                    selected_rows,
                    positions,
                    targets,
                    model.backbone.items.num_embeddings - 1,
                )
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if (config.mixed_precision and config.device.startswith("cuda"))
                else nullcontext()
            )
            with amp:
                _, contextual = model.sampled_logits(
                    history,
                    targets,
                    negatives,
                    positions,
                    selected_rows,
                    timestamps,
                    query_timestamps,
                )
                loss = F.cross_entropy(
                    contextual,
                    torch.zeros(len(targets), dtype=torch.long, device=config.device),
                )
            loss.backward()
            nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        entry = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(losses)),
            "gamma_mean": float(model.gamma.detach().mean()),
        }
        if validation is not None and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            assert validation_timestamps is not None
            base_ndcg, contextual_ndcg = _validation_ndcg10(
                model, validation, validation_timestamps, config
            )
            entry.update(
                {"validation_base_ndcg10": base_ndcg, "validation_ndcg10": contextual_ndcg}
            )
            if contextual_ndcg > best_metric:
                best_metric, best_epoch = contextual_ndcg, epoch + 1
                best = copy.deepcopy(
                    model.state_dict()
                    if joint
                    else {
                        name: value
                        for name, value in model.state_dict().items()
                        if not name.startswith("backbone.")
                    }
                )
        training_log.append(entry)
        print(json.dumps({"stage": "joint" if joint else "context", **entry}), flush=True)
    if best is not None:
        model.load_state_dict(best, strict=joint)
    model.eval()
    return training_log, best_epoch


def run_fair_contextual(
    config: ExperimentConfig,
    data: Path,
    protocol_directory: Path,
    output: Path,
    *,
    validation_only: bool = False,
    selection_recipe: Path | None = None,
) -> Path:
    """Select duration temporally or use a frozen recipe, then export rankings."""
    if output.exists():
        raise FileExistsError(output)
    if validation_only and output.suffix != ".json":
        raise ValueError("Validation-only output must have a .json suffix")
    if validation_only and selection_recipe is not None:
        raise ValueError("Validation-only execution cannot use a fixed selection recipe")
    protocol = FairProtocol.model_validate_json(
        (protocol_directory / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.protocol_hash != protocol.expected_hash():
        raise ValueError("Protocol manifest hash is invalid")
    recipe = (
        _read_selection_recipe(selection_recipe, protocol.protocol_hash, config)
        if selection_recipe is not None
        else None
    )
    if config.max_length != protocol.history_length or config.positive_threshold != 0.5:
        raise ValueError("Contextual config must use protocol history and all ratings")
    if sha256(data / "interactions.parquet") != protocol.interactions_sha256:
        raise ValueError("Contextual data does not match the protocol")
    if sha256(data / "catalog.json") != protocol.catalog_sha256:
        raise ValueError("Contextual catalog does not match the protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sequence_path = protocol_directory / "hstu_sequences.csv"
    if sha256(sequence_path) != protocol.sequence_sha256:
        raise ValueError("Protocol sequence artifact has changed")
    rows = _sequences(sequence_path)
    training_sequence_path = protocol_directory / "hstu_training_sequences.csv"
    if protocol.training_sequence_sha256 is None:
        final_rows = rows
    else:
        if sha256(training_sequence_path) != protocol.training_sequence_sha256:
            raise ValueError("Protocol training sequence artifact has changed")
        final_rows = _sequences(training_sequence_path)

    def create_model() -> ContextualEvidenceRanker:
        return ContextualEvidenceRanker(
            protocol.item_count,
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
        ).to(config.device)

    base_selection_log: list[dict[str, float]] = []
    context_selection_log: list[dict[str, float]] = []
    joint_selection_log: list[dict[str, float]] = []
    weight_soup_validation: dict[str, dict[str, float]] = {}
    evidence_scale_validation: dict[str, dict[str, float]] = {}
    if recipe is not None:
        best_base_epoch = recipe.selected_base_epoch
        best_context_epoch = recipe.selected_context_epoch
        best_joint_epoch = recipe.selected_joint_epoch
        selected_evidence_scale = recipe.selected_evidence_scale
        selected_original_weight = recipe.selected_original_weight
        if (
            best_base_epoch > config.epochs
            or best_context_epoch > config.contextual_epochs
            or best_joint_epoch > config.contextual_joint_epochs
        ):
            raise ValueError("Fixed selection recipe exceeds the configured training budget")
    else:
        validation_rows = _truncate_rows(rows, 1)
        validation_examples = _event_queries(rows, -2, config.max_length, "model_selection")
        validation_timestamps = {
            row["user_id"]: row["timestamps"][:-2][-config.max_length :] for row in rows
        }
        validation_counts = _counts(validation_rows, protocol.item_count)
        shadow = create_model()
        base_selection_log, best_base_epoch = _fit_base(
            shadow,
            validation_rows,
            validation_counts,
            config,
            config.epochs,
            validation_examples,
            validation_timestamps,
        )
        context_selection_log, best_context_epoch = _fit_context(
            shadow,
            validation_rows,
            validation_counts,
            config,
            config.contextual_epochs,
            validation_examples,
            validation_timestamps,
        )
        context_best = max(row.get("validation_ndcg10", -1.0) for row in context_selection_log)
        best_joint_epoch = 0
        selected_original_weight = 1.0
        if config.contextual_joint_epochs:
            before_joint = _cpu_state(shadow)
            joint_selection_log, best_joint_epoch = _fit_context(
                shadow,
                validation_rows,
                validation_counts,
                config,
                config.contextual_joint_epochs,
                validation_examples,
                validation_timestamps,
                joint=True,
            )
            joint_best = max(row.get("validation_ndcg10", -1.0) for row in joint_selection_log)
            if joint_best <= context_best:
                shadow.load_state_dict(before_joint)
                best_joint_epoch = 0
            else:
                selected_original_weight, weight_soup_validation = _select_weight_soup(
                    shadow,
                    before_joint,
                    _cpu_state(shadow),
                    validation_examples,
                    validation_timestamps,
                    config,
                )
        selected_evidence_scale, evidence_scale_validation = _select_evidence_scale(
            shadow, validation_examples, validation_timestamps, config
        )
    if validation_only:
        selected_log = joint_selection_log if best_joint_epoch else context_selection_log
        best = max(selected_log, key=lambda row: row.get("validation_ndcg10", -1.0))
        _apply_evidence_scale(shadow, selected_evidence_scale)
        output.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "test_evaluated": False,
                    "split": "temporal_validation",
                    "selection_metric": "NDCG@10",
                    "protocol_hash": protocol.protocol_hash,
                    "config": config.model_dump(),
                    "selected_base_epoch": best_base_epoch,
                    "selected_context_epoch": best_context_epoch,
                    "selected_joint_epoch": best_joint_epoch,
                    "selected_evidence_scale": selected_evidence_scale,
                    "evidence_scale_validation": evidence_scale_validation,
                    "selected_original_weight": selected_original_weight,
                    "weight_soup_validation": weight_soup_validation,
                    "validation_ndcg10": best["validation_ndcg10"],
                    "validation_base_ndcg10": best["validation_base_ndcg10"],
                    "training": {
                        "base": base_selection_log,
                        "context": context_selection_log,
                        "joint": joint_selection_log,
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                    "parameters": sum(parameter.numel() for parameter in shadow.parameters()),
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
                    if config.device.startswith("cuda")
                    else 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    seed_everything(config.seed)
    model = create_model()
    counts = _counts(final_rows, protocol.item_count)
    final_base_log, _ = _fit_base(model, final_rows, counts, config, best_base_epoch)
    final_context_log, _ = _fit_context(model, final_rows, counts, config, best_context_epoch)
    final_joint_log: list[dict[str, float]] = []
    if best_joint_epoch:
        before_final_joint = _cpu_state(model)
        final_joint_log, _ = _fit_context(
            model,
            final_rows,
            counts,
            config,
            best_joint_epoch,
            joint=True,
        )
        model.load_state_dict(
            _interpolate_state(before_final_joint, _cpu_state(model), selected_original_weight)
        )
    _apply_evidence_scale(model, selected_evidence_scale)
    weights_path = output.with_suffix(".safetensors")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        weights_path,
    )
    queries = pl.read_parquet(protocol_directory / "queries.parquet")
    test_timestamps = {row["user_id"]: row["timestamps"][:-1][-config.max_length :] for row in rows}
    paths = {
        "contextual-base": output.with_name(f"{output.stem}-base{output.suffix}"),
        "contextual-evidence": output,
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in paths}
    inference_started = time.perf_counter()
    for start in range(0, len(queries), 128):
        batch = list(queries.slice(start, 128).iter_rows(named=True))
        history = torch.zeros(len(batch), config.max_length, dtype=torch.long, device=config.device)
        for row, query in enumerate(batch):
            values = query["history"][-config.max_length :]
            history[row, : len(values)] = torch.tensor(values, device=config.device)
        timestamps = _timestamp_tensor(
            [str(query["user_id"]) for query in batch],
            test_timestamps,
            config.max_length,
            config.device,
        )
        query_timestamps = _query_timestamp_tensor(
            timestamps,
            history.ne(0).sum(1),
            torch.tensor([int(query["timestamp"]) for query in batch], device=config.device),
        )
        base, contextual = model.score(
            history, timestamps=timestamps, query_timestamps=query_timestamps
        )
        if not torch.isfinite(base).all() or not torch.isfinite(contextual).all():
            raise FloatingPointError("non-finite test scores")
        for row, query in enumerate(batch):
            allowed = np.ones(protocol.item_count + 1, dtype=bool)
            allowed[0] = False
            allowed[query["seen"]] = False
            if protocol.candidate_item_ids is not None:
                allowed[:] = False
                allowed[protocol.candidate_item_ids] = True
                allowed[query["seen"]] = False
            eligible = np.flatnonzero(allowed)
            limit = min(200, len(eligible))
            for name, scores in zip(paths, (base, contextual), strict=True):
                values = scores[row, eligible].float().cpu().numpy()
                ranking = eligible[np.argsort(-values, kind="stable")[:limit]]
                predictions[name].extend(
                    {
                        "query_id": query["query_id"],
                        "model": name,
                        "seed": config.seed,
                        "protocol_hash": protocol.protocol_hash,
                        "rank": rank,
                        "movie_id": int(movie_id),
                    }
                    for rank, movie_id in enumerate(ranking, 1)
                )
    inference_seconds = time.perf_counter() - inference_started
    hashes = {}
    for name, path in paths.items():
        pl.DataFrame(predictions[name]).write_parquet(path)
        hashes[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "schema_version": 1,
        "model": "contextual-evidence",
        "architecture": (
            "candidate-conditioned contextual evidence over a causal SASRec backbone "
            "with logarithmic event-gap and query-time attention conditioning"
        ),
        "seed": config.seed,
        "protocol_hash": protocol.protocol_hash,
        "config": config.model_dump(),
        "model_selection": {
            "split": (
                "fixed development recipe"
                if recipe is not None
                else "second-last event; training excludes final two events"
            ),
            "metric": "NDCG@10",
            "selected_base_epoch": best_base_epoch,
            "selected_context_epoch": best_context_epoch,
            "selected_joint_epoch": best_joint_epoch,
            "selected_evidence_scale": selected_evidence_scale,
            "evidence_scale_validation": evidence_scale_validation,
            "selected_original_weight": selected_original_weight,
            "weight_soup_validation": weight_soup_validation,
            "selection_recipe": (
                {
                    "path": selection_recipe.name,
                    "sha256": sha256(selection_recipe),
                    "source_protocol_hash": recipe.source_protocol_hash,
                    "config_fingerprint": recipe.config_fingerprint,
                }
                if recipe is not None and selection_recipe is not None
                else None
            ),
        },
        "training_positions": (
            f"up to {config.contextual_positions_per_user} deterministic positions "
            "per user and epoch"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "hardware": {
            "platform": platform.platform(),
            "device": config.device,
            "gpu": torch.cuda.get_device_name() if config.device.startswith("cuda") else None,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "inference_qps": len(queries) / inference_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if config.device.startswith("cuda")
        else 0,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "contextual_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("backbone.")
        ),
        "weights": {
            "path": weights_path.name,
            "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        },
        "predictions": hashes,
        "training": {
            "selection_base": base_selection_log,
            "selection_context": context_selection_log,
            "selection_joint": joint_selection_log,
            "final_base": final_base_log,
            "final_context": final_context_log,
            "final_joint": final_joint_log,
        },
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output
