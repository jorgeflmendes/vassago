"""Deterministic streaming trainers with validation-selected model parameters."""

import copy
import random
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vassago.config import ExperimentConfig
from vassago.data import Example, Movie
from vassago.models import SASRec
from vassago.semantic_ids import SIDExpert
from vassago.serving import history_tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def batches(examples: Iterator[Example], size: int) -> Iterator[list[Example]]:
    batch = []
    for example in examples:
        batch.append(example)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def sequence_loss(
    model: SASRec,
    batch: list[Example],
    movies: list[Movie],
    config: ExperimentConfig,
    warm: np.ndarray | None = None,
) -> torch.Tensor:
    history = history_tensor([e.history for e in batch], config.max_length, config.device)
    targets = torch.tensor([e.target for e in batch], device=config.device)
    logits = model.score(history) / 0.1
    availability = torch.tensor(
        [2**62] + [max(m.available_at, m.metadata_available_at) for m in movies],
        device=config.device,
    )
    timestamps = torch.tensor([e.timestamp for e in batch], device=config.device)
    allowed = availability.unsqueeze(0) <= timestamps.unsqueeze(1)
    if warm is not None:
        allowed &= torch.tensor(warm > 0, device=config.device).unsqueeze(0)
    allowed.scatter_(1, history, False)
    for row, event in enumerate(batch):
        if event.seen:
            allowed[row, event.seen] = False
    valid = allowed.gather(1, targets[:, None]).squeeze(1) & history.ne(0).any(1)
    if not valid.any():
        return logits.sum() * 0
    loss = F.cross_entropy(
        logits[valid].masked_fill(~allowed[valid], -1e9), targets[valid], reduction="none"
    )
    if config.rating_weighted:
        loss *= torch.tensor([e.rating / 5 for e in batch], device=config.device)[valid]
    return loss.mean()


def train_sequence(
    model: SASRec,
    examples: Callable[[str], Iterator[Example]],
    movies: list[Movie],
    config: ExperimentConfig,
    warm: np.ndarray | None = None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best: dict[str, Any] | None = None
    log = []
    for epoch in range(config.epochs):
        model.train()
        total, steps = 0.0, 0
        for batch in batches(examples("train"), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if config.mixed_precision and config.device.startswith("cuda")
                else nullcontext()
            )
            with amp:
                loss = sequence_loss(model, batch, movies, config, warm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += loss.item()
            steps += 1
        model.eval()
        validation = []
        with torch.no_grad():
            for batch in batches(examples("base_validation"), config.batch_size):
                validation.append(sequence_loss(model, batch, movies, config, warm).item())
        value = float(np.mean(validation)) if validation else total / max(steps, 1)
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
        log.append({"epoch": epoch, "loss": total / max(steps, 1), "validation_loss": value})
    if best is None:
        raise ValueError("No finite training checkpoint")
    model.load_state_dict(best)
    model.eval()
    return log


def train_sid(
    model: SIDExpert,
    examples: Callable[[str], Iterator[Example]],
    config: ExperimentConfig,
    warm: np.ndarray,
    differentiable: bool,
) -> list[float]:
    teacher = copy.deepcopy(model.tokenizer).eval().requires_grad_(False)
    model.history_encoder.flatten_parameters()
    model.tokenizer.requires_grad_(differentiable)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best, best_loss, log = None, float("inf"), []
    for epoch in range(config.epochs):
        model.train()
        exploration = max(0.02, 0.2 * (1 - epoch / config.epochs)) if differentiable else 0
        for batch in batches((e for e in examples("train") if e.history), config.batch_size):
            history = history_tensor([e.history for e in batch], config.max_length, config.device)
            targets = torch.tensor([e.target for e in batch], device=config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.recommendation_loss(history, targets, differentiable, exploration, teacher)
            if differentiable:
                item_ids = torch.unique(torch.cat([history.flatten(), targets]))
                item_ids = item_ids[item_ids.ne(0)]
                vectors = model.embeddings[item_ids]
                current, _, _ = model.tokenizer(vectors, exploration=exploration)
                with torch.no_grad():
                    anchored, _, _ = teacher(vectors, differentiable=False)
                loss = (
                    loss
                    + 0.25 * model.tokenizer.reconstruction_loss(vectors, exploration)
                    + 0.5 * F.mse_loss(current, anchored)
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        validation = []
        with torch.no_grad():
            for batch in batches(
                (e for e in examples("base_validation") if e.history), config.batch_size
            ):
                h = history_tensor([e.history for e in batch], config.max_length, config.device)
                targets = torch.tensor([e.target for e in batch], device=config.device)
                validation.append(
                    model.recommendation_loss(
                        h, targets, differentiable, target_tokenizer=teacher
                    ).item()
                )
        value = float(np.mean(validation)) if validation else 0.0
        if value < best_loss:
            best, best_loss = copy.deepcopy(model.state_dict()), value
        log.append(value)
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return log
