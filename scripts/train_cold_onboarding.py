"""Train the small onboarding adapter without changing the warm checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from vassago.config import ExperimentConfig
from vassago.contextual_ranker import ContextualEvidenceRanker, _sequences
from vassago.fair_benchmark import FairProtocol
from vassago.training import seed_everything


def _rows(
    sequences: list[dict[str, Any]], split: str, window: int
) -> list[dict[str, Any]]:
    result = []
    for row in sequences:
        if len(row["items"]) < 4:
            continue
        if split == "train":
            history, target = row["items"][:-3], row["items"][-3]
        elif split == "validation":
            history, target = row["items"][:-2], row["items"][-2]
        else:
            history, target = row["items"][:-1], row["items"][-1]
        history = history[-window:]
        seen = history if split == "train" else row["items"][: -2 if split == "validation" else 1]
        result.append(
            {
                "history": history,
                "target": int(target),
                "seen": set(seen),
            }
        )
    return result


@torch.inference_mode()
def _metrics(
    model: ContextualEvidenceRanker,
    rows: list[dict[str, Any]],
    max_length: int,
    device: str,
) -> dict[str, float]:
    sums = {
        f"{name}@{cutoff}": 0.0
        for cutoff in (10, 50, 200)
        for name in ("Recall", "NDCG", "MRR")
    }
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        history = torch.zeros(len(batch), max_length, dtype=torch.long, device=device)
        for index, row in enumerate(batch):
            values = row["history"][-10:]
            history[index, : len(values)] = torch.tensor(values, device=device)
        scores = model.score(history)[1].float().cpu().numpy()
        for index, row in enumerate(batch):
            allowed = np.ones(scores.shape[1], dtype=bool)
            allowed[0] = False
            allowed[list(row["seen"])] = False
            rank = 1 + int(np.sum(scores[index, allowed] > scores[index, row["target"]]))
            for cutoff in (10, 50, 200):
                hit = rank <= cutoff
                sums[f"Recall@{cutoff}"] += float(hit)
                sums[f"NDCG@{cutoff}"] += 1 / np.log2(rank + 1) if hit else 0.0
                sums[f"MRR@{cutoff}"] += 1 / rank if hit else 0.0
    return {name: value / len(rows) for name, value in sums.items()}


def _fit_epoch(
    model: ContextualEvidenceRanker,
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    window: int,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> float:
    model.train()
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in backbone_ids)
    order = rng.permutation(len(rows))
    losses = []
    item_count = model.backbone.items.num_embeddings - 1
    for start in range(0, len(rows), config.batch_size):
        batch = [rows[int(index)] for index in order[start : start + config.batch_size]]
        history = torch.zeros(len(batch), config.max_length, dtype=torch.long, device=config.device)
        lengths = []
        for index, row in enumerate(batch):
            values = row["history"][-window:]
            history[index, : len(values)] = torch.tensor(values, device=config.device)
            lengths.append(len(values))
        lengths_tensor = torch.tensor(lengths, device=config.device)
        states = model.backbone.sequence_states(history)
        context = states[
            torch.arange(len(batch), device=config.device),
            (lengths_tensor - 1).clamp_min(0),
        ]
        targets = torch.tensor([row["target"] for row in batch], device=config.device)
        item_vectors = model.backbone.item_vectors()
        negatives = torch.randint(
            1,
            item_count + 1,
            (len(batch), config.sampled_negatives or 128),
            device=config.device,
        )
        while negatives.eq(targets[:, None]).any():
            collisions = negatives.eq(targets[:, None])
            negatives[collisions] = torch.randint(
                1,
                item_count + 1,
                (int(collisions.sum()),),
                device=config.device,
            )
        logits = torch.cat(
            [
                (context * item_vectors[targets]).sum(-1, keepdim=True),
                torch.einsum("bd,bkd->bk", context, item_vectors[negatives]),
            ],
            dim=1,
        ) / 0.05
        loss = F.cross_entropy(
            logits,
            torch.zeros(len(batch), dtype=torch.long, device=config.device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.window < 1 or args.learning_rate <= 0:
        parser.error("epochs, window and learning-rate must be positive")
    config = ExperimentConfig.read(args.config).model_copy(update={"device": args.device})
    if args.window > config.max_length:
        parser.error("window cannot exceed the configured max_length")
    protocol = FairProtocol.model_validate_json((args.protocol / "protocol.json").read_text())
    sequences = _sequences(args.protocol / "hstu_sequences.csv")
    train_rows = _rows(sequences, "train", args.window)
    validation_rows = _rows(sequences, "validation", args.window)
    seed_everything(config.seed + 701)
    model = ContextualEvidenceRanker(
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
    ).to(args.device)
    model.load_state_dict(load_file(args.warm_weights, device=args.device), strict=True)
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    rng = np.random.default_rng(config.seed + 701)
    best_metric = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    training: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        loss = _fit_epoch(model, train_rows, config, args.window, optimizer, rng)
        metrics = _metrics(model, validation_rows, config.max_length, args.device)
        entry = {
            "epoch": float(epoch + 1),
            "loss": loss,
            **{f"validation_{k}": v for k, v in metrics.items()},
        }
        training.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["NDCG@10"] > best_metric:
            best_metric = metrics["NDCG@10"]
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("cold adapter did not produce a checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(best_state, args.output)
    manifest = {
        "schema_version": 1,
        "model": "contextual-evidence-cold-onboarding",
        "warm_weights_sha256": hashlib.sha256(args.warm_weights.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "protocol_hash": protocol.protocol_hash,
        "selection": {
            "split": "second-last interaction",
            "metric": "NDCG@10",
            "best_epoch": best_epoch,
        },
        "window": args.window,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
        ),
        "parameters": sum(value.numel() for value in best_state.values()),
        "training": training,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
