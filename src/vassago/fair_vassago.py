"""VASSAGO training and prediction export for the immutable fair protocol."""

import csv
import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from vassago.config import ExperimentConfig
from vassago.data import Example, Movie
from vassago.fair_benchmark import FairProtocol, sha256
from vassago.features import TextEncoder, movie_text
from vassago.models import SASRec
from vassago.retrieval import fusion_features
from vassago.serving import VassagoRanker
from vassago.training import seed_everything, train_sid


def _sequences(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "user_id": row["user_id"],
            "items": [int(value) for value in row["sequence_item_ids"].split(",")],
            "ratings": [float(value) for value in row["sequence_ratings"].split(",")],
            "timestamps": [int(value) for value in row["sequence_timestamps"].split(",")],
        }
        for row in rows
    ]


def _sequence_tensors(
    rows: list[dict[str, Any]], max_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
    targets = torch.zeros_like(history)
    for index, row in enumerate(rows):
        # The last event is the untouched test target. Match DatasetV2's most recent window.
        training = row["items"][:-1][-max_length - 1 :]
        inputs, outputs = training[:-1], training[1:]
        if inputs:
            history[index, : len(inputs)] = torch.tensor(inputs, device=device)
            targets[index, : len(outputs)] = torch.tensor(outputs, device=device)
    return history, targets


def _train_sequence_model(
    model: SASRec,
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    seed: int,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    log = []
    for _ in range(config.epochs):
        model.train()
        order = rng.permutation(len(rows))
        losses = []
        for start in range(0, len(rows), config.batch_size):
            selected = [rows[int(index)] for index in order[start : start + config.batch_size]]
            history, targets = _sequence_tensors(selected, config.max_length, config.device)
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if config.mixed_precision and config.device.startswith("cuda")
                else nullcontext()
            )
            with amp:
                states = model.sequence_states(history)
                valid = targets.ne(0)
                state = states[valid]
                target = targets[valid]
                item_vectors = model.item_vectors()
                if config.sampled_negatives is None:
                    logits = state @ item_vectors.T / 0.05
                else:
                    negative_ids = torch.randint(
                        1,
                        len(item_vectors),
                        (len(target), config.sampled_negatives),
                        device=config.device,
                    )
                    collisions = negative_ids.eq(target.unsqueeze(1))
                    while collisions.any():
                        negative_ids[collisions] = torch.randint(
                            1,
                            len(item_vectors),
                            (int(collisions.sum()),),
                            device=config.device,
                        )
                        collisions = negative_ids.eq(target.unsqueeze(1))
                    positive = (state * item_vectors[target]).sum(1, keepdim=True)
                    negatives = torch.einsum("nd,nkd->nk", state, item_vectors[negative_ids])
                    logits = torch.cat([positive, negatives], dim=1) / 0.05
                    target = torch.zeros_like(target)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        log.append(float(np.mean(losses)))
    model.eval()
    return log


def _training_queries(rows: list[dict[str, Any]], max_length: int) -> list[Example]:
    examples = []
    for row in rows:
        training_items = row["items"][:-1]
        if len(training_items) < 2:
            continue
        history = training_items[:-1][-max_length:]
        examples.append(
            Example(
                user_id=row["user_id"],
                history=history,
                target=training_items[-1],
                timestamp=row["timestamps"][-2],
                rating=row["ratings"][-2],
                partition="train",
                seen=sorted(set(history)),
            )
        )
    return examples


def _event_queries(
    rows: list[dict[str, Any]], target_index: int, max_length: int, partition: str
) -> list[Example]:
    examples = []
    for row in rows:
        if len(row["items"]) < abs(target_index):
            continue
        history = row["items"][:target_index][-max_length:]
        history_timestamps = row["timestamps"][:target_index][-max_length:]
        if not history_timestamps:
            continue
        examples.append(
            Example(
                user_id=row["user_id"],
                history=history,
                target=row["items"][target_index],
                timestamp=history_timestamps[-1],
                rating=row["ratings"][target_index],
                partition=partition,
                seen=sorted(set(history)),
            )
        )
    return examples


def _truncate_rows(rows: list[dict[str, Any]], remove_last: int) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "items": row["items"][:-remove_last],
            "ratings": row["ratings"][:-remove_last],
            "timestamps": row["timestamps"][:-remove_last],
        }
        for row in rows
        if len(row["items"]) > remove_last + 2
    ]


def _counts(rows: list[dict[str, Any]], item_count: int) -> np.ndarray:
    counts = np.zeros(item_count + 1, dtype=np.float32)
    for row in rows:
        for item in row["items"][:-1]:
            counts[item] += 1
    return counts


def _fit_experts(
    model: VassagoRanker,
    rows: list[dict[str, Any]],
    counts: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    training: dict[str, Any] = {
        "id": _train_sequence_model(model.id_expert, rows, config, seed),
        "semantic": _train_sequence_model(model.semantic_expert, rows, config, seed + 1),
    }
    all_items = torch.arange(1, len(counts), device=config.device)
    model.sid_expert.tokenizer.initialize(model.sid_expert.embeddings[all_items])
    tokenizer_optimizer = torch.optim.Adam(
        model.sid_expert.tokenizer.parameters(), lr=config.learning_rate
    )
    for epoch in range(config.tokenizer_epochs):
        permutation = torch.randperm(len(all_items), device=config.device)
        for start in range(0, len(all_items), config.batch_size):
            selected = all_items[permutation[start : start + config.batch_size]]
            tokenizer_optimizer.zero_grad(set_to_none=True)
            loss = model.sid_expert.tokenizer.reconstruction_loss(
                model.sid_expert.embeddings[selected],
                max(0.02, 0.2 * (1 - epoch / config.tokenizer_epochs)),
            )
            loss.backward()
            tokenizer_optimizer.step()
    train_queries = _training_queries(rows, config.max_length)

    def examples(_: str) -> Iterator[Example]:
        return iter(train_queries)

    training["sid"] = train_sid(model.sid_expert, examples, config, counts, True)
    model.refresh_indexes()
    return training


def run_fair_vassago(
    config: ExperimentConfig,
    data: Path,
    protocol_directory: Path,
    output: Path,
) -> Path:
    """Train without test access and export top-200 predictions for shared evaluation."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = FairProtocol.model_validate_json(
        (protocol_directory / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.protocol_hash != protocol.expected_hash():
        raise ValueError("Protocol manifest hash is invalid")
    if config.max_length != protocol.history_length or config.positive_threshold != 0.5:
        raise ValueError("VASSAGO config must use the protocol history and all ratings")
    if sha256(data / "interactions.parquet") != protocol.interactions_sha256:
        raise ValueError("VASSAGO data does not match the protocol")
    if sha256(data / "catalog.json") != protocol.catalog_sha256:
        raise ValueError("VASSAGO catalog does not match the protocol")
    sequence_path = protocol_directory / "hstu_sequences.csv"
    if sha256(sequence_path) != protocol.sequence_sha256:
        raise ValueError("Protocol sequence artifact has changed")
    seed_everything(config.seed)
    started = time.perf_counter()
    rows = _sequences(sequence_path)
    training_sequence_path = protocol_directory / "hstu_training_sequences.csv"
    if protocol.training_sequence_sha256 is None:
        training_rows = rows
    else:
        if sha256(training_sequence_path) != protocol.training_sequence_sha256:
            raise ValueError("Protocol training sequence artifact has changed")
        training_rows = _sequences(training_sequence_path)
    movies = [
        Movie.model_validate({**row, "available_at": 0, "metadata_available_at": 0})
        for row in json.loads((data / "catalog.json").read_text(encoding="utf-8"))
    ]
    documents = [movie_text(movie, 0) for movie in movies]
    encoded = TextEncoder(config.encoder, config.dimension, config.encoder_revision).cached(
        documents, output.parent / "embedding_cache"
    )
    embeddings = np.concatenate([np.zeros((1, encoded.shape[1]), dtype=np.float32), encoded])
    shadow_rows = _truncate_rows(rows, 2)
    shadow_counts = _counts(shadow_rows, len(movies))
    seed_everything(config.seed + 10_000)
    shadow = VassagoRanker(config, movies, embeddings, shadow_counts)
    training: dict[str, Any] = {
        "stacking_protocol": {
            "base_train": "all events before each user's third-last event",
            "calibration": "third-last event",
            "gate_train": "second-last event",
            "final_test": "last event",
            "final_experts": "retrained on every event before final test",
        },
        "shadow": _fit_experts(shadow, shadow_rows, shadow_counts, config, config.seed + 10_000),
    }
    calibration_queries = _event_queries(rows, -3, config.max_length, "base_validation")
    gate_queries = _event_queries(rows, -2, config.max_length, "gate_train")
    calibration_scores, calibration_targets = [], []
    for event in calibration_queries:
        result = shadow.retrieve(event.history, 2**62, event.seen)
        if event.target in result["ids"]:
            calibration_scores.append(result["scores"])
            calibration_targets.append(result["ids"].index(event.target))
    shadow.calibrator.fit(calibration_scores, calibration_targets, "base_validation")
    cached = []
    for event in gate_queries:
        result = shadow.retrieve(event.history, 2**62, event.seen)
        if event.target in result["ids"]:
            compact = {key: result[key] for key in ("ids", "scores", "features")}
            cached.append((compact, result["ids"].index(event.target)))
    gate_optimizer = torch.optim.Adam(shadow.gate.parameters(), lr=config.learning_rate)
    for _ in range(config.stacker_epochs):
        for result, target in cached:
            gate_optimizer.zero_grad(set_to_none=True)
            calibrated = shadow.calibrator.transform(result["scores"])
            features = torch.tensor(
                fusion_features(result["features"], calibrated), device=config.device
            )
            scores = torch.tensor(calibrated, dtype=torch.float32, device=config.device)
            enabled = torch.ones_like(scores, dtype=torch.bool)
            enabled[:, 0] = torch.tensor(shadow_counts[result["ids"]] > 0, device=config.device)
            enabled[:, 2] = torch.tensor(result["features"][:, 6] > 0, device=config.device)
            logits = (shadow.gate(features, enabled) * scores).sum(-1).unsqueeze(0)
            F.cross_entropy(logits, torch.tensor([target], device=config.device)).backward()
            gate_optimizer.step()

    counts = _counts(training_rows, len(movies))
    seed_everything(config.seed)
    model = VassagoRanker(config, movies, embeddings, counts)
    training["final"] = _fit_experts(model, training_rows, counts, config, config.seed)
    model.calibrator = type(model.calibrator).from_state(shadow.calibrator.state())
    model.gate.load_state_dict(shadow.gate.state_dict())
    training["stacking_examples"] = {
        "calibration": len(calibration_scores),
        "gate": len(cached),
    }
    model.eval()
    model_directory = output.parent / "model"
    model.save(model_directory)

    queries = pl.read_parquet(protocol_directory / "queries.parquet")
    methods = {
        "vassago": output,
        "vassago-id": output.with_name(f"{output.stem}-id{output.suffix}"),
        "vassago-semantic": output.with_name(f"{output.stem}-semantic{output.suffix}"),
        "vassago-weighted": output.with_name(f"{output.stem}-weighted{output.suffix}"),
    }
    existing = [path for path in methods.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    prediction_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    inference_started = time.perf_counter()
    for query in queries.iter_rows(named=True):
        result = model.retrieve(query["history"], 2**62, query["seen"])
        full_catalog = np.flatnonzero(result["allowed"])
        method_rankings = {
            "vassago": (
                result["ids"],
                model.score(result, "adaptive")[0],
            ),
            "vassago-id": (
                full_catalog,
                result["raw"]["id"][full_catalog],
            ),
            "vassago-semantic": (
                full_catalog,
                result["raw"]["semantic"][full_catalog],
            ),
            "vassago-weighted": (
                result["ids"],
                model.score(result, "weighted")[0],
            ),
        }
        limit = min(200, protocol.item_count - len(query["seen"]))
        for method, (candidate_ids, score) in method_rankings.items():
            order = np.argsort(-score, kind="stable")[:limit]
            prediction_rows[method].extend(
                {
                    "query_id": query["query_id"],
                    "model": method,
                    "seed": config.seed,
                    "protocol_hash": protocol.protocol_hash,
                    "rank": rank,
                    "movie_id": int(candidate_ids[int(index)]),
                }
                for rank, index in enumerate(order, 1)
            )
    inference_seconds = time.perf_counter() - inference_started
    prediction_hashes = {}
    for method, path in methods.items():
        pl.DataFrame(prediction_rows[method]).write_parquet(path)
        prediction_hashes[method] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    modules = (model.id_expert, model.semantic_expert, model.sid_expert, model.gate)
    manifest = {
        "schema_version": 1,
        "model": "vassago",
        "seed": config.seed,
        "protocol_hash": protocol.protocol_hash,
        "config": config.model_dump(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": {"platform": platform.platform(), "device": config.device},
        "python": sys.version,
        "elapsed_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "inference_qps": len(queries) / inference_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if config.device.startswith("cuda")
        else 0,
        "parameters": sum(
            parameter.numel() for module in modules for parameter in module.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        "artifact_bytes": sum(
            path.stat().st_size for path in model_directory.rglob("*") if path.is_file()
        ),
        "predictions": prediction_hashes,
        "training": training,
        "status": "completed",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output
