"""End-to-end local experiments. Test targets never select parameters."""

import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from vassago.config import ExperimentConfig
from vassago.data import (
    Example,
    Movie,
    iter_examples,
    partition_data,
    prepare_parquet,
    synthetic,
    train_statistics,
)
from vassago.evaluation import (
    beyond_accuracy,
    complementarity,
    paired_bootstrap,
    ranking_metrics,
)
from vassago.features import TextEncoder, movie_text
from vassago.models import BPR, LightGCN
from vassago.retrieval import fusion_features, mmr
from vassago.serving import VassagoRanker, history_tensor
from vassago.training import batches, seed_everything, train_sequence, train_sid


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    config: ExperimentConfig,
    output: Path,
    data: Path | None = None,
    evaluation_partition: str = "test",
) -> Path:
    if evaluation_partition not in {"validation", "test"}:
        raise ValueError("Experiment reporting is restricted to validation or test")
    seed_everything(config.seed)
    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:12]
    destination = output / f"{config.dataset}__ablation__seed-{config.seed}__{config_hash}"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    path = destination / "interactions.parquet"
    if data is None:
        if config.dataset != "synthetic":
            raise ValueError("Real-data experiments require --data with a prepared catalog")
        movies, frame = synthetic(config.seed)
        frame, boundaries = partition_data(frame, config.split_quantiles, config.cutoffs)
        frame.write_parquet(path)
        input_files = {}
    else:
        movies = [
            Movie.model_validate(m)
            for m in json.loads((data / "catalog.json").read_text(encoding="utf-8"))
        ]
        boundaries = prepare_parquet(
            data / "interactions.parquet", path, config.split_quantiles, config.cutoffs
        )
        input_files = {
            p.name: sha256(p) for p in (data / "catalog.json", data / "interactions.parquet")
        }
    rng = np.random.default_rng(config.seed)
    cold = set(
        map(
            int,
            rng.choice(
                np.arange(1, len(movies) + 1),
                int(len(movies) * config.cold_fraction),
                replace=False,
            ),
        )
    )
    counts, users = train_statistics(path, len(movies), config.positive_threshold, cold)

    def examples(partition: str) -> Iterator[Example]:
        stream = iter_examples(
            path, movies, config.positive_threshold, config.max_length, cold, partition
        )
        return (
            iter(islice(stream, config.evaluation_limit))
            if (partition != "train" and config.evaluation_limit)
            else stream
        )

    documents = [movie_text(m, boundaries[0]) for m in movies]
    encoded = TextEncoder(config.encoder, config.dimension, config.encoder_revision).cached(
        documents, destination / "embedding_cache"
    )
    embeddings = np.concatenate([np.zeros((1, encoded.shape[1]), dtype=np.float32), encoded])
    model = VassagoRanker(config, movies, embeddings, counts)
    training_log: dict[str, Any] = {}
    print("Training independent ID and semantic sequence experts", flush=True)
    training_log["id"] = train_sequence(model.id_expert, examples, movies, config, counts)
    training_log["semantic"] = train_sequence(model.semantic_expert, examples, movies, config)
    tokenizer_optimizer = torch.optim.Adam(
        model.sid_expert.tokenizer.parameters(), lr=config.learning_rate
    )
    warm_ids = np.flatnonzero(counts)
    print("Training tokenizer and frozen/differentiable SID experts", flush=True)
    model.sid_expert.tokenizer.initialize(model.sid_expert.embeddings[warm_ids])
    for epoch in range(config.tokenizer_epochs):
        model.sid_expert.tokenizer.train()
        for start in range(0, len(warm_ids), config.batch_size):
            tokenizer_optimizer.zero_grad(set_to_none=True)
            token_items = torch.tensor(
                warm_ids[start : start + config.batch_size], device=config.device
            )
            loss = model.sid_expert.tokenizer.reconstruction_loss(
                model.sid_expert.embeddings[token_items],
                max(0.05, 1 - epoch / config.tokenizer_epochs),
            )
            loss.backward()
            tokenizer_optimizer.step()
    frozen = copy.deepcopy(model.sid_expert)
    training_log["sid_frozen"] = train_sid(frozen, examples, config, counts, False)
    training_log["sid"] = train_sid(model.sid_expert, examples, config, counts, True)
    differentiable = copy.deepcopy(model.sid_expert)
    differentiable_diagnostics = differentiable.tokenizer.diagnostics(differentiable.embeddings[1:])
    selected_sid = "differentiable"
    training_log["sid_selection"] = {
        "criterion": "registered_primary_fusion_variant",
        "selected": selected_sid,
        "frozen_minimum_base_validation_cross_entropy": min(training_log["sid_frozen"]),
        "differentiable_minimum_base_validation_cross_entropy": min(training_log["sid"]),
    }

    user_map = {user: i for i, user in enumerate(sorted(users), 1)}
    # Graph edges are aggregated in Polars; no 32M-row Python object list.
    edges_frame = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("partition") == "train")
            & (pl.col("rating") >= config.positive_threshold)
            & ~pl.col("movie_id").is_in(sorted(cold))
        )
        .select("user_id", "movie_id")
        .unique()
        .sort("user_id", "movie_id")
        .collect()
    )
    edge_users = edges_frame["user_id"].replace_strict(user_map, return_dtype=pl.Int64).to_numpy()
    edges = torch.tensor(
        np.stack([edge_users, edges_frame["movie_id"].to_numpy()]), dtype=torch.long
    )
    baselines = {
        "bpr": BPR(len(users), len(movies), config.dimension).to(config.device),
        "lightgcn": LightGCN(len(users), len(movies), config.dimension, edges).to(config.device),
    }
    print("Training BPR and LightGCN baselines", flush=True)
    positive_items = edges_frame["movie_id"].to_numpy()
    offsets = np.searchsorted(edge_users, np.arange(len(users) + 2))
    for name, baseline in baselines.items():
        optimizer = torch.optim.AdamW(
            baseline.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        best_state, best_metric = copy.deepcopy(baseline.state_dict()), -1.0
        for _ in range(config.epochs):
            baseline.train()
            graph_representations = baseline.representations() if name == "lightgcn" else None
            leaves = (
                tuple(v.detach().requires_grad_(True) for v in graph_representations)
                if graph_representations is not None
                else None
            )
            graph_pairs = 0
            if leaves is not None:
                optimizer.zero_grad(set_to_none=True)
            for batch in batches(examples("train"), config.batch_size):
                valid, negatives = [], []
                for event in batch:
                    user = user_map[event.user_id]
                    positive = positive_items[offsets[user] : offsets[user + 1]]
                    negative = None
                    for candidate in rng.choice(warm_ids, size=32):
                        position = np.searchsorted(positive, candidate)
                        if (position == len(positive) or positive[position] != candidate) and (
                            movies[candidate - 1].available_at <= event.timestamp
                        ):
                            negative = int(candidate)
                            break
                    if negative is not None:
                        valid.append(event)
                        negatives.append(negative)
                if not valid:
                    continue
                user_tensor = torch.tensor(
                    [user_map[e.user_id] for e in valid], device=config.device
                )
                positive_tensor = torch.tensor([e.target for e in valid], device=config.device)
                negative_tensor = torch.tensor(negatives, device=config.device)
                if leaves is None:
                    optimizer.zero_grad(set_to_none=True)
                    baseline.loss(user_tensor, positive_tensor, negative_tensor).backward()
                    optimizer.step()
                else:
                    u, item = leaves
                    loss = -F.logsigmoid(
                        (u[user_tensor] * (item[positive_tensor] - item[negative_tensor])).sum(-1)
                    ).sum()
                    loss.backward()
                    graph_pairs += len(valid)
            if leaves is not None and graph_representations is not None and graph_pairs:
                # Accumulate embedding gradients by chunk, then differentiate graph propagation
                # once per epoch: exact full-batch BPR objective, bounded loss-graph memory.
                gradients = []
                for leaf in leaves:
                    if leaf.grad is None:
                        raise RuntimeError("Missing accumulated graph-embedding gradient")
                    gradients.append(leaf.grad / graph_pairs)
                torch.autograd.backward(graph_representations, gradients)
                optimizer.step()
            baseline.eval()
            values = []
            with torch.no_grad():
                validation_users, validation_items = baseline.representations()
                for event in examples("base_validation"):
                    scores = (
                        (validation_users[user_map.get(event.user_id, 0)] @ validation_items.T)
                        .cpu()
                        .numpy()
                    )
                    allowed = model.eligible(event.seen, event.timestamp) & (counts > 0)
                    ids = np.flatnonzero(allowed)
                    ranked = ids[np.argsort(-scores[ids], kind="stable")][:10].tolist()
                    values.append(ranking_metrics(ranked, {event.target}, 10)["NDCG@10"])
            metric = float(np.mean(values)) if values else 0
            if metric > best_metric:
                best_metric, best_state = metric, copy.deepcopy(baseline.state_dict())
        baseline.load_state_dict(best_state)
        training_log[name] = {"base_validation_ndcg10": best_metric}

    print("Fitting calibration and temporal gate on separate windows", flush=True)
    calibration_scores, calibration_targets = [], []
    for event in islice(examples("base_validation"), 2000):
        result = model.retrieve(event.history, event.timestamp, event.seen)
        if event.target in result["ids"]:
            calibration_scores.append(result["scores"])
            calibration_targets.append(result["ids"].index(event.target))
    model.calibrator.fit(calibration_scores, calibration_targets, "base_validation")
    gate_events: list[tuple[dict[str, Any], int]] = []
    for event in examples("gate_train"):
        result = model.retrieve(event.history, event.timestamp, event.seen)
        if event.target in result["ids"]:
            compact = {key: result[key] for key in ("ids", "scores", "features")}
            gate_events.append((compact, result["ids"].index(event.target)))
    validation_events: list[tuple[dict[str, Any], int]] = []
    for event in examples("validation"):
        result = model.retrieve(event.history, event.timestamp, event.seen)
        compact = {key: result[key] for key in ("ids", "scores", "features")}
        validation_events.append((compact, event.target))
    gate_optimizer = torch.optim.Adam(model.gate.parameters(), lr=config.learning_rate)
    best_gate = copy.deepcopy(model.gate.state_dict())
    best_gate_metric = -1.0
    for _ in range(config.epochs):
        for result, target_index in gate_events:
            model.gate.train()
            gate_optimizer.zero_grad(set_to_none=True)
            calibrated = model.calibrator.transform(result["scores"])
            features = torch.tensor(
                fusion_features(result["features"], calibrated), device=config.device
            )
            gate_scores = torch.tensor(calibrated, dtype=torch.float32, device=config.device)
            enabled = torch.ones_like(gate_scores, dtype=torch.bool)
            enabled[:, 0] = torch.tensor(counts[result["ids"]] > 0, device=config.device)
            enabled[:, 2] = torch.tensor(result["features"][:, 6] > 0, device=config.device)
            weights = model.gate(features, enabled)
            logits = (weights * gate_scores).sum(-1).unsqueeze(0)
            F.cross_entropy(logits, torch.tensor([target_index], device=config.device)).backward()
            gate_optimizer.step()
        validation_values = []
        for result, target in validation_events:
            score, _ = model.score(result, "adaptive")
            ranked = [result["ids"][i] for i in np.argsort(-score, kind="stable")[:10]]
            validation_values.append(ranking_metrics(ranked, {target}, 10)["NDCG@10"])
        gate_metric = float(np.mean(validation_values)) if validation_values else 0
        if gate_metric > best_gate_metric:
            best_gate_metric, best_gate = gate_metric, copy.deepcopy(model.gate.state_dict())
    model.gate.load_state_dict(best_gate)
    training_log["gate"] = {
        "selection_partition": "validation",
        "best_ndcg10": best_gate_metric,
    }
    resolution = 20
    grids = [
        np.array([a, b, resolution - a - b]) / resolution
        for a in range(resolution + 1)
        for b in range(resolution + 1 - a)
    ]
    pair_grids = [np.array([a, resolution - a, 0]) / resolution for a in range(resolution + 1)]
    objectives = np.zeros(len(grids) + len(pair_grids))
    for result, target in validation_events:
        if not result["ids"]:
            continue
        for index, weights in enumerate(grids + pair_grids):
            model.weights = weights
            scores, _ = model.score(result, "weighted")
            ranked = [result["ids"][i] for i in np.argsort(-scores, kind="stable")[:10]]
            objectives[index] += ranking_metrics(ranked, {target}, 10)["NDCG@10"]
    model.weights = grids[int(objectives[: len(grids)].argmax())]
    model.pair_weights = pair_grids[int(objectives[len(grids) :].argmax())]
    model.eval()
    model.save(destination / "model")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in frozen.state_dict().items()},
        destination / "model" / "sid_frozen.safetensors",
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in differentiable.state_dict().items()},
        destination / "model" / "sid_differentiable.safetensors",
    )
    write_json(destination / "model" / "users.json", user_map)
    for name, baseline in baselines.items():
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in baseline.state_dict().items()
            if k != "adjacency"
        }
        if name == "lightgcn":
            state["edges"] = edges.cpu()
        save_file(state, destination / "model" / f"{name}.safetensors")
    print("Evaluating full-catalog experts, fusion, cohorts and complementarity", flush=True)
    outputs: dict[str, list[list[int]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    targets, latencies = [], []
    eligible_catalog: set[int] = set()
    frozen_codes = frozen.codes()
    with torch.no_grad():
        baseline_representations = {
            name: baseline.representations() for name, baseline in baselines.items()
        }
    serving_latencies: list[float] = []
    popularity_values = counts[counts > 0]
    low, high = np.quantile(popularity_values, [0.5, 0.9]) if len(popularity_values) else (0, 0)
    for event in examples(evaluation_partition):
        clock = time.perf_counter()
        result = model.retrieve(event.history, event.timestamp, event.seen)
        model.score(result, "adaptive")
        serving_latencies.append(time.perf_counter() - clock)
        eligible_catalog.update(map(int, np.flatnonzero(result["allowed"])))
        rankings = {}
        for name in ("id", "semantic", "popularity"):
            mask = result["allowed"] & (counts > 0) if name == "id" else result["allowed"]
            ids = np.flatnonzero(mask)
            rankings[name] = ids[np.argsort(-result["raw"][name][ids], kind="stable")][:20].tolist()
        rankings["sid"] = [i for i, _ in result["outputs"]["sid"]][:20]
        with torch.no_grad():
            tensor = history_tensor([event.history], config.max_length, config.device)
            generated = frozen.decoder.generate(
                frozen.context(tensor)[0], frozen_codes, config.beam_width, len(movies)
            )
            rankings["sid_frozen"] = [
                int(r["movie_id"]) for r in generated if result["allowed"][int(r["movie_id"])]
            ][:20]
            for name in baselines:
                baseline_users, baseline_items = baseline_representations[name]
                raw_scores = (
                    (baseline_users[user_map.get(event.user_id, 0)] @ baseline_items.T)
                    .cpu()
                    .numpy()
                )
                ids = np.flatnonzero(result["allowed"] & (counts > 0))
                rankings[name] = ids[np.argsort(-raw_scores[ids], kind="stable")][:20].tolist()
        for strategy in ("pair", "weighted", "adaptive", "refined", "diverse"):
            score, _ = model.score(result, strategy)
            rankings[strategy] = (
                mmr(result["ids"], score, embeddings, 20, config.mmr_lambda)
                if strategy == "diverse"
                else [result["ids"][i] for i in np.argsort(-score, kind="stable")[:20]]
            )
        latencies.append(time.perf_counter() - clock)
        targets.append(event.target)
        for name, ranking in rankings.items():
            outputs[name].append(ranking)
            query_metrics = {
                key: value
                for k in (5, 10, 20)
                for key, value in ranking_metrics(ranking, {event.target}, k).items()
            }
            rows.append(
                {
                    "user_id": event.user_id,
                    "timestamp": event.timestamp,
                    "target": event.target,
                    "method": name,
                    "cold_item": counts[event.target] == 0,
                    "heldout_item": event.target in cold,
                    "popularity_cohort": "head"
                    if counts[event.target] >= high
                    else ("long_tail" if counts[event.target] <= low else "mid_tail"),
                    "history_cohort": "cold"
                    if users.get(event.user_id, 0) == 0
                    else ("few_shot" if users[event.user_id] <= 5 else "warm"),
                    **query_metrics,
                }
            )
    if not rows:
        raise ValueError("Final test window has no eligible targets")
    frame = pl.DataFrame(rows)
    frame.write_parquet(destination / "per_query_metrics.parquet")
    metric_names = [c for c in frame.columns if "@" in c]
    per_user = frame.group_by("method", "user_id").agg([pl.col(c).mean() for c in metric_names])
    per_user.write_parquet(destination / "per_user_metrics.parquet")
    metrics = (
        per_user.group_by("method").agg([pl.col(c).mean() for c in metric_names]).sort("method")
    )
    write_json(destination / "metrics.json", metrics.to_dicts())
    for cohort in ("cold_item", "heldout_item", "history_cohort", "popularity_cohort"):
        cohort_frame = (
            frame.group_by("method", cohort, "user_id")
            .agg([pl.col(c).mean() for c in metric_names])
            .group_by("method", cohort)
            .agg([pl.col(c).mean() for c in metric_names] + [pl.len().alias("users")])
        )
        write_json(destination / f"{cohort}_metrics.json", cohort_frame.to_dicts())
    write_json(
        destination / "complementarity.json",
        complementarity({k: outputs[k] for k in ("id", "semantic", "sid")}, targets),
    )
    write_json(
        destination / "beyond_accuracy.json",
        {
            name: beyond_accuracy(
                lists,
                counts,
                embeddings,
                [set()] + [set(m.genres) for m in movies],
                eligible_catalog,
            )
            for name, lists in outputs.items()
        },
    )
    a = per_user.filter(pl.col("method") == "adaptive").sort("user_id")["NDCG@10"].to_numpy()
    b = per_user.filter(pl.col("method") == "id").sort("user_id")["NDCG@10"].to_numpy()
    write_json(destination / "significance.json", paired_bootstrap(a, b, config.seed))
    write_json(
        destination / "tokenizer.json",
        {
            "frozen": frozen.tokenizer.diagnostics(frozen.embeddings[1:]),
            "differentiable": differentiable_diagnostics,
            "selected": selected_sid,
        },
    )
    write_json(destination / "training.json", training_log)

    def git(args: list[str]) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    manifest = {
        "schema_version": 1,
        "evaluation_partition": evaluation_partition,
        "config": config.model_dump(),
        "seed": config.seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": config.device,
        },
        "input_sha256": input_files,
        "dataset_manifest": {
            "name": config.dataset,
            "synthetic": data is None,
            "source_snapshot": "generated"
            if data is None
            else "user-supplied; see input manifests",
            "prepared_sha256": sha256(path),
            "boundaries": boundaries,
            "heldout_items": sorted(cold),
            "calibration_max_queries": 2000,
        },
        "parameter_counts": {
            name: sum(p.numel() for p in getattr(model, name).parameters())
            for name in ("id_expert", "semantic_expert", "sid_expert", "gate")
        },
        "baseline_parameter_counts": {
            name: sum(p.numel() for p in baseline.parameters())
            for name, baseline in baselines.items()
        },
        "inference_latency_p50_seconds": float(np.quantile(serving_latencies, 0.5)),
        "inference_latency_p95_seconds": float(np.quantile(serving_latencies, 0.95)),
        "inference_qps": len(serving_latencies) / sum(serving_latencies),
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_pipeline_latency_p50_seconds": float(np.quantile(latencies, 0.5)),
        "evaluation_pipeline_latency_p95_seconds": float(np.quantile(latencies, 0.95)),
        "evaluation_pipeline_qps": len(latencies) / sum(latencies),
        "artifact_bytes": sum(
            p.stat().st_size for p in (destination / "model").rglob("*") if p.is_file()
        ),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if config.device.startswith("cuda")
        else 0,
        "status": "completed",
        "claim_scope": "exploratory; not publication reproduction",
    }
    write_json(destination / "run_manifest.json", manifest)
    return destination
