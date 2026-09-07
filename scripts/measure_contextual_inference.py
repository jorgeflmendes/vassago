"""Measure full-catalog CUDA serving memory and throughput for a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import polars as pl
import torch
from safetensors.torch import load_file

from vassago.config import ExperimentConfig
from vassago.contextual_ranker import ContextualEvidenceRanker
from vassago.fair_benchmark import FairProtocol


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-chunk-size", type=int, default=16)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run the model in bfloat16 for a matched mixed-precision serving measurement",
    )
    args = parser.parse_args()
    if (
        args.history_length < 1
        or args.history_length > 200
        or args.batch_size < 1
        or args.attention_chunk_size < 1
    ):
        parser.error(
            "history-length must be in [1, 200]; "
            "batch-size and attention-chunk-size must be positive"
        )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("this measurement requires an available CUDA device")
    config = ExperimentConfig.read(args.config).model_copy(update={"device": args.device})
    protocol = FairProtocol.model_validate_json((args.protocol / "protocol.json").read_text())
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
    model.load_state_dict(load_file(args.weights, device=args.device), strict=True)
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    queries = list(pl.read_parquet(args.protocol / "queries.parquet").iter_rows(named=True))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        sequence_length = min(config.max_length, args.history_length)
        history = torch.zeros(len(batch), sequence_length, dtype=torch.long, device=args.device)
        for row, query in enumerate(batch):
            values = query["history"][-args.history_length :]
            history[row, : len(values)] = torch.tensor(values, device=args.device)
        base, contextual = model.score(history, inference_chunk_size=args.attention_chunk_size)
        if not torch.isfinite(base).all() or not torch.isfinite(contextual).all():
            raise FloatingPointError("non-finite serving scores")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "model": "contextual-evidence",
        "protocol_hash": protocol.protocol_hash,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "history_length": args.history_length,
        "batch_size": args.batch_size,
        "inference_precision": "bfloat16" if args.bf16 else "float32",
        "attention_chunk_size": args.attention_chunk_size,
        "queries": len(queries),
        "inference_seconds": elapsed,
        "inference_qps": len(queries) / elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "status": "completed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
