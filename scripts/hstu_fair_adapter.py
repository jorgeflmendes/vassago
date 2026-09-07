"""Run Meta's pinned HSTU/SASRec code against a VASSAGO fair protocol.

This adapter imports the upstream package without copying or modifying its model
implementation. It replaces only the dataset factory and captures final rankings.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _truncate_training_sequences(
    source: Path,
    destination: Path,
    history_length: int,
) -> None:
    """Create a train-only view with the latest N causal events per sequence."""
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError("Upstream sequence artifact has no header")
        required = {"user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"}
        if required - set(reader.fieldnames):
            raise ValueError("Upstream sequence artifact is missing required columns")
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                values = {
                    key: list(ast.literal_eval(row[key]))
                    for key in ("sequence_item_ids", "sequence_ratings", "sequence_timestamps")
                }
                keep = history_length + 1  # retain the causal target for DatasetV2
                if any(len(value) != len(values["sequence_item_ids"]) for value in values.values()):
                    raise ValueError("Upstream sequence columns have inconsistent lengths")
                for key, value in values.items():
                    row[key] = repr(value[-keep:])
                writer.writerow(row)


def _install_fbgemm_fallbacks(torch: Any) -> None:
    """Use native torch kernels when the wheel lacks the active GPU architecture."""
    try:
        torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.ones(1, dtype=torch.int64, device="cuda")
        )
        return
    except Exception as error:
        if "no kernel image" not in str(error).lower():
            raise

    def complete_cumsum(lengths: Any) -> Any:
        return torch.cat(
            [torch.zeros(1, dtype=lengths.dtype, device=lengths.device), torch.cumsum(lengths, 0)]
        )

    def jagged_to_padded_dense(
        values: Any, offsets: Any, max_lengths: Any, padding_value: float = 0.0
    ) -> Any:
        offsets_tensor = offsets[0]
        batch_size = offsets_tensor.numel() - 1
        max_length = int(max_lengths[0])
        tail_shape = tuple(values.shape[1:])
        if values.shape[0] == 0:
            return torch.full(
                (batch_size, max_length, *tail_shape),
                padding_value,
                dtype=values.dtype,
                device=values.device,
            )
        lengths = offsets_tensor[1:] - offsets_tensor[:-1]
        positions = torch.arange(max_length, device=values.device).expand(batch_size, -1)
        valid = positions < lengths[:, None]
        indices = (offsets_tensor[:-1, None] + positions).clamp_max(values.shape[0] - 1)
        gathered = values[indices]
        fill = torch.full_like(gathered, padding_value)
        return torch.where(valid.reshape(*valid.shape, *([1] * len(tail_shape))), gathered, fill)

    def dense_to_jagged(dense: Any, offsets: Any) -> tuple[Any]:
        offsets_tensor = offsets[0]
        lengths = offsets_tensor[1:] - offsets_tensor[:-1]
        positions = torch.arange(dense.shape[1], device=dense.device).expand(dense.shape[0], -1)
        return (dense[(positions < lengths[:, None])],)

    torch.ops.fbgemm.asynchronous_complete_cumsum = complete_cumsum
    torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded_dense
    torch.ops.fbgemm.dense_to_jagged = dense_to_jagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=["hstu", "hstu-large", "sasrec"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=12355)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Override the upstream evaluation batch size for comparable memory measurements",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run the upstream model and evaluator in bfloat16 for serving-memory comparison",
    )
    parser.add_argument("--epochs", type=int, help="Explicit smoke/debug override")
    parser.add_argument(
        "--capture-top-k",
        type=int,
        default=2500,
        help="Number of upstream candidates captured before full seen-item filtering",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted upstream checkpoint for evaluation-only execution",
    )
    parser.add_argument(
        "--evaluation-history-length",
        type=int,
        help="Encode only the latest N items while retaining full seen-item filtering",
    )
    parser.add_argument(
        "--train-history-length",
        type=int,
        help=(
            "Train on the latest N items while keeping the shared protocol and "
            "full seen-item filtering"
        ),
    )
    parser.add_argument(
        "--model-history-length",
        type=int,
        help=(
            "Set the model input capacity to N items. Short capacities are useful for "
            "cold-adapted runs; full seen-item filtering is applied after capture."
        ),
    )
    parser.add_argument(
        "--output-model",
        help="Model label in the prediction artifact (defaults to --model)",
    )
    parser.add_argument(
        "--upstream-commit",
        default="6035c3f9b2512791b0983e0adc71749b2a22e7dc",
    )
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    config = args.config.resolve()
    protocol_directory = args.protocol.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    manifest = json.loads((protocol_directory / "protocol.json").read_text())
    if args.evaluation_history_length is not None and args.evaluation_history_length < 1:
        parser.error("--evaluation-history-length must be positive")
    if args.train_history_length is not None and args.train_history_length < 1:
        parser.error("--train-history-length must be positive")
    if args.model_history_length is not None and args.model_history_length < 1:
        parser.error("--model-history-length must be positive")
    if args.capture_top_k < 200:
        parser.error("--capture-top-k must be at least 200")
    if (
        args.train_history_length is not None
        and args.train_history_length > manifest["history_length"]
    ):
        parser.error("--train-history-length cannot exceed the shared protocol history")
    if (
        args.model_history_length is not None
        and args.model_history_length > manifest["history_length"]
    ):
        parser.error("--model-history-length cannot exceed the shared protocol history")
    model_history_length = args.model_history_length or manifest["history_length"]
    evaluation_history_length = args.evaluation_history_length or args.model_history_length
    if evaluation_history_length is not None and evaluation_history_length > model_history_length:
        parser.error("--evaluation-history-length cannot exceed the model history length")
    protocol_payload = {key: value for key, value in manifest.items() if key != "protocol_hash"}
    expected_protocol_hash = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest["protocol_hash"] != expected_protocol_hash or manifest["history_length"] != 200:
        raise ValueError("HSTU adapter requires a valid 200-item fair protocol")
    if output.exists():
        raise FileExistsError(output)
    upstream_commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream_commit != args.upstream_commit:
        raise ValueError(
            f"Upstream commit {upstream_commit} does not match pin {args.upstream_commit}"
        )

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    import fbgemm_gpu  # noqa: F401
    import gin
    import torch
    from generative_recommenders.research.data.dataset import DatasetV2
    from generative_recommenders.research.data.reco_dataset import RecoDataset
    from generative_recommenders.research.trainer import train as trainer
    _install_fbgemm_fallbacks(torch)

    sequence_file = str(protocol_directory / "hstu_sequences.csv")
    if hashlib.sha256(Path(sequence_file).read_bytes()).hexdigest() != manifest["sequence_sha256"]:
        raise ValueError("Protocol sequence artifact has changed")
    training_sequence_file = Path(sequence_file)
    temporary_training_file: Path | None = None
    if args.train_history_length is not None:
        temporary_handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"{args.model}-adapted-", delete=False
        )
        temporary_training_file = Path(temporary_handle.name)
        temporary_handle.close()
        _truncate_training_sequences(
            Path(sequence_file), temporary_training_file, args.train_history_length
        )
        training_sequence_file = temporary_training_file

    def shared_dataset(
        dataset_name: str,
        max_sequence_length: int,
        chronological: bool,
        positional_sampling_ratio: float = 1.0,
    ) -> RecoDataset:
        del dataset_name
        if not chronological or max_sequence_length != model_history_length:
            raise ValueError("Upstream configuration does not match the fair protocol")
        common = {"chronological": True}
        return RecoDataset(
            max_sequence_length=max_sequence_length,
            num_unique_items=manifest["item_count"],
            max_item_id=manifest["item_count"],
            all_item_ids=list(range(1, manifest["item_count"] + 1)),
            train_dataset=DatasetV2(
                ratings_file=str(training_sequence_file),
                **common,
                # Keep tensor width at the model capacity. When an adapted CSV is
                # used, it contains only the latest N events and earlier positions
                # are deterministic padding that never enters the loss.
                padding_length=max_sequence_length + 1,
                ignore_last_n=1,
                sample_ratio=positional_sampling_ratio,
            ),
            eval_dataset=DatasetV2(
                ratings_file=sequence_file,
                **common,
                padding_length=max_sequence_length + 1,
                ignore_last_n=0,
                sample_ratio=1.0,
            ),
        )

    query_path = protocol_directory / "queries.jsonl"
    digest = hashlib.sha256(query_path.read_bytes()).hexdigest()
    if digest != manifest["adapter_queries_sha256"]:
        raise ValueError("Adapter query artifact does not match the protocol")
    queries = [json.loads(line) for line in query_path.read_text().splitlines()]
    match_window = evaluation_history_length or model_history_length
    fingerprints = {
        (tuple(row["history"][-match_window:]), int(row["target"])): str(row["query_id"])
        for row in queries
    }
    full_seen = {str(row["query_id"]): set(row["seen"]) for row in queries}
    predictions: dict[str, list[int]] = {}
    model_statistics: dict[str, int] = {}
    inference_memory_reset = False
    original_eval = trainer.eval_metrics_v2_from_tensors
    original_model_factory = trainer.get_sequential_encoder
    original_data_loader = trainer.create_data_loader
    data_loader_calls = 0

    def capturing_model_factory(*positional: Any, **keyword: Any) -> Any:
        model = original_model_factory(*positional, **keyword)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = {
                key.removeprefix("module."): value
                for key, value in payload["model_state_dict"].items()
            }
            model.load_state_dict(state, strict=True)
        model_statistics["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        model_statistics["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        return model

    def capturing_eval(*positional: Any, **keyword: Any) -> Any:
        nonlocal inference_memory_reset
        if not inference_memory_reset:
            torch.cuda.reset_peak_memory_stats()
            inference_memory_reset = True
        eval_state = positional[0]
        seq_features = positional[2]
        target_ids = keyword.get("target_ids", positional[3] if len(positional) > 3 else None)
        if target_ids is None:
            raise RuntimeError("Upstream evaluator did not provide target IDs")
        captured: list[torch.Tensor] = []
        original_top_k = eval_state.candidate_index.get_top_k_outputs
        model = positional[1]
        original_encode = model.encode

        def encode_with_short_history(*encode_positional: Any, **encode_keyword: Any) -> Any:
            window = args.evaluation_history_length or args.train_history_length
            if window is None:
                return original_encode(*encode_positional, **encode_keyword)
            raw_ids = encode_keyword["past_ids"]
            source_shape = raw_ids.shape[:2]
            full_ids = raw_ids
            model_length = max(
                int(getattr(model, "_max_sequence_length", full_ids.shape[1])),
                full_ids.shape[1],
            )

            def pad_sequence(value: torch.Tensor) -> torch.Tensor:
                if value.ndim < 2 or value.shape[:2] != source_shape:
                    return value
                if value.shape[1] == model_length:
                    return value
                padded_shape = (*value.shape[:1], model_length, *value.shape[2:])
                padded = torch.zeros(padded_shape, dtype=value.dtype, device=value.device)
                padded[:, : value.shape[1]] = value
                return padded

            full_ids = pad_sequence(full_ids)
            full_lengths = encode_keyword["past_lengths"]
            short_ids = torch.zeros_like(full_ids)
            short_lengths = torch.minimum(full_lengths, torch.full_like(full_lengths, window))
            short_payloads = {
                name: torch.zeros_like(pad_sequence(value))
                if value.ndim >= 2 and value.shape[:2] == source_shape
                else value
                for name, value in encode_keyword["past_payloads"].items()
            }
            padded_payloads = {
                name: pad_sequence(value) for name, value in encode_keyword["past_payloads"].items()
            }
            for row, (full_length, short_length) in enumerate(
                zip(full_lengths.tolist(), short_lengths.tolist(), strict=True)
            ):
                source = slice(full_length - short_length, full_length)
                short_ids[row, :short_length] = full_ids[row, source]
                for name, value in padded_payloads.items():
                    if value.ndim >= 2 and value.shape[:2] == full_ids.shape[:2]:
                        short_payloads[name][row, :short_length] = value[row, source]
            encode_keyword.update(
                past_lengths=short_lengths,
                past_ids=short_ids,
                past_embeddings=model.get_item_embeddings(short_ids),
                past_payloads=short_payloads,
            )
            return original_encode(*encode_positional, **encode_keyword)

        def capture_top_k(*top_positional: Any, **top_keyword: Any) -> Any:
            # Keep the upstream evaluator's full candidate index and requested
            # k so its own metrics remain unchanged. We only retain a prefix of
            # the returned global ranking for the authoritative full-history
            # filter below; 400 candidates are sufficient after removing at most
            # the protocol's 200-item history.
            result = original_top_k(*top_positional, **top_keyword)
            captured.append(result[0][:, : args.capture_top_k].detach().cpu())
            return result

        eval_state.candidate_index.get_top_k_outputs = capture_top_k
        model.encode = encode_with_short_history
        try:
            result = original_eval(*positional, **keyword)
        finally:
            model.encode = original_encode
            eval_state.candidate_index.get_top_k_outputs = original_top_k
        ranked = torch.cat(captured, dim=0)
        histories = seq_features.past_ids.detach().cpu()
        targets = target_ids.detach().cpu().reshape(-1)
        if len(ranked) != len(histories):
            raise RuntimeError("Captured rankings and upstream batch are misaligned")
        for history, target, ranking in zip(histories, targets, ranked, strict=True):
            key = (tuple(int(item) for item in history if item)[-match_window:], int(target))
            query_id = fingerprints.get(key)
            if query_id is not None:
                filtered = [
                    int(item)
                    for item in ranking.tolist()
                    if int(item) not in full_seen[query_id]
                ]
                if len(filtered) < 200:
                    raise RuntimeError(
                        f"Captured ranking for {query_id} has fewer than 200 full-catalog items"
                    )
                predictions[query_id] = filtered[:200]
        return result

    def checkpoint_data_loader(*positional: Any, **keyword: Any) -> Any:
        nonlocal data_loader_calls
        data_loader_calls += 1
        if checkpoint is not None and data_loader_calls == 1:
            return None, []
        return original_data_loader(*positional, **keyword)

    trainer.get_reco_dataset = shared_dataset
    trainer.get_sequential_encoder = capturing_model_factory
    trainer.eval_metrics_v2_from_tensors = capturing_eval
    trainer.create_data_loader = checkpoint_data_loader
    gin.parse_config_file(str(config))
    configured = str(gin.query_parameter("train_fn.main_module")).lower()
    expected = "sasrec" if args.model == "sasrec" else "hstu"
    if configured != expected:
        raise ValueError(f"Config selects {configured}, but adapter requested {args.model}")
    gin.bind_parameter("train_fn.random_seed", args.seed)
    if args.model_history_length is not None:
        gin.bind_parameter("train_fn.max_sequence_length", model_history_length)
    if checkpoint is not None:
        gin.bind_parameter("train_fn.num_epochs", 1)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("Epoch override must be positive")
        gin.bind_parameter("train_fn.num_epochs", args.epochs)
    if args.model_history_length is not None and checkpoint is None:
        # Cold-adapted runs only need a validation signal at the beginning and the
        # final epoch. Keeping one partial batch between them preserves training
        # while avoiding repeated full-catalog scans.
        effective_epochs = int(gin.query_parameter("train_fn.num_epochs"))
        gin.bind_parameter("train_fn.partial_eval_num_iters", 1)
        if effective_epochs > 1:
            gin.bind_parameter("train_fn.full_eval_every_n", effective_epochs - 1)
    if args.eval_batch_size is not None:
        if args.eval_batch_size < 1:
            raise ValueError("Evaluation batch size must be positive")
        gin.bind_parameter("train_fn.eval_batch_size", args.eval_batch_size)
    if args.bf16:
        gin.bind_parameter("train_fn.main_module_bf16", True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        trainer.train_fn(0, 1, args.master_port)
    finally:
        if temporary_training_file is not None:
            temporary_training_file.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started

    missing = sorted({row["query_id"] for row in queries} - predictions.keys())
    if missing:
        raise RuntimeError(f"Final upstream evaluation missed {len(missing)} protocol queries")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_eval_batch_size = int(gin.query_parameter("train_fn.eval_batch_size"))
    except ValueError:
        # The pinned quality configs leave this optional upstream default unbound.
        resolved_eval_batch_size = args.eval_batch_size or 128
    try:
        resolved_partial_eval_iters = int(gin.query_parameter("train_fn.partial_eval_num_iters"))
    except ValueError:
        resolved_partial_eval_iters = None
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "model", "seed", "protocol_hash", "rank", "movie_id"],
        )
        writer.writeheader()
        for query_id in sorted(predictions):
            for rank, movie_id in enumerate(predictions[query_id], 1):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "model": args.output_model or args.model,
                        "seed": args.seed,
                        "protocol_hash": manifest["protocol_hash"],
                        "rank": rank,
                        "movie_id": movie_id,
                    }
                )
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": args.model,
                "output_model": args.output_model or args.model,
                "seed": args.seed,
                "protocol_hash": manifest["protocol_hash"],
                "upstream": "https://github.com/meta-recsys/generative-recommenders",
                "upstream_commit": upstream_commit,
                "config": str(config.relative_to(upstream)),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "epochs": int(gin.query_parameter("train_fn.num_epochs")),
                "execution_mode": "checkpoint-evaluation" if checkpoint else "training",
                "checkpoint_sha256": (
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    if checkpoint is not None
                    else None
                ),
                "evaluation_history_length": evaluation_history_length,
                "training_history_length": args.train_history_length,
                "model_history_length": model_history_length,
                "full_seen_post_filter": True,
                "partial_eval_num_iters": resolved_partial_eval_iters,
                "capture_top_k": args.capture_top_k,
                "eval_batch_size": resolved_eval_batch_size,
                "inference_precision": "bfloat16" if args.bf16 else "float32",
                "memory_measurement": "inference-only",
                "seen_item_filter": "full protocol history",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                **model_statistics,
                "elapsed_seconds": elapsed,
                "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "queries": len(predictions),
                "status": "completed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
