"""Export canonical VASSAGO checkpoint to production-ready ONNX format."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_and_eval_vassago import MemoryOptimizedVassagoRanker


class VassagoServingONNX(nn.Module):
    """Production serving wrapper for VASSAGO with pre-indexed catalog projections."""

    item_emb_t: torch.Tensor
    item_ctx_t: torch.Tensor

    def __init__(self, ranker: MemoryOptimizedVassagoRanker, alpha: float = 0.5) -> None:
        super().__init__()
        self.ranker = ranker
        self.alpha = alpha

        with torch.no_grad():
            item_emb = F.normalize(ranker.backbone.items.weight, dim=-1)
            item_ctx = ranker.ctx_item_proj(item_emb)
            self.register_buffer("item_emb_t", item_emb.transpose(0, 1).contiguous())
            self.register_buffer("item_ctx_t", item_ctx.transpose(0, 1).contiguous())

    def forward(self, history: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        valid = history.ne(0)
        positions = valid.sum(1).clamp_min(1) - 1
        states = self.ranker.backbone.sequence_states(history, timestamps, timestamps)
        batch = torch.arange(states.shape[0], device=states.device)
        last_state = states[batch, positions]

        base_scores = torch.matmul(last_state, self.item_emb_t)

        memory, mem_mask = self.ranker._memory(states, valid, positions)
        mem_proj = self.ranker.ctx_state_proj(memory)
        mem_ev = self.ranker.evidence_head(mem_proj)

        scale = self.ranker.temperature**0.5
        mask_expanded = ~mem_mask.unsqueeze(-1)

        sim = torch.matmul(mem_proj, self.item_ctx_t) / scale
        sim = sim.masked_fill(mask_expanded, -10000.0)
        attn = torch.softmax(sim, dim=1)
        ev = torch.matmul(attn.transpose(1, 2), mem_ev).squeeze(-1)

        return base_scores + self.alpha * ev


def export_vassago_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    print(f"Loading checkpoint from: {checkpoint_path}")
    ranker = MemoryOptimizedVassagoRanker(
        n_items=87586,
        dim=64,
        max_length=200,
        heads=4,
        layers=2,
        dropout=0.0,
        ctx_dim=32,
        mem_win=8,
        temp=0.1,
        ctx_heads=1,
        ffn_dim=64,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ranker.load_state_dict(state_dict)
    ranker.eval()

    model = VassagoServingONNX(ranker, alpha=0.5)
    model.eval()

    dummy_history = torch.randint(1, 1000, (2, 200), dtype=torch.long)
    dummy_timestamps = torch.arange(1000, 1200, dtype=torch.long).unsqueeze(0).repeat(2, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX graph to: {output_path} (opset {opset_version})...")

    torch.onnx.export(
        model,
        (dummy_history, dummy_timestamps),
        str(output_path),
        input_names=["history", "timestamps"],
        output_names=["scores"],
        dynamic_axes={
            "history": {0: "batch_size"},
            "timestamps": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    print("Verifying ONNX structural integrity with onnx.checker...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("Structure is valid.")

    print("Verifying numerical parity against PyTorch using ONNX Runtime...")
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    test_h = np.random.randint(1, 1000, size=(4, 200), dtype=np.int64)
    test_ts = np.arange(1000, 1200, dtype=np.int64).reshape(1, 200).repeat(4, axis=0)

    raw_ort_scores = session.run(None, {"history": test_h, "timestamps": test_ts})[0]
    ort_scores = np.asarray(raw_ort_scores, dtype=np.float32)

    with torch.no_grad():
        pt_scores = (
            ranker.score(
                torch.from_numpy(test_h),
                torch.from_numpy(test_ts),
                torch.from_numpy(test_ts),
                chunk_size=8192,
                alpha=0.5,
            )
            .cpu()
            .numpy()
        )

    max_diff = float(np.max(np.abs(ort_scores - pt_scores)))
    rel_diff = float(np.max(np.abs(ort_scores - pt_scores) / (np.abs(pt_scores) + 1e-7)))
    print(f"Max absolute discrepancy: {max_diff:.6e}")
    print(f"Max relative discrepancy: {rel_diff:.6e}")

    pt_top10 = np.argsort(-pt_scores, axis=1)[:, :10]
    ort_top10 = np.argsort(-ort_scores, axis=1)[:, :10]
    ranks_match = np.array_equal(pt_top10, ort_top10)
    print(f"Top-10 ranking parity: {'MATCH' if ranks_match else 'MISMATCH'}")
    assert ranks_match, "Top-10 predictions between PyTorch and ONNX Runtime diverged."
    print("Export and verification complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VASSAGO to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/vassago_ml32m.pt"),
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vassago_ml32m.onnx"),
        help="Target ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_vassago_onnx(args.checkpoint, args.output, args.opset)
