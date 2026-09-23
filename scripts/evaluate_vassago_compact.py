import sys
from pathlib import Path
sys.path.insert(0, 'src')
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from vassago.config import ExperimentConfig
from vassago.models import SASRec
from vassago.contextual_ranker import (
    ContextualEvidenceRanker,
    ContextualBackbone,
    _timestamp_tensor,
    _query_timestamp_tensor,
    _fit_base,
    _fit_context,
)
from vassago.fair_vassago import _sequences, _counts

protocol_dir = Path('data/processed/ml100k-external-v3')
rows = _sequences(protocol_dir / 'hstu_sequences.csv')
counts = _counts(rows, 1682)
queries = list(pl.read_parquet(protocol_dir / 'queries.parquet').iter_rows(named=True))
targets = {str(q['query_id']): q['target'] for q in queries}
test_timestamps = {r['user_id']: r['timestamps'][:-1][-200:] for r in rows}

class CompactBackbone(SASRec):
    def __init__(self, n_items: int, dimension: int, max_length: int, heads: int, layers: int, dropout: float, ffn_dim: int) -> None:
        super().__init__(n_items, dimension, max_length, heads, layers, dropout)
        layer = nn.TransformerEncoderLayer(
            dimension, heads, ffn_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
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
        attention_mask = torch.ones(history.shape[1], history.shape[1], device=history.device, dtype=torch.bool).triu(1)
        padding = ~valid.clone()
        padding[:, 0] = False
        key_padding = padding
        if timestamps is not None:
            gaps = timestamps - torch.roll(timestamps, 1, dims=1)
            gaps[:, 0] = 0
            x = x + self.gap_embeddings(self._time_buckets(gaps))
            query_times = timestamps if query_timestamps is None else query_timestamps
            elapsed = query_times[:, :, None] - timestamps[:, None, :]
            bias = self.temporal_bias(self._time_buckets(elapsed)).permute(0, 3, 1, 2)
            attention_mask = bias.reshape(len(history) * self.temporal_heads, history.shape[1], history.shape[1])
            allowed_keys = valid.clone()
            allowed_keys[:, 0] = True
            invalid = positions[:, None].lt(positions[None, :])[None, None] | ~allowed_keys[:, None, None, :]
            expanded_invalid = invalid.expand_as(attention_mask.view(len(history), self.temporal_heads, history.shape[1], history.shape[1])).reshape_as(attention_mask)
            attention_mask = attention_mask.masked_fill(expanded_invalid, -10_000.0)
            key_padding = torch.zeros_like(timestamps, dtype=x.dtype)
        x = self.dropout(x) * valid.unsqueeze(-1)
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        if timestamps is not None and not self.training:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            x = self.encoder(x, mask=attention_mask, src_key_padding_mask=key_padding)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
        return F.normalize(self.norm(x), dim=-1) * valid.unsqueeze(-1)

class CompactContextualRanker(ContextualEvidenceRanker):
    def __init__(self, n_items, dim, max_len, heads, layers, dropout, ctx_dim, mem_win, temp, ctx_heads, ffn_dim):
        super().__init__(n_items, dim, max_len, heads, layers, dropout, ctx_dim, mem_win, temp, ctx_heads)
        self.backbone = CompactBackbone(n_items, dim, max_len, heads, layers, dropout, ffn_dim)

# Run 5-seed evaluation
seeds = [42, 43, 44, 45, 46]
results = []

for seed in seeds:
    cfg = ExperimentConfig(
        dataset="movielens-100k-external",
        seed=seed,
        dimension=48,
        heads=2,
        layers=2,
        max_length=200,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=128,
        sampled_negatives=128,
        positive_threshold=0.5,
        device="cuda",
        mixed_precision=True,
        contextual_dimension=24,
        contextual_memory_window=32,
        contextual_positions_per_user=64,
        contextual_heads=1,
        contextual_temperature=0.1,
        epochs=80,
        contextual_epochs=15,
        contextual_joint_epochs=0,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CompactContextualRanker(
        1682, cfg.dimension, cfg.max_length, cfg.heads, cfg.layers,
        cfg.dropout, cfg.contextual_dimension, cfg.contextual_memory_window,
        cfg.contextual_temperature, cfg.contextual_heads, ffn_dim=48
    ).to("cuda")

    _fit_base(model, rows, counts, cfg, epochs=80)
    _fit_context(model, rows, counts, cfg, epochs=15)

    # Evaluate with scale = 0.5
    model.eval()
    all_ranks = []
    with torch.no_grad():
        for start in range(0, len(queries), 128):
            batch = queries[start:start+128]
            h = torch.zeros(len(batch), 200, dtype=torch.long, device="cuda")
            for r, q in enumerate(batch):
                v = q["history"][-200:]
                h[r, :len(v)] = torch.tensor(v, device="cuda")
            ts = _timestamp_tensor([str(q["user_id"]) for q in batch], test_timestamps, 200, "cuda")
            q_ts = _query_timestamp_tensor(ts, h.ne(0).sum(1), torch.tensor([int(q["timestamp"]) for q in batch], device="cuda"))
            base, ctx = model.score(h, timestamps=ts, query_timestamps=q_ts)
            scores = base + 0.5 * (ctx - base)
            for r, q in enumerate(batch):
                scores[r, 0] = -torch.inf
                if q["seen"]:
                    scores[r, q["seen"]] = -torch.inf
            targets_t = torch.tensor([q["target"] for q in batch], device="cuda")
            target_scores = scores.gather(1, targets_t[:, None])
            all_ranks.extend(((scores > target_scores).sum(1) + 1).cpu().tolist())

    arr = np.array(all_ranks)
    metrics = {
        "seed": seed,
        "NDCG@10": float(np.mean(np.where(arr <= 10, 1.0 / np.log2(arr + 1), 0.0))),
        "Recall@10": float(np.mean(arr <= 10)),
        "MRR@10": float(np.mean(np.where(arr <= 10, 1.0 / arr, 0.0))),
        "NDCG@50": float(np.mean(np.where(arr <= 50, 1.0 / np.log2(arr + 1), 0.0))),
        "Recall@50": float(np.mean(arr <= 50)),
        "NDCG@200": float(np.mean(np.where(arr <= 200, 1.0 / np.log2(arr + 1), 0.0))),
        "Recall@200": float(np.mean(arr <= 200)),
    }
    print(f"Seed {seed}: {metrics}")
    results.append(metrics)

print("\n=== Summary Across 5 Seeds ===")
for m_key in ["NDCG@10", "Recall@10", "MRR@10", "NDCG@50", "Recall@50", "NDCG@200", "Recall@200"]:
    vals = [r[m_key] for r in results]
    print(f"{m_key:12s}: Mean = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
