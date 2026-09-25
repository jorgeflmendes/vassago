"""Train and evaluate the production VASSAGO model on MovieLens-32M under Protocol v4."""

import argparse
import csv
import gc
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryOptimizedSDPABackbone(nn.Module):
    causal_mask: torch.Tensor

    def __init__(
        self,
        n_items: int,
        dimension: int,
        max_length: int,
        heads: int,
        layers: int,
        dropout: float,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.max_length = max_length
        self.temporal_heads = heads
        self.layers_count = layers

        self.items = nn.Embedding(n_items, dimension, padding_idx=0)
        self.positions = nn.Embedding(max_length, dimension)
        self.gap_embeddings = nn.Embedding(32, dimension)
        self.log_gamma = nn.Parameter(torch.tensor([-3.0, -2.3, -1.6, -0.7]))
        self.dropout = nn.Dropout(dropout)

        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(layers)])
        self.layer_q_proj = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.layer_k_proj = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.layer_v_proj = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.layer_out_proj = nn.ModuleList(
            [nn.Linear(dimension, dimension) for _ in range(layers)]
        )
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(layers)])
        self.layer_ffn1 = nn.ModuleList([nn.Linear(dimension, ffn_dim) for _ in range(layers)])
        self.layer_ffn2 = nn.ModuleList([nn.Linear(ffn_dim, dimension) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dimension)

        pos = torch.arange(max_length)
        causal = pos[:, None].lt(pos[None, :])
        self.register_buffer("causal_mask", causal, persistent=False)

    def sequence_states(
        self,
        history: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        query_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = history.ne(0)
        L = history.shape[1]
        positions = torch.arange(L, device=history.device)
        x = self.items(history) * (self.dimension**0.5) + self.positions(positions)
        head_dim = self.dimension // self.temporal_heads

        if timestamps is not None:
            gaps = timestamps - torch.roll(timestamps, 1, dims=1)
            gaps = torch.cat([torch.zeros_like(gaps[:, :1]), gaps[:, 1:]], dim=1)
            time_buckets = torch.log2(gaps.clamp_min(0).float() / 60.0 + 1.0).long().clamp_max(31)
            x = x + self.gap_embeddings(time_buckets)

            query_times = timestamps if query_timestamps is None else query_timestamps
            elapsed = (query_times[:, :, None] - timestamps[:, None, :]).clamp_min(0).to(
                x.dtype
            ) / 60.0
            log_elapsed = torch.log1p(elapsed)

            gammas = torch.exp(self.log_gamma).to(x.dtype).view(1, self.temporal_heads, 1, 1)
            decay_bias = -gammas * log_elapsed.unsqueeze(1)

            causal_slice = self.causal_mask[:L, :L]
            invalid = causal_slice[None, None, :, :] | ~valid[:, None, None, :]
            attn_mask = decay_bias.masked_fill(invalid, -10000.0)
        else:
            causal_slice = self.causal_mask[:L, :L]
            attn_mask = torch.zeros(
                history.shape[0], self.temporal_heads, L, L, device=history.device, dtype=x.dtype
            )
            attn_mask = attn_mask.masked_fill(
                causal_slice[None, None, :, :] | ~valid[:, None, None, :], -10000.0
            )

        x = self.dropout(x) * valid.unsqueeze(-1)
        B = x.shape[0]
        for i in range(self.layers_count):
            normed = self.layer_norms1[i](x)
            q = (
                self.layer_q_proj[i](normed)
                .view(B, L, self.temporal_heads, head_dim)
                .transpose(1, 2)
            )
            k = (
                self.layer_k_proj[i](normed)
                .view(B, L, self.temporal_heads, head_dim)
                .transpose(1, 2)
            )
            v = (
                self.layer_v_proj[i](normed)
                .view(B, L, self.temporal_heads, head_dim)
                .transpose(1, 2)
            )

            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            context = context.transpose(1, 2).contiguous().view(B, L, self.dimension)
            x = x + self.layer_out_proj[i](context)
            normed2 = self.layer_norms2[i](x)
            x = x + self.layer_ffn2[i](F.relu(self.layer_ffn1[i](normed2)))
            x = x * valid.unsqueeze(-1)

        return F.normalize(self.final_norm(x), dim=-1) * valid.unsqueeze(-1)


class MemoryOptimizedVassagoRanker(nn.Module):
    def __init__(
        self,
        n_items: int,
        dim: int,
        max_length: int,
        heads: int,
        layers: int,
        dropout: float,
        ctx_dim: int,
        mem_win: int,
        temp: float,
        ctx_heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = MemoryOptimizedSDPABackbone(
            n_items, dim, max_length, heads, layers, dropout, ffn_dim
        )
        self.memory_window = mem_win
        self.temperature = temp
        self.ctx_state_proj = nn.Linear(dim, ctx_dim)
        self.ctx_item_proj = nn.Linear(dim, ctx_dim)
        self.evidence_head = nn.Linear(ctx_dim, 1)

    def _memory(
        self, states: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(self.memory_window, device=states.device)
        indices = positions[:, None] - (self.memory_window - 1 - offsets)[None, :]
        mask = indices.ge(0)
        indices = indices.clamp_min(0)
        batch = torch.arange(states.shape[0], device=states.device)[:, None]
        memory = states[batch, indices]
        mask = mask & valid[batch, indices]
        return memory, mask

    def score(
        self,
        history: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        query_timestamps: torch.Tensor | None = None,
        chunk_size: int = 8192,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        valid = history.ne(0)
        positions = valid.sum(1).clamp_min(1) - 1
        states = self.backbone.sequence_states(history, timestamps, query_timestamps)
        batch = torch.arange(states.shape[0], device=states.device)
        last_state = states[batch, positions]

        item_emb = F.normalize(self.backbone.items.weight, dim=-1)
        scores = torch.matmul(last_state, item_emb.transpose(0, 1))

        memory, mem_mask = self._memory(states, valid, positions)
        mem_proj = self.ctx_state_proj(memory)
        mem_ev = self.evidence_head(mem_proj)
        item_ctx_proj = self.ctx_item_proj(item_emb)

        N = item_emb.shape[0]
        scale = self.temperature**0.5
        mask_expanded = ~mem_mask.unsqueeze(-1)

        for c_start in range(0, N, chunk_size):
            c_end = min(c_start + chunk_size, N)
            sub_items = item_ctx_proj[c_start:c_end]
            sub_sim = torch.matmul(mem_proj, sub_items.transpose(0, 1)) / scale
            sub_sim = sub_sim.masked_fill(mask_expanded, -10000.0)
            sub_attn = torch.softmax(sub_sim, dim=1)
            sub_ev = torch.bmm(sub_attn.transpose(1, 2), mem_ev).squeeze(-1)
            scores[:, c_start:c_end].add_(sub_ev, alpha=alpha)

        return scores


def evaluate_vassago(
    model: MemoryOptimizedVassagoRanker,
    queries: list[dict[str, Any]],
    n_items: int,
    max_length: int,
    warm_items: np.ndarray,
    counts: np.ndarray,
    genres_by_item: list[set[str]],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    """Run Protocol v4 full-catalog evaluation across all queries."""
    model.eval()
    all_ranks: list[int] = []
    top10_recs: list[list[int]] = []

    eligible = set(warm_items)
    tail_cutoff = np.percentile(counts[warm_items], 80)
    tail = {i for i in warm_items if counts[i] <= tail_cutoff}
    total_inter = counts.sum()

    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            B = len(batch)
            h_t = torch.zeros(B, max_length, dtype=torch.long, device=device)
            ts_t = torch.zeros(B, max_length, dtype=torch.long, device=device)

            for i, q in enumerate(batch):
                hist = q["history"][-max_length:]
                h_ts = q["history_timestamps"][-max_length:]
                if len(hist) > 0:
                    h_t[i, : len(hist)] = torch.tensor(hist, device=device)
                    ts_t[i, : len(h_ts)] = torch.tensor(h_ts, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                scores = model.score(h_t, ts_t, ts_t, chunk_size=8192, alpha=0.5)

            scores[:, 0] = -torch.inf
            for i, q in enumerate(batch):
                if q["seen"]:
                    seen_idx = [x for x in q["seen"] if x < n_items]
                    scores[i, seen_idx] = -torch.inf

            targets_t = torch.tensor([q["target"] for q in batch], device=device)
            target_scores = scores.gather(1, targets_t.unsqueeze(1))
            ranks = ((scores > target_scores).sum(1) + 1).cpu().tolist()
            all_ranks.extend(ranks)

            top10 = torch.topk(scores, 10, dim=-1).indices.cpu().numpy()
            top10_recs.extend(top10.tolist())

    arr = np.array(all_ranks)
    rec10 = float(np.mean(arr <= 10))
    discounts10 = 1.0 / np.log2(arr + 1)
    ndcg10 = float(np.mean(np.where(arr <= 10, discounts10, 0.0)))
    mrr10 = float(np.mean(np.where(arr <= 10, 1.0 / arr, 0.0)))
    rec50 = float(np.mean(arr <= 50))
    ndcg50 = float(np.mean(np.where(arr <= 50, discounts10, 0.0)))
    rec200 = float(np.mean(arr <= 200))
    ndcg200 = float(np.mean(np.where(arr <= 200, discounts10, 0.0)))
    med_rank = float(np.median(arr))
    mean_rank = float(np.mean(arr))

    flattened = [item for row in top10_recs for item in row]
    recommended = set(flattened)
    cov10 = len(recommended & eligible) / max(len(eligible), 1)
    tail_cov10 = len(recommended & tail) / max(len(tail), 1)
    avg_pop = float(np.mean(counts[flattened]))
    probability = (counts.astype(np.float64) + 1.0) / (total_inter + n_items)
    novelty = float(np.mean(-np.log2(probability[flattened])))

    genre_divs = []
    for row in top10_recs:
        for a, b in combinations(row, 2):
            u = genres_by_item[a] | genres_by_item[b]
            if u:
                genre_divs.append(1.0 - len(genres_by_item[a] & genres_by_item[b]) / len(u))
    genre_div = float(np.mean(genre_divs)) if genre_divs else 0.0

    # Measure serving latency & VRAM
    torch.cuda.reset_peak_memory_stats()
    dummy_h = torch.randint(1, n_items, (64, max_length), device=device)
    dummy_ts = torch.arange(1000, 1000 + max_length, device=device).unsqueeze(0).expand(64, -1)

    for _ in range(15):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model.score(dummy_h, dummy_ts, dummy_ts, chunk_size=8192, alpha=0.5)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model.score(dummy_h, dummy_ts, dummy_ts, chunk_size=8192, alpha=0.5)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))
    qps = float(64.0 / (mean_lat / 1000.0))
    peak_vram = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    return {
        "Parameters": float(sum(p.numel() for p in model.parameters())),
        "NDCG@10": ndcg10,
        "Recall@10": rec10,
        "MRR@10": mrr10,
        "NDCG@50": ndcg50,
        "Recall@50": rec50,
        "NDCG@200": ndcg200,
        "Recall@200": rec200,
        "Median_Rank": med_rank,
        "Mean_Rank": mean_rank,
        "Catalog_Coverage@10": cov10,
        "Long_Tail_Coverage@10": tail_cov10,
        "Novelty@10": novelty,
        "Genre_Diversity@10": genre_div,
        "Average_Popularity@10": avg_pop,
        "P95_Latency": p95,
        "Throughput_QPS": qps,
        "Peak_VRAM_MB": peak_vram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate VASSAGO production model on MovieLens-32M"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/ml32m-global-temporal-v4")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/vassago_ml32m.pt"))
    parser.add_argument(
        "--eval-only", action="store_true", default=True, help="Run evaluation without retraining"
    )
    parser.add_argument("--train", action="store_true", help="Retrain residual context head")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--sample-queries", type=int, default=0, help="Evaluate on subset of queries (0 = all)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    with open(args.data_dir / "protocol.json", encoding="utf-8") as f:
        protocol = json.load(f)

    item_count = protocol["item_count"]
    n_items = item_count + 1
    max_length = 200
    dim = 64
    heads = 4
    layers = 2
    dropout = 0.2
    ctx_dim = 32
    mem_win = 8
    temp = 0.1
    ctx_heads = 1
    ffn_dim = 64

    # Load catalog & genres
    with open(args.data_dir / "catalog.json", encoding="utf-8") as f:
        catalog_list = json.load(f)

    genres_by_item: list[set[str]] = [set() for _ in range(n_items)]
    for m in catalog_list:
        item_id = m["movie_id"]
        g = m.get("genres", [])
        if isinstance(g, str):
            g = g.split("|")
        genres_by_item[item_id] = set(g)

    csv.field_size_limit(2**31 - 1)
    with open(args.data_dir / "hstu_training_sequences.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        train_items_list = []
        train_ts_list = []
        for r in reader:
            items = [int(x) for x in r["sequence_item_ids"].split(",")]
            ts = [int(x) for x in r["sequence_timestamps"].split(",")]
            train_items_list.append(items)
            train_ts_list.append(ts)

    n_train = len(train_items_list)
    counts = np.zeros(n_items, dtype=np.int32)
    for items in train_items_list:
        for item in items:
            if 0 < item < n_items:
                counts[item] += 1
    warm_items = np.flatnonzero(counts > 0)

    queries_df = pl.read_parquet(args.data_dir / "queries.parquet")
    if args.sample_queries > 0:
        queries_df = queries_df.slice(0, args.sample_queries)
    queries = list(queries_df.iter_rows(named=True))

    model = MemoryOptimizedVassagoRanker(
        n_items=n_items,
        dim=dim,
        max_length=max_length,
        heads=heads,
        layers=layers,
        dropout=dropout,
        ctx_dim=ctx_dim,
        mem_win=mem_win,
        temp=temp,
        ctx_heads=ctx_heads,
        ffn_dim=ffn_dim,
    ).to(device)

    if args.train:
        print("\n--- Training VASSAGO Residual Context Head with Inverse-Frequency Debiasing ---")
        total_inter = counts.sum()
        item_prob = (counts.astype(np.float32) + 1.0) / (total_inter + n_items)
        log_prob = torch.tensor(np.log(item_prob), device=device, dtype=torch.float32)

        nn.init.xavier_uniform_(model.ctx_item_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(model.ctx_state_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(model.evidence_head.weight, gain=0.1)
        nn.init.zeros_(model.ctx_item_proj.bias)
        nn.init.zeros_(model.ctx_state_proj.bias)
        nn.init.zeros_(model.evidence_head.bias)

        ctx_params = [
            model.ctx_item_proj.parameters(),
            model.ctx_state_proj.parameters(),
            model.evidence_head.parameters(),
        ]
        ctx_optimizer = torch.optim.AdamW(
            [p for params in ctx_params for p in params], lr=2e-3, weight_decay=1e-5
        )
        scaler = torch.amp.GradScaler("cuda")
        indices = np.arange(n_train)
        alpha_debias = 0.25

        t0 = time.time()
        for epoch in range(1, 6):
            model.train()
            model.backbone.eval()
            np.random.shuffle(indices)
            total_loss = 0.0
            steps = 0

            for start in range(0, n_train, 256):
                batch_idx = indices[start : start + 256]
                B = len(batch_idx)

                h_t = torch.zeros(B, max_length, dtype=torch.long, device=device)
                ts_t = torch.zeros(B, max_length, dtype=torch.long, device=device)
                targets_t = torch.zeros(B, dtype=torch.long, device=device)

                for i, idx in enumerate(batch_idx):
                    s_items = train_items_list[idx]
                    s_ts = train_ts_list[idx]
                    if len(s_items) > 1:
                        w_items = s_items[-max_length:]
                        w_ts = s_ts[-max_length:]
                        h_t[i, : len(w_items) - 1] = torch.tensor(w_items[:-1], device=device)
                        ts_t[i, : len(w_ts) - 1] = torch.tensor(w_ts[:-1], device=device)
                        targets_t[i] = w_items[-1]
                    else:
                        targets_t[i] = s_items[0]

                valid = h_t.ne(0)
                positions = valid.sum(1).clamp_min(1) - 1

                ctx_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        states = model.backbone.sequence_states(h_t, ts_t)
                        batch_range = torch.arange(B, device=device)
                        last_state = states[batch_range, positions]

                    memory, mem_mask = model._memory(states, valid, positions)
                    mem_proj = model.ctx_state_proj(memory)

                    neg_items = torch.from_numpy(np.random.choice(warm_items, size=64)).to(device)
                    eval_items = torch.cat(
                        [targets_t.unsqueeze(-1), neg_items.expand(B, -1)], dim=-1
                    )

                    cand_emb = F.normalize(model.backbone.items(eval_items), dim=-1)
                    cand_ctx = model.ctx_item_proj(cand_emb)

                    raw_sim = torch.einsum("bmd,bnd->bmn", mem_proj, cand_ctx) / (
                        model.temperature**0.5
                    )
                    raw_sim = raw_sim.masked_fill(~mem_mask.unsqueeze(-1), -10000.0)
                    attn_weights = torch.softmax(raw_sim, dim=1)

                    pooled = torch.einsum("bmn,bmd->bnd", attn_weights, mem_proj)
                    evidence = model.evidence_head(pooled).squeeze(-1)

                    base_scores = (last_state.unsqueeze(1) * cand_emb).sum(-1)
                    total_logits = base_scores + 0.5 * evidence
                    train_logits = total_logits + alpha_debias * log_prob[eval_items]

                    labels = torch.zeros(B, dtype=torch.long, device=device)
                    loss = F.cross_entropy(train_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(ctx_optimizer)
                scaler.update()

                total_loss += loss.item()
                steps += 1

            print(
                f"Epoch {epoch}/5 | Loss: {total_loss / steps:.4f} | Time: {time.time() - t0:.1f}s",
                flush=True,
            )

        del ctx_optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")
    else:
        print(f"\nLoading production checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print("Checkpoint loaded successfully.")

    print(f"\n--- Running Full-Catalog Protocol v4 Evaluation ({len(queries)} queries) ---")
    t_eval = time.time()
    results = evaluate_vassago(
        model=model,
        queries=queries,
        n_items=n_items,
        max_length=max_length,
        warm_items=warm_items,
        counts=counts,
        genres_by_item=genres_by_item,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"Evaluation finished in {time.time() - t_eval:.2f}s")

    print("\n========================================================")
    print("VASSAGO Official Evaluation Results (Protocol v4):")
    print("========================================================")
    for k, v in results.items():
        if "NDCG" in k or "Recall" in k or "MRR" in k or "Coverage" in k or "Diversity" in k:
            print(f"  {k:26s}: {v:.4f}")
        elif "Latency" in k or "VRAM" in k or "Novelty" in k or "Mean" in k or "Popularity" in k:
            print(f"  {k:26s}: {v:.2f}")
        else:
            print(f"  {k:26s}: {v}")


if __name__ == "__main__":
    main()
