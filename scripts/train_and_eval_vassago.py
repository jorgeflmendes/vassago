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
import yaml


def sample_batch_negatives(
    warm_items_t: torch.Tensor, targets_t: torch.Tensor, n_neg: int = 64
) -> torch.Tensor:
    """Sample independent negative items per sequence without replacement or target collision."""
    B = targets_t.shape[0]
    W = len(warm_items_t)
    rand_idx = torch.randint(0, W, (B, n_neg), device=targets_t.device)
    cands = warm_items_t[rand_idx]

    target_collision = cands == targets_t.unsqueeze(1)
    sorted_cands, _ = torch.sort(cands, dim=1)
    dups = (sorted_cands[:, 1:] == sorted_cands[:, :-1]).any(dim=1)
    needs_fix = target_collision.any(dim=1) | dups

    if needs_fix.any():
        fix_rows = torch.where(needs_fix)[0]
        for idx in fix_rows:
            target = targets_t[idx]
            row_cands = cands[idx]
            valid = row_cands[row_cands != target]
            u = torch.unique(valid)
            while len(u) < n_neg:
                more = warm_items_t[torch.randint(0, W, (n_neg * 2,), device=targets_t.device)]
                more = more[more != target]
                u = torch.unique(torch.cat([u, more]))
            cands[idx] = u[:n_neg]
    return cands


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
    candidate_mask: torch.Tensor | None = None,
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

            if candidate_mask is not None:
                scores[:, ~candidate_mask] = -torch.inf
            else:
                scores[:, 0] = -torch.inf
            for i, q in enumerate(batch):
                if q["seen"]:
                    seen_idx = [x for x in q["seen"] if x < n_items]
                    scores[i, seen_idx] = -torch.inf

            targets_t = torch.tensor([q["target"] for q in batch], device=device)
            target_scores = scores.gather(1, targets_t.unsqueeze(1))
            item_ids = torch.arange(n_items, device=device).unsqueeze(0)
            strictly_greater = (scores > target_scores).sum(1)
            ties_lower_id = ((scores == target_scores) & (item_ids < targets_t.unsqueeze(1))).sum(1)
            ranks = (strictly_greater + ties_lower_id + 1).cpu().tolist()
            all_ranks.extend(ranks)

            scores_tie = scores - item_ids * 1e-7
            top10 = torch.topk(scores_tie, 10, dim=-1).indices.cpu().numpy()
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
        "--config",
        type=Path,
        default=Path("configs/vassago_ml32m.yaml"),
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/ml32m-global-temporal-v4"),
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/vassago_ml32m.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute 2-stage training pipeline (backbone pretraining + contextual head training)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Run full Protocol v4 evaluation without training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size override (defaults to 128)",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Evaluate on subset of queries (0 = all)",
    )
    args = parser.parse_args()

    # Load configuration from YAML
    cfg: dict[str, Any] = {}
    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    serving_cfg = cfg.get("serving", {})

    dim = model_cfg.get("dimension", 64)
    heads = model_cfg.get("heads", 4)
    layers = model_cfg.get("layers", 2)
    max_length = model_cfg.get("max_length", 200)
    dropout = model_cfg.get("dropout", 0.2)
    ffn_dim = model_cfg.get("ffn_dim", 64)

    ctx_cfg = model_cfg.get("contextual_head", {})
    mem_win = ctx_cfg.get("memory_window", 8)
    ctx_dim = ctx_cfg.get("context_dimension", 32)
    temp = ctx_cfg.get("temperature", 0.1)

    train_batch_size = train_cfg.get("batch_size", 256)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    backbone_epochs = train_cfg.get("backbone_epochs", 25)
    context_epochs = train_cfg.get("context_epochs", 8)
    n_neg = train_cfg.get("negatives_per_sequence", 256)
    logit_scale = float(train_cfg.get("logit_scale", 10.0))
    debias_cfg = train_cfg.get("debiasing", {})
    alpha_debias = debias_cfg.get("alpha", 0.02)

    _ = serving_cfg.get("tiling_chunk_size", 8192)

    eval_batch_size = args.batch_size if args.batch_size is not None else 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    with open(args.data_dir / "protocol.json", encoding="utf-8") as f:
        protocol = json.load(f)

    item_count = protocol["item_count"]
    n_items = item_count + 1

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
    seq_path = args.data_dir / "hstu_training_sequences.csv"
    with open(seq_path, newline="", encoding="utf-8") as f:
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
    warm_items_t = torch.tensor(warm_items, device=device, dtype=torch.long)

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
        ffn_dim=ffn_dim,
    ).to(device)

    if args.train:
        print("\n=================================================================")
        print("VASSAGO Authentic 2-Stage Training Pipeline (Driven by YAML config)")
        print(f"  Backbone Epochs: {backbone_epochs} | Context Epochs: {context_epochs}")
        print(f"  Batch Size: {train_batch_size} | LR: {lr} | Weight Decay: {weight_decay}")
        print(f"  Debiasing Alpha: {alpha_debias} | Negatives: {n_neg} per sequence")
        print(f"  Logit Temperature Scaling: {logit_scale:.1f}")
        print("=================================================================")

        print("Pre-tensorizing training sequences for high GPU throughput...")
        t_pre = time.time()
        pre_hist = np.zeros((n_train, max_length), dtype=np.int32)
        pre_ts = np.zeros((n_train, max_length), dtype=np.int64)
        pre_targets = np.zeros(n_train, dtype=np.int32)
        for idx in range(n_train):
            s_items = train_items_list[idx]
            s_ts = train_ts_list[idx]
            if len(s_items) > 1:
                w_items = s_items[-max_length:]
                w_ts = s_ts[-max_length:]
                L = len(w_items) - 1
                pre_hist[idx, :L] = w_items[:-1]
                pre_ts[idx, :L] = w_ts[:-1]
                pre_targets[idx] = w_items[-1]
            else:
                pre_targets[idx] = s_items[0]
        print(
            f"Pre-tensorization complete in {time.time() - t_pre:.2f}s "
            f"({(pre_hist.nbytes + pre_ts.nbytes + pre_targets.nbytes) / (1024 * 1024):.1f} MB)"
        )

        total_inter = counts.sum()
        item_prob = (counts.astype(np.float32) + 1.0) / (total_inter + n_items)
        log_prob = torch.tensor(np.log(item_prob), device=device, dtype=torch.float32)
        scaler = torch.amp.GradScaler("cuda")
        indices = np.arange(n_train)

        # -------------------------------------------------------------
        # Stage 1: Causal Next-Item Backbone Pretraining (In-Batch + Hard Negatives)
        # -------------------------------------------------------------
        print("\n--- Stage 1: Training Causal Temporal Decay Transformer Backbone ---")
        backbone_optimizer = torch.optim.AdamW(
            model.backbone.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_steps_s1 = backbone_epochs * (n_train // train_batch_size + 1)
        scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            backbone_optimizer, T_max=total_steps_s1, eta_min=1e-5
        )
        t0_stage1 = time.time()

        for epoch in range(1, backbone_epochs + 1):
            model.train()
            np.random.shuffle(indices)
            total_loss = 0.0
            steps = 0

            for start in range(0, n_train, train_batch_size):
                batch_idx = indices[start : start + train_batch_size]
                B = len(batch_idx)

                h_t = torch.from_numpy(pre_hist[batch_idx]).long().to(device, non_blocking=True)
                ts_t = torch.from_numpy(pre_ts[batch_idx]).long().to(device, non_blocking=True)
                targets_t = (
                    torch.from_numpy(pre_targets[batch_idx]).long().to(device, non_blocking=True)
                )

                valid = h_t.ne(0)
                positions = valid.sum(1).clamp_min(1) - 1

                # Sample independent negatives per sequence without replacement
                cands = sample_batch_negatives(warm_items_t, targets_t, n_neg=n_neg)

                backbone_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    states = model.backbone.sequence_states(h_t, ts_t)
                    batch_range = torch.arange(B, device=device)
                    last_state = states[batch_range, positions]

                    target_embs = F.normalize(model.backbone.items(targets_t), dim=-1)
                    rand_embs = F.normalize(model.backbone.items(cands), dim=-1)

                    # 1. In-batch contrasts: dot product between each sequence state and all targets
                    inbatch_logits = (
                        torch.matmul(last_state, target_embs.transpose(0, 1)) * logit_scale
                    )
                    # Mask false negatives where other sequences in batch share identical target
                    false_neg = targets_t.unsqueeze(1) == targets_t.unsqueeze(0)
                    false_neg.fill_diagonal_(False)
                    inbatch_logits = inbatch_logits.masked_fill(false_neg, -10000.0)

                    # 2. Sampled uniform/long-tail negatives
                    rand_logits = (last_state.unsqueeze(1) * rand_embs).sum(-1) * logit_scale

                    # Frequency debiasing
                    inbatch_logits = inbatch_logits + alpha_debias * log_prob[targets_t].unsqueeze(
                        0
                    )
                    rand_logits = rand_logits + alpha_debias * log_prob[cands]

                    all_logits = torch.cat([inbatch_logits, rand_logits], dim=1)
                    labels = torch.arange(B, device=device)
                    loss = F.cross_entropy(all_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(backbone_optimizer)
                scaler.update()
                scheduler_s1.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            print(
                f"  [Stage 1] Epoch {epoch:02d}/{backbone_epochs:02d} | "
                f"Backbone Loss: {avg_loss:.4f} | Time: {time.time() - t0_stage1:.1f}s",
                flush=True,
            )

        del backbone_optimizer, scheduler_s1
        torch.cuda.empty_cache()
        gc.collect()

        # -------------------------------------------------------------
        # Stage 2: Candidate-Conditioned Contextual Head Training (with Joint Discriminative Tuning)
        # -------------------------------------------------------------
        print("\n--- Stage 2: Training Candidate-Conditioned Contextual Cross-Attention Head ---")
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
        param_groups = [
            {
                "params": [p for params in ctx_params for p in params],
                "lr": 2e-3,
                "weight_decay": 1e-5,
            },
            {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 1e-5},
        ]
        ctx_optimizer = torch.optim.AdamW(param_groups)
        total_steps_s2 = context_epochs * (n_train // train_batch_size + 1)
        scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            ctx_optimizer, T_max=total_steps_s2, eta_min=1e-5
        )
        t0_stage2 = time.time()

        for epoch in range(1, context_epochs + 1):
            model.train()
            np.random.shuffle(indices)
            total_loss = 0.0
            steps = 0

            for start in range(0, n_train, train_batch_size):
                batch_idx = indices[start : start + train_batch_size]
                B = len(batch_idx)

                h_t = torch.from_numpy(pre_hist[batch_idx]).long().to(device, non_blocking=True)
                ts_t = torch.from_numpy(pre_ts[batch_idx]).long().to(device, non_blocking=True)
                targets_t = (
                    torch.from_numpy(pre_targets[batch_idx]).long().to(device, non_blocking=True)
                )

                valid = h_t.ne(0)
                positions = valid.sum(1).clamp_min(1) - 1

                # Sample independent negatives per sequence without replacement
                cands = sample_batch_negatives(warm_items_t, targets_t, n_neg=n_neg)
                eval_items = torch.cat([targets_t.unsqueeze(1), cands], dim=1)

                ctx_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    states = model.backbone.sequence_states(h_t, ts_t)
                    batch_range = torch.arange(B, device=device)
                    last_state = states[batch_range, positions]
                    cand_emb = F.normalize(model.backbone.items(eval_items), dim=-1)

                    memory, mem_mask = model._memory(states, valid, positions)
                    mem_proj = model.ctx_state_proj(memory)
                    cand_ctx = model.ctx_item_proj(cand_emb)

                    raw_sim = torch.einsum("bmd,bnd->bmn", mem_proj, cand_ctx) / (
                        model.temperature**0.5
                    )
                    raw_sim = raw_sim.masked_fill(~mem_mask.unsqueeze(-1), -10000.0)
                    attn_weights = torch.softmax(raw_sim, dim=1)

                    pooled = torch.einsum("bmn,bmd->bnd", attn_weights, mem_proj)
                    evidence = model.evidence_head(pooled).squeeze(-1)

                    base_scores = (last_state.unsqueeze(1) * cand_emb).sum(-1) * logit_scale
                    total_logits = base_scores + 0.5 * evidence
                    train_logits = total_logits + alpha_debias * log_prob[eval_items]

                    labels = torch.zeros(B, dtype=torch.long, device=device)
                    loss = F.cross_entropy(train_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(ctx_optimizer)
                scaler.update()
                scheduler_s2.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            print(
                f"  [Stage 2] Epoch {epoch:02d}/{context_epochs:02d} | "
                f"Context Head Loss: {avg_loss:.4f} | Time: {time.time() - t0_stage2:.1f}s",
                flush=True,
            )

        del ctx_optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

        torch.save(model.state_dict(), args.checkpoint)
        print(f"\nSaved trained 2-stage checkpoint to {args.checkpoint}")
    else:
        print(f"\nLoading production checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print("Checkpoint loaded successfully.")

    candidate_mask = None
    if "candidate_item_ids" in protocol and protocol["candidate_item_ids"]:
        candidate_mask = torch.zeros(n_items, dtype=torch.bool, device=device)
        candidate_mask[protocol["candidate_item_ids"]] = True
        n_eligible = len(protocol["candidate_item_ids"])
        print(f"Applying strict candidate mask: {n_eligible} eligible items")

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
        candidate_mask=candidate_mask,
        batch_size=eval_batch_size,
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
