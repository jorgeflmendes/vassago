"""Executa a avaliacao completa dos 3 modelos sob o Protocolo v4 rigoroso (6765 queries).

Salva as metricas oficiais consolidadas em artifacts/benchmark_ml32m.json.
"""

import csv
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from train_and_eval_vassago import MemoryOptimizedVassagoRanker
from verify_audit_fairness import TemporalMetaHSTU, TemporalMetaSASRec


def evaluate_model_protocol_v4(
    model: nn.Module,
    model_name: str,
    is_vassago: bool,
    queries: list[dict[str, Any]],
    n_items: int,
    max_length: int,
    warm_items: np.ndarray,
    counts: np.ndarray,
    genres_by_item: list[set[str]],
    candidate_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    all_ranks: list[int] = []
    top10_recs: list[list[int]] = []

    eligible = set(warm_items)
    tail_cutoff = np.percentile(counts[warm_items], 80)
    tail = {i for i in warm_items if counts[i] <= tail_cutoff}
    total_inter = counts.sum()

    print(f"\n--- Avaliando {model_name} ({len(queries)} queries) ---")
    t0 = time.time()

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
                if is_vassago:
                    scores = cast(MemoryOptimizedVassagoRanker, model).score(
                        h_t, ts_t, ts_t, chunk_size=8192, alpha=0.5
                    )
                else:
                    scores = model.score(h_t, ts_t)

            scores[:, ~candidate_mask] = -torch.inf
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

    # Latencia e VRAM
    torch.cuda.reset_peak_memory_stats()
    dummy_h = torch.randint(1, n_items, (64, max_length), device=device)
    dummy_ts = torch.arange(1000, 1000 + max_length, device=device).unsqueeze(0).expand(64, -1)

    for _ in range(15):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_vassago:
                _ = cast(MemoryOptimizedVassagoRanker, model).score(
                    dummy_h, dummy_ts, dummy_ts, chunk_size=8192, alpha=0.5
                )
            else:
                _ = model.score(dummy_h, dummy_ts)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_vassago:
                _ = cast(MemoryOptimizedVassagoRanker, model).score(
                    dummy_h, dummy_ts, dummy_ts, chunk_size=8192, alpha=0.5
                )
            else:
                _ = model.score(dummy_h, dummy_ts)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    qps = float(64 / (np.median(latencies) / 1000.0))
    peak_vram = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    print(
        f"  NDCG@10: {ndcg10:.4f} | Recall@10: {rec10:.4f} | "
        f"MRR@10: {mrr10:.4f} | MedRank: {med_rank:.1f}"
    )
    print(
        f"  P95 Latency: {p95:.2f} ms | QPS: {qps:.0f} | "
        f"VRAM: {peak_vram:.1f} MB (Tempo: {time.time() - t0:.1f}s)"
    )

    return {
        "Parameters": int(sum(p.numel() for p in model.parameters())),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data/processed/ml32m-global-temporal-v4")
    artifacts_dir = Path("artifacts")

    with open(data_dir / "protocol.json", encoding="utf-8") as f:
        proto = json.load(f)

    n_items = proto["item_count"] + 1
    max_length = 200

    candidate_mask = torch.zeros(n_items, dtype=torch.bool, device=device)
    if "candidate_item_ids" in proto and proto["candidate_item_ids"]:
        candidate_mask[proto["candidate_item_ids"]] = True
    else:
        candidate_mask[1:] = True

    with open(data_dir / "catalog.json", encoding="utf-8") as f:
        catalog_list = json.load(f)

    genres_by_item = [set() for _ in range(n_items)]
    for m in catalog_list:
        item_id = m["movie_id"]
        g = m.get("genres", [])
        if isinstance(g, str):
            g = g.split("|")
        genres_by_item[item_id] = set(g)

    csv.field_size_limit(2**31 - 1)
    seq_path = data_dir / "hstu_training_sequences.csv"
    train_items_list = []
    with open(seq_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            train_items_list.append([int(x) for x in r["sequence_item_ids"].split(",")])

    counts = np.zeros(n_items, dtype=np.int32)
    for items in train_items_list:
        for item in items:
            if 0 < item < n_items:
                counts[item] += 1
    warm_items = np.flatnonzero(counts > 0)

    queries_df = pl.read_parquet(data_dir / "queries.parquet")
    queries = list(queries_df.iter_rows(named=True))

    results = {}

    # 1. Temporal Meta-SASRec
    sasrec = TemporalMetaSASRec(n_items, 64, max_length, 4, 2, 0.2).to(device)
    sasrec.load_state_dict(
        torch.load(
            artifacts_dir / "temporal_sasrec_ml32m.pt",
            map_location=device,
            weights_only=True,
        )
    )
    results["Temporal Meta-SASRec"] = evaluate_model_protocol_v4(
        sasrec,
        "Temporal Meta-SASRec",
        False,
        queries,
        n_items,
        max_length,
        warm_items,
        counts,
        genres_by_item,
        candidate_mask,
        device,
    )

    # 2. Temporal Meta-HSTU
    hstu = TemporalMetaHSTU(n_items, 64, max_length, 4, 2, 0.2).to(device)
    hstu.load_state_dict(
        torch.load(artifacts_dir / "temporal_hstu_ml32m.pt", map_location=device, weights_only=True)
    )
    results["Temporal Meta-HSTU"] = evaluate_model_protocol_v4(
        hstu,
        "Temporal Meta-HSTU",
        False,
        queries,
        n_items,
        max_length,
        warm_items,
        counts,
        genres_by_item,
        candidate_mask,
        device,
    )

    # 3. VASSAGO
    vassago = MemoryOptimizedVassagoRanker(n_items, 64, max_length, 4, 2, 0.2, 32, 8, 0.1, 64).to(
        device
    )
    vassago.load_state_dict(
        torch.load(artifacts_dir / "vassago_ml32m.pt", map_location=device, weights_only=True)
    )
    results["VASSAGO"] = evaluate_model_protocol_v4(
        vassago,
        "VASSAGO",
        True,
        queries,
        n_items,
        max_length,
        warm_items,
        counts,
        genres_by_item,
        candidate_mask,
        device,
    )

    output_path = artifacts_dir / "benchmark_ml32m.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados oficiais consolidados e salvos com sucesso em {output_path}!")


if __name__ == "__main__":
    main()
