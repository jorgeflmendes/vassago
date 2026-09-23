"""Full-catalog ranking measures, paired user inference and expert complementarity."""

from itertools import combinations
from typing import Any

import numpy as np


def deterministic_rank(scores: np.ndarray, target: int, eligible: np.ndarray) -> int:
    """Rank one target by descending score and ascending item ID."""
    if scores.ndim != 1 or eligible.ndim != 1 or scores.shape != eligible.shape:
        raise ValueError("scores and eligible must be aligned one-dimensional arrays")
    if target < 0 or target >= len(scores) or not eligible[target]:
        raise ValueError("target must identify an eligible item")
    if not np.isfinite(scores[eligible]).all():
        raise ValueError("eligible scores must be finite")
    item_ids = np.arange(len(scores))
    target_score = scores[target]
    precedes = (scores > target_score) | ((scores == target_score) & (item_ids < target))
    return 1 + int(np.count_nonzero(precedes & eligible))


def ranking_metrics(predicted: list[int], relevant: set[int], k: int) -> dict[str, float]:
    if k < 1 or len(set(predicted)) != len(predicted):
        raise ValueError("k must be positive and recommendations unique")
    hits = np.array([item in relevant for item in predicted[:k]], dtype=float)
    ranks = np.arange(1, len(hits) + 1)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        f"Recall@{k}": float(hits.sum() / max(len(relevant), 1)),
        f"NDCG@{k}": float((hits / np.log2(ranks + 1)).sum() / max(ideal, 1e-12)),
        f"HitRate@{k}": float(hits.any()),
        f"MRR@{k}": float(max(hits / ranks, default=0)),
        f"MAP@{k}": float((hits * np.cumsum(hits) / ranks).sum() / max(min(k, len(relevant)), 1)),
    }


def catalog_metrics(
    lists: list[list[int]], counts: np.ndarray, genres: list[set[str]], eligible: set[int]
) -> dict[str, float]:
    flattened = [item for row in lists for item in row]
    recommended = set(flattened)
    observed = counts[1:][counts[1:] > 0]
    threshold = float(np.quantile(observed, 0.5)) if len(observed) else 0
    tail = {item for item in eligible if counts[item] <= threshold}
    # Laplace smoothing strictly over catalog items (excluding item index 0)
    item_counts = counts[1:]
    total_observations = item_counts.sum()
    catalog_size = len(item_counts)
    probability = np.zeros(len(counts), dtype=float)
    probability[1:] = (item_counts + 1.0) / (total_observations + catalog_size)
    genre_diversity = [
        1 - len(genres[a] & genres[b]) / max(len(genres[a] | genres[b]), 1)
        for row in lists
        for a, b in combinations(row, 2)
    ]
    return {
        "coverage": len(recommended & eligible) / max(len(eligible), 1),
        "long_tail_coverage": len(recommended & tail) / max(len(tail), 1),
        "average_popularity": float(np.mean(counts[flattened])) if flattened else 0,
        "novelty": float(np.mean(-np.log2(probability[flattened]))) if flattened else 0,
        "genre_diversity": float(np.mean(genre_diversity)) if genre_diversity else 0,
    }


def beyond_accuracy(
    lists: list[list[int]],
    counts: np.ndarray,
    vectors: np.ndarray,
    genres: list[set[str]],
    eligible: set[int],
) -> dict[str, float]:
    diversity: list[float] = []
    for row in lists:
        pairs = list(combinations(row, 2))
        diversity.extend(1 - float(vectors[a] @ vectors[b]) for a, b in pairs)
    return {
        **catalog_metrics(lists, counts, genres, eligible),
        "semantic_diversity": float(np.mean(diversity)) if diversity else 0,
    }


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, seed: int = 42, samples: int = 2000
) -> dict[str, float]:
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired vectors must be nonempty, aligned per user")
    delta = a - b
    rng = np.random.default_rng(seed)
    estimates = np.array(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)]
    )
    # Centered bootstrap approximates the null distribution, not a posterior probability.
    null = estimates - delta.mean()
    return {
        "mean_difference": float(delta.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_value": float((1 + (abs(null) >= abs(delta.mean())).sum()) / (samples + 1)),
    }


def paired_seed_bootstrap(
    a: dict[int, np.ndarray],
    b: dict[int, np.ndarray],
    seed: int = 42,
    samples: int = 10_000,
) -> dict[str, float]:
    """Paired crossed multiplier bootstrap over seeds and users.

    Note: With small seed counts (e.g. 5 seeds), this test primarily captures
    user-level variance conditioned on the evaluated runs rather than an asymptotic
    distribution over the full stochastic model initialization space.
    """
    if set(a) != set(b) or len(a) < 2:
        raise ValueError("Paired runs must contain the same two or more seeds")
    deltas = []
    query_count: int | None = None
    for run_seed in sorted(a):
        if a[run_seed].shape != b[run_seed].shape or a[run_seed].ndim != 1:
            raise ValueError("Paired seed vectors must be aligned per user")
        if query_count is None:
            query_count = len(a[run_seed])
        elif len(a[run_seed]) != query_count:
            raise ValueError("Paired seed vectors must share the same users")
        deltas.append(a[run_seed] - b[run_seed])
    if query_count is None or not query_count:
        raise ValueError("Paired seed vectors must be nonempty")
    matrix = np.stack(deltas)
    observed = float(matrix.mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        seed_weights = rng.exponential(1.0, size=matrix.shape[0])
        user_weights = rng.exponential(1.0, size=matrix.shape[1])
        weights = np.outer(seed_weights, user_weights)
        estimates[sample] = float((matrix * weights).sum() / weights.sum())
    null = estimates - observed
    return {
        "mean_difference": observed,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_value": float((1 + (abs(null) >= abs(observed)).sum()) / (samples + 1)),
    }


def complementarity(
    outputs: dict[str, list[list[int]]], targets: list[int], k: int = 10
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for a, b in combinations(outputs, 2):
        jaccard, both, unique_a, unique_b = [], 0, 0, 0
        for ra, rb, target in zip(outputs[a], outputs[b], targets, strict=True):
            sa, sb = set(ra[:k]), set(rb[:k])
            jaccard.append(len(sa & sb) / max(len(sa | sb), 1))
            both += target in sa and target in sb
            unique_a += target in sa and target not in sb
            unique_b += target in sb and target not in sa
        result[f"{a}__{b}"] = {
            "jaccard": float(np.mean(jaccard)) if jaccard else 0,
            "both_hit": both,
            "a_unique_hits": unique_a,
            "b_unique_hits": unique_b,
            "disagreement_rate": (unique_a + unique_b) / max(len(targets), 1),
        }
    result["oracle_hit_rate"] = sum(
        any(target in outputs[name][i][:k] for name in outputs) for i, target in enumerate(targets)
    ) / max(len(targets), 1)
    winners: dict[str, int] = {name: 0 for name in outputs}
    winners["tie"] = 0
    for index, target in enumerate(targets):
        scores = {
            name: ranking_metrics(rows[index], {target}, k)[f"NDCG@{k}"]
            for name, rows in outputs.items()
        }
        best = max(scores.values(), default=0)
        tied = [name for name, score in scores.items() if score == best]
        winners[tied[0] if len(tied) == 1 else "tie"] += 1
    result["query_winner_counts"] = winners
    return result
