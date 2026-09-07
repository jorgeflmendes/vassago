"""Generate measured visual reports."""

import json
from pathlib import Path

import numpy as np
import polars as pl


def complementarity_plot(run: Path, output: Path) -> None:
    """Measured per-user winners; ties remain explicit rather than arbitrarily assigned."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pl.read_parquet(run / "per_user_metrics.parquet")
    expert_names = ["id", "semantic", "sid"]
    vectors = [
        frame.filter(pl.col("method") == name).sort("user_id")["NDCG@10"].to_numpy()
        for name in expert_names
    ]
    scores = np.stack(vectors, axis=1)
    maxima = scores.max(1, keepdims=True)
    winners = np.isclose(scores, maxima, atol=1e-12)
    unique = winners.sum(1) == 1
    counts = [(winners[:, i] & unique).sum() for i in range(3)] + [(~unique).sum()]
    manifest = json.loads((run / "run_manifest.json").read_text())
    figure, axes = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = axes.bar(
        ["ID", "Semantic", "SID", "Tied"],
        counts,
        color=["#245c9f", "#178577", "#ba7031", "#777777"],
    )
    axes.bar_label(bars, padding=3)
    axes.set_ylabel("Users")
    axes.set_title(f"{manifest['config']['dataset']}: per-user NDCG@10 winner")
    axes.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
