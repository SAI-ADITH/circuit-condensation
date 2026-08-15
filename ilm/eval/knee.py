"""The headline knee curve: accuracy (y) vs. number of ON edges (x).

Each eval checkpoint contributes one point. As the sparsity penalty ramps up the
edge count falls; a good result keeps accuracy flat until a late "knee". This
reads the eval log written by the trainer and renders the plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_knee(eval_log_path: str | Path, out_path: str | Path) -> None:
    records = load_eval_log(eval_log_path)
    if not records:
        raise ValueError(f"no eval records in {eval_log_path}")
    records = sorted(records, key=lambda r: r["num_edges"])
    edges = [r["num_edges"] for r in records]
    acc = [r["accuracy"] for r in records]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(edges, acc, "o-", color="#2a7", lw=2)
    ax.invert_xaxis()  # more concentrated (fewer edges) to the right
    ax.set_xlabel("number of ON edges  (more concentrated →)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("ILM knee curve: accuracy vs. circuit size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
