"""Compare a discovered head-level IOI circuit against Wang et al. 2023's
ground-truth circuit ("Interpretability in the Wild", GPT-2 small).

This is the IOI payoff: with head-level writers (``attn_{i}_h{j}``) we can read
off *which heads* a run kept and score them against the published circuit. Works
on any checkpoint saved at head granularity (``gates.pt`` + a head-level graph).

Run:
    python -m ilm.eval.wang_overlap checkpoints/ioi_head_C0/lam_0.05
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from ..masking.graph import CircuitGraph

# Hard-concrete constants (must match ilm/masking/hard_concrete.py).
GAMMA, ZETA = -0.1, 1.1

# Wang 2023 IOI circuit, by functional role -> list of (layer, head) for GPT-2
# small. 26 heads total. (Layer.head in the paper's notation.)
WANG_IOI_HEADS: dict[str, list[tuple[int, int]]] = {
    "previous_token":      [(2, 2), (4, 11)],
    "duplicate_token":     [(0, 1), (0, 10), (3, 0)],
    "induction":           [(5, 5), (5, 8), (5, 9), (6, 9)],
    "s_inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "name_mover":          [(9, 6), (9, 9), (10, 0)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "backup_name_mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6),
                            (10, 10), (11, 2), (11, 9)],
}


def wang_head_set() -> set[tuple[int, int]]:
    return {h for heads in WANG_IOI_HEADS.values() for h in heads}


def open_mask(log_alpha: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Deterministic 0/1 edge mask (matches HardConcreteGate.hard_mask)."""
    s = torch.sigmoid(log_alpha) * (ZETA - GAMMA) + GAMMA
    return (s.clamp(0.0, 1.0) > threshold)


def head_scores(gates_path: str | Path, num_layers: int = 12, n_heads: int = 12,
                threshold: float = 0.5) -> dict[tuple[int, int], int]:
    """Per-head count of OPEN outgoing edges. A head is "in the circuit" iff it
    writes to at least one kept reader (score > 0)."""
    state = torch.load(gates_path, map_location="cpu")
    log_alpha = state["log_alpha"] if isinstance(state, dict) and "log_alpha" in state else state
    graph = CircuitGraph(num_layers=num_layers, n_heads=n_heads, granularity="head")
    edges = graph.edges()
    if len(log_alpha) != len(edges):
        raise ValueError(f"gate count {len(log_alpha)} != head-graph edges {len(edges)}; "
                         "is this checkpoint head-level?")
    mask = open_mask(log_alpha, threshold)
    scores: dict[tuple[int, int], int] = {}
    for eidx, (writer, _reader) in enumerate(edges):
        if not writer.startswith("attn_") or "_h" not in writer:
            continue
        if not mask[eidx]:
            continue
        _, layer, head = writer.split("_")          # attn, "3", "h7"
        key = (int(layer), int(head[1:]))
        scores[key] = scores.get(key, 0) + 1
    return scores


def overlap(kept: set[tuple[int, int]], reference: set[tuple[int, int]]) -> dict:
    tp = len(kept & reference)
    precision = tp / max(len(kept), 1)
    recall = tp / max(len(reference), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "kept": len(kept), "reference": len(reference),
            "precision": precision, "recall": recall, "f1": f1}


def role_breakdown(kept: set[tuple[int, int]]) -> str:
    lines = []
    for role, heads in WANG_IOI_HEADS.items():
        found = sorted(h for h in heads if h in kept)
        lines.append(f"  {role:20s} {len(found)}/{len(heads)}  {found}")
    return "\n".join(lines)


def analyse(run_dir: str | Path, num_layers: int = 12, n_heads: int = 12,
            threshold: float = 0.5) -> dict:
    run_dir = Path(run_dir)
    scores = head_scores(run_dir / "gates.pt", num_layers, n_heads, threshold)
    kept = set(scores)
    ref = wang_head_set()
    stats = overlap(kept, ref)
    print(f"\n=== {run_dir} ===")
    print(f"kept heads: {len(kept)}  | Wang heads: {len(ref)}  | matched: {stats['tp']}")
    print(f"precision={stats['precision']:.2f}  recall={stats['recall']:.2f}  f1={stats['f1']:.2f}")
    print("Wang role coverage (matched/total):")
    print(role_breakdown(kept))
    extra = sorted(kept - ref, key=lambda h: -scores[h])[:10]
    print(f"top kept heads NOT in Wang: {extra}")
    return {**stats, "kept_heads": sorted(kept)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="+", help="checkpoint dir(s) with gates.pt")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    for d in args.run_dir:
        analyse(d, args.layers, args.heads, args.threshold)


if __name__ == "__main__":
    main()
