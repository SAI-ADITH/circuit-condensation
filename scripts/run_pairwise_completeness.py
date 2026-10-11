#!/usr/bin/env python3
"""Verification experiment V2: EXACT pairwise-
interaction completeness. For a condensed circuit with k kept edges we test EVERY pair (i,j) by
exact masked forward: does knocking out i and j jointly differ from the sum of the two single
knockouts (non-additive interaction)? This is the O(E^2) pairwise arm of Wang et al. 2023's
completeness test, run exactly. It is the next honest check after single-edge necessity (V3/O(E))
and one rung below the exhaustive subset test (V1/O(2^E)).

Pass count is the headline: condensed k -> k*(k-1)/2 pairs (minutes); a frozen matched-accuracy
circuit (256-400e) -> ~33k-80k pairs (33x); the full model (~32k edges) -> ~5e8 pairs (marked
infeasible). We report the exact pass-count for the condensed circuit and the projected (infeasible)
counts for frozen/full, plus the real interaction map (which pairs are non-additive).

Interaction is measured on the contrastive logit margin (logit(answer) - logit(foil)):
    base   = margin with all k edges ON
    d_i    = base - margin(remove edge i)              (single-edge effect)
    d_ij   = base - margin(remove edges i and j)       (joint effect)
    inter  = d_ij - (d_i + d_j)                        (non-additivity; ~0 => independent)
-> results/verification/pairwise_<task>_<model>_<arm>_s<seed>.json
"""
import argparse, json, os, time
from itertools import combinations

import torch
from ilm.data.dataset import PairCollator
from ilm.training.setup import load_tokenizer
from ilm.discovery.pipeline import set_circuit
from scripts._task_eval import ROOT, MPATH, load_condensed, kept_edges, make_ds, logit_margin
import os

# The C1 arm is a variable, not a literal. Default = the current arm; override with
# C1_ARM=<arm> to reproduce an older one. Output filenames carry the arm, so a
# forgotten override produces a differently-named file rather than a silent overwrite.
ARM = os.environ.get("C1_ARM", "c1gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--nb", type=int, default=25, help="eval batches per masked forward")
    ap.add_argument("--inter-tol", type=float, default=0.05,
                    help="|interaction| above this margin = a non-additive (interacting) pair")
    ap.add_argument("--max-k", type=int, default=200, help="refuse above this (pair loop too big)")
    a = ap.parse_args()
    os.environ.setdefault("GRAN", "head")
    os.environ["COND_CKPT_ROOT"] = f"checkpoints/headsave/{ARM}/seed{a.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(str(ROOT / MPATH[a.model]))
    coll = PairCollator(tok.pad_token_id)
    ds = make_ds(a.task, a.model, tok, split="test")
    m = load_condensed(a.task, a.model, device)
    kept = sorted(kept_edges(m))
    k = len(kept)
    n_pairs = k * (k - 1) // 2
    print(f"{a.task}/{a.model} seed{a.seed}: k={k} kept edges, {n_pairs} pairs", flush=True)
    if k > a.max_k:
        json.dump({"task": a.task, "model": a.model, "arm": ARM, "seed": a.seed, "k": k, "n_pairs": n_pairs,
                   "feasible": False}, open(_out(a), "w"), indent=2)
        print(f"  k={k} > max_k={a.max_k}: refused."); return

    def margin_of(edges):
        set_circuit(m, edges)
        return round(logit_margin(m, a.task, ds, coll, device, nb=a.nb), 4)

    base = margin_of(kept)
    print(f"  base margin (all {k} ON) = {base}", flush=True)

    # single-edge deltas (O(k))
    d = {}
    for i in range(k):
        d[i] = round(base - margin_of([kept[j] for j in range(k) if j != i]), 4)

    # pairwise (O(k^2))
    t0 = time.time()
    inters, n_inter = [], 0
    for i, j in combinations(range(k), 2):
        d_ij = round(base - margin_of([kept[t] for t in range(k) if t not in (i, j)]), 4)
        inter = round(d_ij - (d[i] + d[j]), 4)
        interacting = abs(inter) >= a.inter_tol
        n_inter += interacting
        inters.append({"i": kept[i], "j": kept[j], "d_i": d[i], "d_j": d[j],
                       "d_ij": d_ij, "interaction": inter, "interacting": interacting})
    elapsed = round(time.time() - t0, 2)

    inters.sort(key=lambda x: -abs(x["interaction"]))
    # cost projections for the arms we CANNOT run
    # Frozen comparison edge count, MEASURED per cell from the val-selection headline (every arm
    # selected at the same bar), not a per-task constant -- the real C0 count ranges 68 to 6021
    # across cells.
    # "Best frozen" = the SMALLEST frozen arm that still clears the bar; an arm below the bar is
    # not a circuit for this behaviour, so counting it would flatter the baseline.
    frozen_k, frozen_arm = _best_frozen_k(a.task, a.model, a.seed)
    out = {"task": a.task, "model": a.model, "arm": ARM, "seed": a.seed, "k": k, "n_pairs": n_pairs,
           "wall_clock_s": elapsed, "eval_batches": a.nb, "base_margin": base,
           "feasible": True, "inter_tol": a.inter_tol,
           "n_interacting_pairs": int(n_inter),
           "frac_interacting": round(n_inter / max(n_pairs, 1), 4),
           "single_edge_deltas": {str(kept[i]): d[i] for i in range(k)},
           "top_interactions": inters[:20],
           "cost_projection": {
               "condensed_pairs": n_pairs,
               "frozen_arm": frozen_arm, "frozen_edges": frozen_k,
               "frozen_matched_pairs": frozen_k * (frozen_k - 1) // 2,
               "frozen_ratio_x": round((frozen_k * (frozen_k - 1) / 2) / max(n_pairs, 1), 1),
               "full_model_edges": 32000, "full_model_pairs": 32000 * 31999 // 2,
               "full_model_feasible": False}}
    json.dump(out, open(_out(a), "w"), indent=2)
    print(f"  EXACT PAIRWISE DONE: {n_pairs} pairs in {elapsed}s; "
          f"{n_inter} interacting ({out['frac_interacting']*100:.1f}%)")
    print(f"  frozen matched-acc would be {out['cost_projection']['frozen_matched_pairs']} pairs "
          f"({out['cost_projection']['frozen_ratio_x']}x); full model infeasible")
    print(f"PAIRWISE_COMPLETENESS_DONE -> {_out(a)}")



def _best_frozen_k(task, model, seed, default=300):
    """Smallest frozen-arm edge count that clears the val-selection bar, for this exact cell.

    Falls back to the full graph when no frozen arm passes, and to `default` only when the
    headline file is missing entirely -- so a stale checkout degrades loudly, not silently.
    """
    import json as _json
    f = ROOT / "results/valselection_headline_c1gate.json"
    if not f.exists():
        return default, "unavailable"
    rows = _json.load(open(f))["rows"]
    row = next((r for r in rows if r["task"] == task and r["model"] == model
                and r["seed"] == seed), None)
    if row is None:
        return default, "cell-missing"
    best = None
    for arm in ("C0", "EAP", "EAP-IG"):
        v = row.get(arm) or {}
        if "edges" not in v or v.get("acc_test") is None or v.get("bar_test") is None:
            continue
        if v["acc_test"] + 1e-9 < v["bar_test"]:
            continue
        if best is None or v["edges"] < best[0]:
            best = (v["edges"], arm)
    if best:
        return best[0], best[1]
    full = next((v.get("full_edges") for v in row.values()
                 if isinstance(v, dict) and v.get("full_edges")), None)
    return (full or default), "none-passed"


def _out(a):
    dd = ROOT / "results/verification"; dd.mkdir(parents=True, exist_ok=True)
    return dd / f"pairwise_{a.task}_{a.model}_{ARM}_s{a.seed}.json"


if __name__ == "__main__":
    main()
