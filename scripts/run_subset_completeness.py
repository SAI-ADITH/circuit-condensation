#!/usr/bin/env python3
"""Verification experiment V1 (plan 2026-07-29-verification-cost-experiments): EXHAUSTIVE subset
completeness/minimality. On a small condensed circuit (k kept edges) we enumerate ALL 2^k edge
subsets, run an exact masked forward for each, and record the accuracy of every subset. This is the
un-approximated version of Wang et al. 2023's completeness test -- which they call "computationally
intractable" (2^|C|) and replace with three sampling heuristics. Condensation puts k in the 4-14
range, so the exact test is feasible; a frozen matched-accuracy circuit (256e) would need 2^256.

Outputs, per cell:
  - full_acc (all k edges ON), k, num_subsets=2^k, wall-clock
  - exact minimal faithful subset (smallest subset within TOL of full_acc) -> true minimality
  - whether any (k-1)-subset is already faithful -> is the circuit padded (honest minimality)
  - accuracy landscape: best accuracy achievable at each subset size 0..k
-> results/verification/subset_completeness_<task>_<model>_<arm>_s<seed>.json
"""
import argparse, json, os, time
from itertools import combinations

import torch
from ilm.data.dataset import PairCollator
from ilm.training.setup import load_tokenizer
from ilm.discovery.pipeline import set_circuit
from scripts._task_eval import ROOT, MPATH, load_condensed, kept_edges, make_ds, task_acc
import os

# The C1 arm is a variable, not a literal. Default = the current arm; override with
# C1_ARM=<arm> to reproduce an older one. Output filenames carry the arm, so a
# forgotten override produces a differently-named file rather than a silent overwrite.
ARM = os.environ.get("C1_ARM", "c1gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--model", default="gemma1b")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tol", type=float, default=0.05, help="faithful iff acc >= full_acc - tol")
    ap.add_argument("--nb", type=int, default=25, help="eval batches per subset")
    ap.add_argument("--max-k", type=int, default=16, help="refuse full enumeration above this")
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
    print(f"{a.task}/{a.model} seed{a.seed}: k={k} kept edges, 2^k={2**k} subsets", flush=True)
    if k > a.max_k:
        print(f"  k={k} > max_k={a.max_k}: exact enumeration refused (would be {2**k} subsets).")
        print(f"  (this is exactly the intractability the frozen circuit hits -- reporting infeasible)")
        json.dump({"task": a.task, "model": a.model, "arm": ARM, "seed": a.seed, "k": k, "num_subsets": 2**k,
                   "feasible": False}, open(_out(a), "w"), indent=2)
        return

    def acc_of(subset):
        set_circuit(m, list(subset) if subset else [])
        return round(task_acc(m, a.task, ds, coll, device, nb=a.nb), 4)

    set_circuit(m, kept)
    full_acc = round(task_acc(m, a.task, ds, coll, device, nb=a.nb), 4)
    thresh = full_acc - a.tol
    print(f"  full_acc={full_acc}  faithful threshold={thresh:.3f}", flush=True)

    t0 = time.time()
    best_at_size = {s: -1.0 for s in range(k + 1)}   # best accuracy achievable with exactly s edges
    faithful = []                                     # (size, subset) that stay faithful
    n_eval = 0
    for size in range(0, k + 1):
        for subset in combinations(range(k), size):
            edges = [kept[i] for i in subset]
            acc = acc_of(edges)
            n_eval += 1
            best_at_size[size] = max(best_at_size[size], acc)
            if acc >= thresh:
                faithful.append((size, subset))
    elapsed = round(time.time() - t0, 2)

    faithful.sort(key=lambda x: x[0])
    min_faithful_size = faithful[0][0] if faithful else None
    # padded iff some subset SMALLER than k is already faithful
    padded = min_faithful_size is not None and min_faithful_size < k
    any_k_minus_1_faithful = any(s == k - 1 for s, _ in faithful)

    out = {"task": a.task, "model": a.model, "arm": ARM, "seed": a.seed, "k": k, "num_subsets": 2 ** k,
           "subsets_evaluated": n_eval, "wall_clock_s": elapsed, "eval_batches": a.nb,
           "full_acc": full_acc, "faithful_tol": a.tol, "faithful_threshold": round(thresh, 4),
           "feasible": True,
           "exact_minimal_faithful_size": min_faithful_size,
           "circuit_is_minimal": (min_faithful_size == k),
           "padded_beyond_minimal": padded,
           "any_k_minus_1_faithful": any_k_minus_1_faithful,
           "best_acc_by_size": {str(s): round(best_at_size[s], 4) for s in range(k + 1)},
           "n_faithful_subsets": len(faithful)}
    json.dump(out, open(_out(a), "w"), indent=2)
    print(f"  EXHAUSTIVE COMPLETENESS DONE: {n_eval} subsets in {elapsed}s")
    print(f"  minimal faithful subset = {min_faithful_size}/{k} edges "
          f"({'MINIMAL' if min_faithful_size==k else 'PADDED -> smaller subset suffices'})")
    print(f"  best acc by size: {out['best_acc_by_size']}")
    print(f"SUBSET_COMPLETENESS_DONE -> {_out(a)}")


def _out(a):
    d = ROOT / "results/verification"; d.mkdir(parents=True, exist_ok=True)
    return d / f"subset_completeness_{a.task}_{a.model}_{ARM}_s{a.seed}.json"


if __name__ == "__main__":
    main()
