#!/usr/bin/env python3
"""Symmetric validation-selection comparison.

Selecting frozen baselines by matching C1's accuracy ON THE TEST SPLIT would let the
test split enter the selection of the number we then report. This builds the headline
under one rule applied to EVERY arm, C1 included:

    select the smallest circuit whose VALIDATION accuracy is within 0.05 of that
    arm's own full-graph validation accuracy, then read the sealed test split ONCE.

That is Equation 1's own constraint, so no arm is held to a bar derived from
another arm's result, and nothing is selected on test.

Inputs
  results/discovery/head_valsel/<task>/<model>.json   frozen curves w/ acc_val+acc_test
  results/headgrid/<arm>/seed{11,22,33}/<task>_<model>_knee.json    C1 endpoints (--c1-arm)

Writes results/valselection_headline.json and prints the paper-ready aggregates.
"""
import os
import json
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALSEL = ROOT / "results/discovery/head_valsel"
# Which controller arms to read. Overridden by --c1-arm/--c0-arm; kept as module
# globals because c1_endpoint() is called from several places.
ARM_C1 = os.environ.get("C1_ARM", "c1gate")
ARM_C0 = os.environ.get("C0_ARM", "c0lcb")   # c0lcb is the C0 arm the paper reports
SEEDS = (11, 22, 33)
TOL = 0.05
ARMS = ("EAP", "EAP-IG", "Random")
REPORT_ARMS = ("EAP", "EAP-IG", "Random", "C0")
# valsel uses the short task keys; the C1 grid uses the same ones.
TASKS = ("ioi", "agreement", "induction", "doc")
MODELS = ("gpt2", "llama", "llama3b", "gemma1b", "gemma4b",
          "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b")


def select_on_val(points, tol=TOL):
    """Smallest-edge point whose acc_val is within tol of the full-graph acc_val.

    points: [{'edges','acc_val','acc_test'}...]. The full graph is the largest
    'edges' entry, which is that arm's own reference rather than any other arm's.
    """
    pts = [p for p in points if p.get("acc_val") is not None]
    if not pts:
        return None
    full = max(pts, key=lambda p: p["edges"])
    bar = full["acc_val"] - tol
    ok = [p for p in pts if p["acc_val"] >= bar]
    if not ok:
        return None
    chosen = min(ok, key=lambda p: p["edges"])
    bar_test = (full.get("acc_test") - tol) if full.get("acc_test") is not None else None
    return {"edges": chosen["edges"], "acc_val": chosen["acc_val"],
            "acc_test": chosen.get("acc_test"), "bar_val": round(bar, 4),
            "full_val": full["acc_val"], "full_edges": full["edges"],
            "full_test": full.get("acc_test"),
            "bar_test": round(bar_test, 4) if bar_test is not None else None}


def c1_endpoint(task, model, seed, arm=None):
    # Default resolves at call time from ARM_C1 so that --c1-arm actually takes
    # effect; a module-level default would silently pin the old arm.
    arm = arm or ARM_C1
    f = ROOT / "results/headgrid" / arm / f"seed{seed}" / f"{task}_{model}_knee.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    edges = d.get("best_faithful_edges")
    if not edges:
        return None
    ft = d.get("final_test") or {}
    m0 = d.get("m0_baseline") or {}
    m0t = d.get("m0_baseline_test") or {}
    # Two bars, deliberately: the val bar is what SELECTED the endpoint, the test
    # bar is the same 0.05 tolerance measured against the full graph on the sealed
    # split. "Does the endpoint survive?" must use the test-side bar, or a cell
    # whose full-graph accuracy simply differs between splits is scored unfairly.
    return {"edges": edges, "acc_val": d.get("best_faithful_acc"),
            "acc_test": ft.get("circuit_acc"), "full_val": m0.get("full_cap"),
            "bar_val": round((m0.get("full_cap") or 0) - TOL, 4),
            "full_test": m0t.get("full_cap"),
            "bar_test": round((m0t.get("full_cap") or 0) - TOL, 4)}


def cluster_bootstrap(cells, n=10000, seed=0):
    """Resample CELLS (not seed-cells): the three seeds of a cell are dependent."""
    import random
    rng = random.Random(seed)
    meds = []
    keys = list(cells)
    for _ in range(n):
        draw = [cells[rng.choice(keys)] for _ in keys]
        flat = [r for grp in draw for r in grp]
        if flat:
            meds.append(st.median(flat))
    meds.sort()
    return meds[int(0.025 * len(meds))], meds[int(0.975 * len(meds))]


def sign_test_p(wins, total):
    """Two-sided exact binomial at p=0.5."""
    from math import comb
    tail = sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def main():
    import argparse
    global ARM_C1, ARM_C0
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-arm", default=ARM_C1)
    ap.add_argument("--c0-arm", default=ARM_C0)
    ap.add_argument("--out", default="results/valselection_headline.json")
    args = ap.parse_args()
    ARM_C1, ARM_C0 = args.c1_arm, args.c0_arm
    print(f"C1 arm = {ARM_C1}   C0 arm = {ARM_C0}")

    rows, missing = [], []
    for task in TASKS:
        for model in MODELS:
            f = VALSEL / task / f"{model}.json"
            if not f.exists():
                missing.append(f"{task}/{model}")
                continue
            curves = json.loads(f.read_text())["curves"]
            for seed in SEEDS:
                c1 = c1_endpoint(task, model, seed)
                if not c1:
                    missing.append(f"{task}/{model}/s{seed} (C1)")
                    continue
                rec = {"task": task, "model": model, "seed": seed, "C1": c1}
                for arm in ARMS:
                    pts = (curves.get(arm) or {}).get(str(seed))
                    rec[arm] = select_on_val(pts) if pts else None
                # C0 is the frozen-weight controller: its endpoint is already chosen
                # by the same val rule, so it needs no re-selection to be comparable.
                rec["C0"] = c1_endpoint(task, model, seed, arm=ARM_C0)
                rows.append(rec)

    print(f"{len(rows)} seed-cells; missing: {missing or 'none'}\n")

    for arm in REPORT_ARMS:
        pairs = [(r, r[arm]) for r in rows if r[arm] and r[arm]["edges"]]
        unmatched = [r for r in rows if not r[arm]]
        wins = sum(1 for r, a in pairs if r["C1"]["edges"] < a["edges"])
        ratios = [a["edges"] / r["C1"]["edges"] for r, a in pairs]
        print(f"--- C1 vs {arm} under val-selection ---")
        print(f"  comparable seed-cells: {len(pairs)}   C1 smaller in {wins}"
              f"   ({len(unmatched)} arm-cells never reach their own val bar)")
        if ratios:
            print(f"  median size ratio: {st.median(ratios):.2f}x   "
                  f"range {min(ratios):.2f}-{max(ratios):.2f}x")
        # cell-level aggregation (unit = cell, seeds pooled inside)
        cells = {}
        for r, a in pairs:
            cells.setdefault((r["task"], r["model"]), []).append(
                a["edges"] / r["C1"]["edges"])
        cellwins = sum(1 for v in cells.values() if st.median(v) > 1.0)
        print(f"  cells: C1 smaller in {cellwins}/{len(cells)}", end="")
        if cells:
            p = sign_test_p(cellwins, len(cells))
            lo, hi = cluster_bootstrap(cells)
            print(f"   sign test p={p:.2g}   bootstrap 95% CI on median "
                  f"[{lo:.1f}, {hi:.1f}]")
        else:
            print()
        # Does the val-selected point still clear the same tolerance on test,
        # measured against the full graph's OWN test accuracy?
        held = [1 for r, a in pairs if a["acc_test"] is not None
                and a["bar_test"] is not None and a["acc_test"] >= a["bar_test"] - 1e-9]
        print(f"  arm's val-selected point also clears its test-side bar: "
              f"{len(held)}/{len(pairs)}")
        c1held = [1 for r, a in pairs if r["C1"]["acc_test"] is not None
                  and r["C1"]["acc_test"] >= r["C1"]["bar_test"] - 1e-9]
        print(f"  C1 endpoint clears its test-side bar: {len(c1held)}/{len(pairs)}")
        # Strictest reading: restrict to seed-cells where BOTH arms survive on test.
        both = [(r, a) for r, a in pairs
                if a["acc_test"] is not None and a["bar_test"] is not None
                and a["acc_test"] >= a["bar_test"] - 1e-9
                and r["C1"]["acc_test"] is not None
                and r["C1"]["acc_test"] >= r["C1"]["bar_test"] - 1e-9]
        if both:
            bw = sum(1 for r, a in both if r["C1"]["edges"] < a["edges"])
            br = [a["edges"] / r["C1"]["edges"] for r, a in both]
            print(f"  both-survive subset: {len(both)} seed-cells, C1 smaller in "
                  f"{bw}, median {st.median(br):.2f}x\n")
        else:
            print()

    out = ROOT / args.out
    out.write_text(json.dumps({"protocol": "rank=train select=val report=test",
                               "tol": TOL, "c1_arm": ARM_C1, "c0_arm": ARM_C0,
                               "rows": rows}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
