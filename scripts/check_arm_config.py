#!/usr/bin/env python3
"""Assert that a finished (or in-progress) arm actually ran with the settings we intended.

Written after the 2026-08-12 capability-gate incident: the `c1lcb` arm ran with
`CAP_GATE_AUTO=1`, which silently replaced the documented 1.05 perplexity budget with a
self-calibrated ~1.011. The launcher said 1.05, the paper said 1.05, and the only place the
truth existed was a runtime variable that overwrote its own input. 52 of 96 runs then stopped
on general-capability drift rather than faithfulness, and every loss to the baselines was one
of those runs.

The lesson is not "be careful" -- it is that a setting which changes the returned circuit must
be readable off the artifact and checked mechanically. This script does the checking.

    # what we expect the C1 re-run to be
    python scripts/check_arm_config.py --arm c1gate \
        --expect cfg_cap_gate_effective=1.05 --expect cfg_accept_lcb=1.645 \
        --expect cfg_adaptive_min=1 --expect cfg_cap_gate_auto=False \
        --expect-contains cfg_eval_data=eval_big.jsonl

    # or: diff every recorded setting against a reference arm
    python scripts/check_arm_config.py --arm c1gate --reference c1lcb

Exit code is nonzero if any cell violates an expectation, so it can gate a submitter.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Fields that describe the run's configuration rather than its results. Anything here that
# differs between two cells of the same arm, or from the declared expectation, is a finding.
CONFIG_KEYS = [
    "mode", "start", "rerank", "heal_steps_per_round", "heal_lr", "heal_alpha", "heal_delta",
    "heal_cap_tol", "heal_acc_floor", "heal_delta_max", "heal_dual_lr", "heal_circ_floor",
    "cap_gate", "cap_abort", "retry_floor", "retry_max",
    "cfg_accept_lcb", "cfg_adaptive_min", "cfg_adaptive", "cfg_freeze_weights",
    "cfg_cap_gate_auto", "cfg_cap_gate_requested", "cfg_cap_gate_effective",
    "cfg_eval_data", "cfg_eval_batches", "cfg_test_data", "cfg_adaptive_tol", "cfg_heal_mode",
]


def load(arm):
    out = {}
    for f in sorted((ROOT / "results/headgrid" / arm).glob("seed*/*_knee.json")):
        try:
            out[f"{f.parent.name}/{f.stem}"] = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable (still being written?): {f}")
    return out


def coerce(v):
    """Compare '1.05' and 1.05, 'False' and False, as equal."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reference", help="second arm to diff config against")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--expect-contains", action="append", default=[], metavar="KEY=SUBSTRING")
    ap.add_argument("--min-cells", type=int, default=1)
    args = ap.parse_args()

    cells = load(args.arm)
    print(f"arm {args.arm}: {len(cells)} cells")
    if len(cells) < args.min_cells:
        print(f"FAIL: need at least {args.min_cells} cells, found {len(cells)}")
        return 1
    if not cells:
        return 1

    fails = []

    # --- 1. every declared expectation, on every cell -------------------------------
    for spec in args.expect:
        key, _, want = spec.partition("=")
        bad = []
        for name, d in cells.items():
            if key not in d:
                bad.append(f"{name}: key absent (run predates the provenance fields?)")
            elif coerce(d[key]) != coerce(want):
                bad.append(f"{name}: {key}={d[key]!r}, expected {want!r}")
        if bad:
            fails.append((spec, bad))
        print(f"  {'FAIL' if bad else 'ok  '}  {key} == {want}"
              f"{'' if not bad else f'   ({len(bad)}/{len(cells)} cells violate)'}")

    for spec in args.expect_contains:
        key, _, want = spec.partition("=")
        bad = [f"{n}: {key}={d.get(key)!r}" for n, d in cells.items()
               if want not in str(d.get(key, ""))]
        if bad:
            fails.append((spec, bad))
        print(f"  {'FAIL' if bad else 'ok  '}  {key} contains {want!r}"
              f"{'' if not bad else f'   ({len(bad)}/{len(cells)} cells violate)'}")

    # --- 2. the arm must be internally consistent -----------------------------------
    print("\n  internal consistency (a config field must not vary across cells of one arm):")
    varied = 0
    for k in CONFIG_KEYS:
        vals = defaultdict(list)
        for name, d in cells.items():
            if k in d:
                vals[json.dumps(d[k])].append(name)
        # cap_gate legitimately varies per cell when auto-calibrated -- that is the bug
        # signature, not an exemption, so it is reported rather than skipped.
        if len(vals) > 1:
            varied += 1
            shown = {v: len(n) for v, n in vals.items()}
            print(f"    VARIES  {k}: {shown}")
    if not varied:
        print("    all config fields constant across cells")

    # --- 3. optional diff against a reference arm -----------------------------------
    if args.reference:
        ref = load(args.reference)
        print(f"\n  config diff vs {args.reference} ({len(ref)} cells), on shared cells:")
        shared = sorted(set(cells) & set(ref))
        diffs = defaultdict(set)
        for name in shared:
            for k in CONFIG_KEYS:
                a, b = cells[name].get(k, "<absent>"), ref[name].get(k, "<absent>")
                if json.dumps(a) != json.dumps(b):
                    diffs[k].add((json.dumps(a), json.dumps(b)))
        if not diffs:
            print("    identical")
        for k, pairs in sorted(diffs.items()):
            sample = list(pairs)[:3]
            print(f"    {k}: " + "; ".join(f"{a} vs {b}" for a, b in sample)
                  + ("  ..." if len(pairs) > 3 else ""))

    print()
    if fails:
        print(f"RESULT: FAIL -- {len(fails)} expectation(s) violated")
        for spec, bad in fails:
            print(f"  {spec}")
            for line in bad[:5]:
                print(f"     {line}")
            if len(bad) > 5:
                print(f"     ... and {len(bad) - 5} more")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
