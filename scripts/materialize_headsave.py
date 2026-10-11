#!/usr/bin/env python3
"""Materialise the legacy `headsave` layout for a controller arm from its saved rounds.

Why this exists
---------------
The payoff/verification suite (run_pairwise_completeness, run_target_specificity,
run_iia_wang, run_failure_prediction, run_subset_completeness, run_role_recovery,
run_orig_faith, run_unlearn_robustness, cross_corruption.slurm, ...) loads a saved
condensed model from

    checkpoints/headsave/<arm>/seed<N>/<prefix>_<model>/{lora_healed, gates_healed.pt, kept_edges.json}

That layout is written by prune_heal_gradual.py's `--save-healed` path. The C1 grid
(slurm/regrid_gate.slurm) instead uses `SAVE_ROUNDS`, which writes a
rolling window of the last few accepted rounds to

    checkpoints/<arm>/seed<N>/<prefix>_<model>/r<NNN>_e<edges>/{lora, gates.pt, meta.json}

Same content, different names, and no kept_edges.json. Rather than edit twenty
downstream scripts' file handling, this translates the round window into the layout
they already expect. Symlinks, not copies: the c1gate round window is ~27 GB.

The kept set is re-derived from the gate parameters rather than trusted from a
filename. That derivation is checked against a kept_edges.json written by the controller
itself (`--save-healed`) by `--verify-only`, which is the control for this whole script: if the materialiser and
the controller disagree about what "the circuit" is, every downstream number is wrong.

Usage
-----
    python scripts/materialize_headsave.py --arm c1gate
    python scripts/materialize_headsave.py --arm <arm> --verify-only    # the control, on an
                                                                        # arm run with --save-healed
"""
import argparse
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Hard-concrete constants, mirrored from ilm/masking/hard_concrete.py. Duplicated on
# purpose: this script must be able to read an archived checkpoint even if the module's
# defaults are ever changed, and a silent constant drift would silently redefine every
# saved circuit. If these ever diverge, --verify-only fails loudly.
GAMMA, ZETA = -0.1, 1.1
THRESHOLD = 0.5

PREFIX = {"ioi": "ioi", "agreement": "agr", "induction": "ind", "doc": "doc"}
SEEDS = (11, 22, 33)


def kept_from_gates(gate_path: Path) -> list:
    """Deterministic 0/1 circuit from a saved gate state dict.

    Matches HardConcreteGate.hard_mask(): the deterministic gate value is
    sigmoid(log_alpha) stretched to [GAMMA, ZETA], clamped, thresholded at 0.5.
    """
    sd = torch.load(gate_path, map_location="cpu")
    la = sd["log_alpha"].float()
    s = torch.sigmoid(la) * (ZETA - GAMMA) + GAMMA
    return s.clamp(0.0, 1.0).gt(THRESHOLD).nonzero().flatten().tolist()


def pick_round(cell_dir: Path, edges: int, acc):
    """The saved round that IS the reported endpoint.

    Rounds are named r<NNN>_e<edges>. Edge count alone can repeat across rounds (the
    controller can re-heal at the same size after a rejected cut), so when it does,
    prefer the round whose recorded eval_acc matches the knee's reported accuracy, and
    fall back to the latest such round. Picking the wrong one would hand downstream a
    differently-healed model with the right edge count -- a silent, plausible-looking
    error, which is the kind worth spending code on.
    """
    cands = sorted(cell_dir.glob(f"r*_e{edges}"), key=lambda p: p.name)
    if not cands:
        return None, "no round dir at the endpoint edge count"
    if len(cands) == 1:
        return cands[0], None
    if acc is not None:
        exact = []
        for c in cands:
            try:
                m = json.loads((c / "meta.json").read_text())
            except Exception:
                continue
            ea = m.get("eval_acc")
            if ea is not None and abs(ea - acc) < 1e-9:
                exact.append(c)
        if len(exact) == 1:
            return exact[0], None
        if exact:
            return exact[-1], f"{len(exact)} rounds match edges+acc; took latest"
    return cands[-1], f"{len(cands)} rounds at e{edges}; took latest"


def relink(dst: Path, src: Path):
    """Point dst at src as a symlink, replacing whatever is there.

    Re-runnable by design: this script is expected to run once on the finished cells
    and again when the grid completes, and a half-materialised tree from an interrupted
    run must not survive into the second pass.
    """
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            import shutil
            shutil.rmtree(dst)
    dst.symlink_to(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="c1gate")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not write; instead check the derived kept set against an "
                         "existing headsave/<arm> tree. This is the control: run it on "
                         "an arm whose kept_edges.json the controller itself wrote.")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    grid = ROOT / "results/headgrid" / a.arm
    knees = sorted(f for s in seeds for f in (grid / f"seed{s}").glob("*_knee.json"))
    if not knees:
        raise SystemExit(f"no knee files under {grid}/seed{{{a.seeds}}}")

    ok = skipped = failed = 0
    notes, problems = [], []

    for f in knees:
        seed = int(re.search(r"seed(\d+)", str(f)).group(1))
        stem = f.name[: -len("_knee.json")]
        task, model = stem.split("_", 1)
        d = json.loads(f.read_text())
        edges = d.get("best_faithful_edges")
        acc = d.get("best_faithful_acc")
        cell = f"{task}/{model}/s{seed}"
        if not edges:
            skipped += 1
            problems.append(f"{cell}: knee has no best_faithful_edges")
            continue

        name = f"{PREFIX[task]}_{model}"
        dst = ROOT / "checkpoints/headsave" / a.arm / f"seed{seed}" / name

        if a.verify_only:
            gp = dst / "gates_healed.pt"
            kp = dst / "kept_edges.json"
            if not (gp.exists() and kp.exists()):
                skipped += 1
                problems.append(f"{cell}: no existing headsave cell to verify")
                continue
            derived = kept_from_gates(gp)
            stored = sorted(json.loads(kp.read_text()))
            if derived == stored:
                ok += 1
            else:
                failed += 1
                problems.append(f"{cell}: derived {len(derived)} edges != stored {len(stored)}")
            continue

        src_cell = ROOT / "checkpoints" / a.arm / f"seed{seed}" / name
        rd, note = pick_round(src_cell, edges, acc)
        if rd is None:
            failed += 1
            problems.append(f"{cell}: {note} (wanted e{edges} under {src_cell})")
            continue
        if note:
            notes.append(f"{cell}: {note}")

        kept = kept_from_gates(rd / "gates.pt")
        if len(kept) != edges:
            # Refuse rather than write a cell whose circuit is not the reported endpoint.
            failed += 1
            problems.append(f"{cell}: {rd.name} gates give {len(kept)} edges, knee says {edges}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        relink(dst / "lora_healed", rd / "lora")
        relink(dst / "gates_healed.pt", rd / "gates.pt")
        (dst / "kept_edges.json").write_text(json.dumps(sorted(kept)))
        (dst / "source_round.json").write_text(json.dumps(
            {"arm": a.arm, "task": task, "model": model, "seed": seed,
             "round_dir": str(rd.relative_to(ROOT)), "edges": edges,
             "best_faithful_acc": acc}, indent=1))
        ok += 1

    verb = "verified" if a.verify_only else "materialised"
    print(f"arm={a.arm}  {verb}: {ok}   failed: {failed}   skipped: {skipped}   "
          f"(of {len(knees)} knee files)")
    for n in notes:
        print(f"  note: {n}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
