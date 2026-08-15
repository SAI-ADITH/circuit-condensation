#!/usr/bin/env python3
"""Frozen discovery curves scored on BOTH the validation and sealed test splits.

Why this exists (2026-08-04 review remediation, plan 2026-08-04-review-remediation.md): the
matched-accuracy comparison previously selected each frozen budget off the TEST-split curve --
test-set selection, reviewer fatal E2. The fix is to select every arm's budget on the
VALIDATION split under the controller's own faithfulness rule (val acc >= full-model val acc -
0.05) and then read the test accuracy of that one selected budget. This script produces the
missing ingredient: per-budget accuracy on eval.jsonl (200 ex, the controller's decision split)
alongside a fresh per-budget accuracy on test.jsonl (1000 ex).

The test-side numbers double as an independent verification of the 2026-07-30 regeneration of
results/discovery/head (reviewer fatal E1): post-processing diffs them against the stored curves.

Rankings mirror run_discovery_baseline.py exactly -- rank on the train split, same per-seed
shuffle, same budget grid -- so each (cell, seed) is paired with the C1 run that saw the same
data draw.

    python scripts/run_discovery_valselect.py --cells ioi/gpt2
"""
import argparse
import json
import os
from pathlib import Path

import torch

from ilm.config import TrainConfig
from ilm.data.dataset import PairCollator
from ilm.training.setup import build_masked_model, load_tokenizer
from ilm.discovery import EAP, EAPIG, RandomRank
from ilm.discovery.acdc import acdc_knee
from ilm.discovery.methods import BIN
from ilm.discovery.pipeline import _budgets, circuit_accuracy, set_circuit
from scripts.run_discovery_baseline import ALL_CELLS, MODELS, SEEDS, shuffle_rank
from scripts.task_finetune import build_datasets

ROOT = Path(os.environ.get("ILM_ROOT", Path(__file__).resolve().parent.parent))
ACDC_MAX_EDGES = 2800  # same tractability cutoff as run_discovery_baseline.py


def run_cell(task, model_name, device, seeds):
    mpath = f"{ROOT}/{MODELS[model_name]}"
    tdir = "docstring" if task == "doc" else task
    rank_path = f"{ROOT}/data/{tdir}/{model_name}/train.jsonl"
    val_path = f"{ROOT}/data/{tdir}/{model_name}/eval.jsonl"
    test_path = f"{ROOT}/data/{tdir}/{model_name}/test.jsonl"
    cfg = TrainConfig(model_path=mpath, task=task, graph_granularity="head", dtype="float32")
    tok = load_tokenizer(mpath)
    model = build_masked_model(cfg, device)
    collator = PairCollator(tok.pad_token_id)
    E = len(model.edges)
    budgets = _budgets(E)
    print(f"\n### {task}/{model_name}: {E} edges, {len(budgets)} budgets", flush=True)
    out = {"task": task, "model": model_name, "n_edges": E, "granularity": "head",
           "seeds": seeds, "budgets": budgets, "protocol": "rank=train select=val report=test",
           "curves": {}}

    la = model.gate.log_alpha
    saved = la.data.clone()
    try:
        for seed in seeds:
            torch.manual_seed(seed)  # same call order as run_discovery_baseline -> paired draw
            rank_ds, val_ds = build_datasets(task, rank_path, val_path, tok, max_length=64)
            _, test_ds = build_datasets(task, rank_path, test_path, tok, max_length=64)
            rank_ds = shuffle_rank(rank_ds, seed)
            for m in (EAP(), EAPIG(n_steps=5), RandomRank()):
                la.data.fill_(BIN)  # rank on the full frozen model, all edges ON
                order = torch.argsort(
                    m.rank_edges(model, rank_ds, collator, device, rank_batches=8),
                    descending=True).tolist()
                curve = []
                for k in budgets:
                    set_circuit(model, set(order[:k]))
                    acc_val = circuit_accuracy(model, val_ds, collator, device, task,
                                               num_batches=25)     # full 200-example val split
                    acc_test = circuit_accuracy(model, test_ds, collator, device, task,
                                                num_batches=125)   # full 1000-example test split
                    curve.append({"edges": k, "acc_val": round(acc_val, 4),
                                  "acc_test": round(acc_test, 4)})
                out["curves"].setdefault(m.name, {})[seed] = curve
                print(f"  s{seed} [{m.name:8s}] " +
                      " ".join(f"{p['edges']}e:{p['acc_val']:.2f}/{p['acc_test']:.2f}"
                               for p in curve[:6]) + " ...", flush=True)
            if E <= ACDC_MAX_EDGES:
                # ACDC prunes on train; its val curve here pairs by tau with the stored test curve.
                la.data.fill_(BIN)
                ac = acdc_knee(model, rank_ds, collator, device, task, eval_ds=val_ds,
                               taus=(0.0007, 0.003, 0.012, 0.05), rank_batches=2, eval_batches=25)
                out["curves"].setdefault("ACDC", {})[seed] = ac["curve"]
                print(f"  s{seed} [ACDC    ] " +
                      " ".join(f"t{p['tau']}:{p['edges']}e:{p['acc']:.2f}" for p in ac["curve"]),
                      flush=True)
    finally:
        la.data.copy_(saved)

    dst = ROOT / "results" / "discovery" / "head_valsel" / task / f"{model_name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    print(f"  -> {dst.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(ALL_CELLS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cell in a.cells.split(","):
        task, model_name = cell.split("/")
        try:
            run_cell(task, model_name, device, seeds)
        except Exception as e:
            import traceback
            print(f"  !! {cell} FAILED: {e}"); traceback.print_exc()
    print("\nDISCOVERY_VALSELECT_DONE")


if __name__ == "__main__":
    main()
