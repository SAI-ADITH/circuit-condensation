#!/usr/bin/env python3
"""Run the frozen-model circuit-discovery baselines (EAP, EAP-IG, ACDC, exact ActivationPatching)
across cells and write comparable knee curves to results/discovery/<task>/<model>.json. These are
the paper's "how small a circuit does the FIELD's discovery get on the frozen model" table, to sit
opposite our C1 (reshaped) knees. EAP-GP is NOT run (no public source; cited as related work).

Method selection is by graph size: gradient methods (EAP, EAP-IG) always run; the O(edges) exact
methods (ACDC, ActivationPatching) run only when edges <= --exact-max (default 700), else skipped.

    python scripts/run_discovery_baseline.py                      # default cell set, on a GPU node
    python scripts/run_discovery_baseline.py --cells ioi/gpt2
"""
import os
import argparse
import json
from pathlib import Path

import torch

from ilm.config import TrainConfig
from ilm.data.dataset import PairCollator
from ilm.training.setup import build_masked_model, load_tokenizer
from ilm.discovery import EAP, EAPIG, RandomRank, discovery_knee
from ilm.discovery.acdc import acdc_knee
from scripts.task_finetune import build_datasets

ROOT = Path(os.environ.get("ILM_ROOT", Path(__file__).resolve().parent.parent))
MODELS = {"gpt2": "models/gpt2", "llama": "models/Llama-3.2-1B-Instruct",
          "gemma1b": "models/gemma-3-1b-it", "qwen3-1.7b": "models/Qwen3-1.7B",
          "llama3b": "models/Llama-3.2-3B-Instruct", "gemma4b": "models/gemma-3-4b-it",
          "qwen3-0.6b": "models/Qwen3-0.6B", "qwen3-4b": "models/Qwen3-4B"}
ALL_CELLS = [f"{t}/{m}" for t in ("ioi", "agreement", "induction", "doc") for m in MODELS]

# BASELINE SET (2026-07-17, per user): EAP, EAP-IG, Random run on EVERY cell (all gradient/O(1) in
# forward passes, so they scale to the 44k-edge head graphs). ACDC runs on SMALL cells only: it is
# greedy and O(edges) forward passes per threshold, ~800x more expensive on qwen3-4b than gpt2, so
# it is physically infeasible on the big head graphs -- the same limit ACDC's own paper hit. Exact
# single-edge activation patching was DROPPED (also O(edges), and EAP-IG is the primary attribution
# baseline). C0/C0.5 are produced by the condensation controller with frozen weights, not here.
ACDC_MAX_EDGES = 2800   # ~gpt2 head (2041) in; llama/gemma/qwen head graphs (>=3537) out.
SEEDS = [11, 22, 33]    # SAME seeds as C1, so baseline-seed-k is paired with C1-seed-k (same data draw).


def shuffle_rank(ds, seed):
    """Permute the ranking examples -- the frozen analog of C1's healing data order. Same-seed as
    C1 makes the comparison PAIRED: C1 and the baseline see the same draw at each seed."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds.examples), generator=g).tolist()
    ds.examples = [ds.examples[i] for i in perm]
    return ds


def run_cell(task, model_name, device, granularity, rank_batches, eval_batches, seeds):
    mpath = f"{ROOT}/{MODELS[model_name]}"
    tdir = "docstring" if task == "doc" else task
    # Protocol parity with C1: rank on TRAIN (2000 ex), score on the SEALED TEST split. (Earlier
    # this used eval.jsonl for both, an in-sample ranking that flattered the baselines.)
    rank_data = f"{ROOT}/data/{tdir}/{model_name}/train.jsonl"
    data = f"{ROOT}/data/{tdir}/{model_name}/test.jsonl"
    cfg = TrainConfig(model_path=mpath, task=task, graph_granularity=granularity, dtype="float32")
    tok = load_tokenizer(mpath)
    model = build_masked_model(cfg, device)
    collator = PairCollator(tok.pad_token_id)
    E = len(model.edges)
    do_acdc = E <= ACDC_MAX_EDGES
    print(f"\n### {task}/{model_name}: {E} edges  (ACDC {'ON' if do_acdc else 'skipped: too big'})",
          flush=True)
    # curves[method][seed] = knee curve; per-seed so the main table can report mean+-std, paired w/ C1.
    out = {"task": task, "model": model_name, "n_edges": E, "granularity": granularity,
           "seeds": seeds, "curves": {}}

    for seed in seeds:
        torch.manual_seed(seed)      # makes RandomRank reproducible per seed
        rank_ds, eval_ds = build_datasets(task, rank_data, data, tok, max_length=64)
        rank_ds = shuffle_rank(rank_ds, seed)
        methods = [EAP(), EAPIG(n_steps=5), RandomRank()]
        for m in methods:
            res = discovery_knee(model, m, eval_ds, collator, device, task, rank_dataset=rank_ds,
                                 rank_batches=rank_batches, eval_batches=eval_batches)
            out["curves"].setdefault(m.name, {})[seed] = res["curve"]
            print(f"  s{seed} [{m.name:8s}] " +
                  "  ".join(f"{p['edges']}e:{p['acc']:.2f}" for p in res["curve"]), flush=True)
        if do_acdc:
            # ACDC prunes on train (rank_ds), scores on the sealed test split (eval_ds); 4 taus.
            ac = acdc_knee(model, rank_ds, collator, device, task, eval_ds=eval_ds,
                           taus=(0.0007, 0.003, 0.012, 0.05), rank_batches=2, eval_batches=eval_batches)
            out["curves"].setdefault("ACDC", {})[seed] = ac["curve"]
            print(f"  s{seed} [ACDC    ] " +
                  "  ".join(f"t{p['tau']}:{p['edges']}e:{p['acc']:.2f}" for p in ac["curve"]), flush=True)

    dst = ROOT / "results" / "discovery" / granularity / task / f"{model_name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    print(f"  -> {dst.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(ALL_CELLS))
    ap.add_argument("--granularity", default="head", choices=["head", "block"])
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--rank-batches", type=int, default=8)
    # The sealed test sets contain 1000 examples and evaluation uses batches of 8.
    # Use the full split by default so frozen curves and C1 endpoints share the
    # same evaluation set. The option remains available for smoke tests.
    ap.add_argument("--eval-batches", type=int, default=125)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cell in a.cells.split(","):
        task, model_name = cell.split("/")
        try:
            run_cell(task, model_name, device, a.granularity, a.rank_batches, a.eval_batches, seeds)
        except Exception as e:
            import traceback
            print(f"  !! {cell} FAILED: {e}"); traceback.print_exc()
    print("\nDISCOVERY_BASELINE_DONE")


if __name__ == "__main__":
    main()
