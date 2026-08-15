#!/usr/bin/env python3
"""Score the W3 (joint gates+LoRA) endpoints on the common ruler (2026-08-14).

plans/2026-08-14-w3-joint-gates-lora.md. Each W3 run (one cell x one delta x one lambda) leaves
a trained checkpoint whose hard-concrete gates can sit at fractional values at eval time. The
paper's unit of comparison is a BINARY edge set, so this pass snaps each run's gates to 0/1
(hard_mask -> set_circuit, the same determinization every discovery arm uses) and scores the
resulting circuit with task_acc (forced-choice metrics, BOS-fixed) on:
    val  = eval_big.jsonl (2000 examples)  -- the selection split, same as every other arm
    test = test.jsonl     (1000 examples)  -- sealed; recorded per endpoint, quoted only for
                                              the endpoint the analysis-time rule selects
Also recorded per endpoint: the trained model's own gates-open val/test (its "full graph"),
and the probe_mixed capability-ppl ratio -- the same quantity C1's 1.05 gate is enforced on --
so a W3 win bought by trashing general capability is visible in the artifact itself. The base
(adapter-disabled) model's bar is scored once per cell.

Selection happens at ANALYSIS time with the same symmetric val rule as the headline; this
script only measures and records.

    python scripts/score_w3_endpoints.py --task ioi --model gpt2
    python scripts/score_w3_endpoints.py --task ioi --model gpt2 --ckpt-root checkpoints/lgate_ioi_gpt2 \
        --flat-layout --out results/w3joint/sanity_lgate_ioi_gpt2.json   # frozen-arm cross-check

Writes results/w3joint/<task>_<model>.json (skip-if-exists; --force to recompute).
"""
import argparse
import json
from pathlib import Path

import torch

from ilm.data.dataset import PairCollator
from ilm.discovery.pipeline import set_circuit
from ilm.training.setup import load_tokenizer
from scripts._task_eval import MPATH, ROOT, load_masked, make_ds, kept_edges, task_acc
from scripts.prune_heal_gradual import capability_probe

DELTAS = ["1.0", "10.0"]
LAMBDAS = ["0.002", "0.005", "0.02", "0.05", "0.1", "0.3", "0.8", "2.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt-root", default=None,
                    help="override checkpoint root (default checkpoints/w3joint/<task>_<model>)")
    ap.add_argument("--flat-layout", action="store_true",
                    help="checkpoints live at <root>/lam_<l> (the lgate layout) instead of "
                         "<root>/d<delta>_lam_<l>; used for the frozen-arm sanity cross-check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--val-batches", type=int, default=250, help="250x8 = the 2000-example split")
    ap.add_argument("--test-batches", type=int, default=125, help="125x8 = the 1000-example split")
    ap.add_argument("--probe", default="data/probe_mixed.jsonl")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dst = Path(args.out) if args.out else ROOT / f"results/w3joint/{args.task}_{args.model}.json"
    if dst.exists() and not args.force:
        print(f"{args.task}/{args.model}: already scored at {dst}; skipping (pass --force to "
              f"recompute). Guard against an overlapping resubmission racing this file.")
        return
    root = Path(args.ckpt_root) if args.ckpt_root else \
        ROOT / f"checkpoints/w3joint/{args.task}_{args.model}"

    tok = load_tokenizer(str(ROOT / MPATH[args.model]))
    coll = PairCollator(tok.pad_token_id)
    val_ds = make_ds(args.task, args.model, tok, split="eval_big")
    test_ds = make_ds(args.task, args.model, tok, split="test")
    with open(ROOT / args.probe) as f:
        probe_prompts = [json.loads(l)["clean_prompt"] for l in f if l.strip()]

    runs = [(None, l) for l in LAMBDAS] if args.flat_layout else \
        [(d, l) for d in DELTAS for l in LAMBDAS]

    endpoints, base_bar = [], None
    for dlt, lam in runs:
        ck = root / (f"lam_{lam}" if dlt is None else f"d{dlt}_lam{lam}")
        if not (ck / "gates.pt").exists():
            print(f"  MISSING {ck} -- run not finished; recorded as absent, not silently skipped")
            endpoints.append({"delta": dlt, "lambda": lam, "status": "missing"})
            continue
        cfg = json.loads((ck / "config.json").read_text())
        m = load_masked(args.model, ck / "lora", ck / "gates.pt", device, granularity="head")

        # LoRA B-matrices are zero-initialized, so ||B||>0 is direct artifact-level proof the
        # weights trained (and ==0 that they were frozen). Recorded, not just asserted.
        b_norm = sum(float(p.float().norm()) for n, p in m.peft_model.named_parameters()
                     if "lora_B" in n)

        kept = kept_edges(m)
        set_circuit(m, kept)                       # binarize: the paper's unit is a 0/1 edge set
        av = task_acc(m, args.task, val_ds, coll, device, nb=args.val_batches)
        at = task_acc(m, args.task, test_ds, coll, device, nb=args.test_batches)

        m.gate.log_alpha.data.fill_(20.0)          # the trained model's own full graph
        fv = task_acc(m, args.task, val_ds, coll, device, nb=args.val_batches)
        ft = task_acc(m, args.task, test_ds, coll, device, nb=args.test_batches)
        cap = capability_probe(m, probe_prompts, tok, device, max_length=128)

        if base_bar is None:                        # adapter off = the original model's bar
            with m.peft_model.disable_adapter():
                base_bar = {
                    "val": task_acc(m, args.task, val_ds, coll, device, nb=args.val_batches),
                    "test": task_acc(m, args.task, test_ds, coll, device, nb=args.test_batches),
                }

        row = {"delta": dlt, "lambda": lam, "status": "ok",
               "edges": len(kept), "n_edges_total": len(m.edges),
               "acc_val": round(av, 4), "acc_test": round(at, 4),
               "full_val": round(fv, 4), "full_test": round(ft, 4),
               "cap_ppl_ratio": cap["probe_ppl_ratio"], "lora_B_norm": round(b_norm, 4),
               "cfg_freeze_weights": cfg["freeze_weights"],
               "cfg_granularity": cfg["graph_granularity"],
               "cfg_lambda_sparse": cfg["loss"]["lambda_sparse"],
               "cfg_delta_preserve": cfg["loss"]["delta_preserve"],
               "cfg_num_steps": cfg["num_steps"], "cfg_model_lr": cfg["model_lr"],
               "cfg_mask_lr": cfg["mask_lr"], "cfg_seed": cfg["seed"]}
        endpoints.append(row)
        print(f"  d={dlt} lam={lam}: {len(kept)}e val={av:.3f} test={at:.3f} "
              f"full_val={fv:.3f} cap={cap['probe_ppl_ratio']:.4f} ||B||={b_norm:.2f}", flush=True)
        del m
        torch.cuda.empty_cache()

    out = {"task": args.task, "model": args.model, "granularity": "head", "arm": "w3joint",
           "protocol": "binarized gates; select=val(2000) report=test(1000); "
                       "selection applied at analysis time, same rule as the headline",
           "probe": args.probe, "n_val": args.val_batches * 8, "n_test": args.test_batches * 8,
           "base_bar": base_bar, "endpoints": endpoints}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
