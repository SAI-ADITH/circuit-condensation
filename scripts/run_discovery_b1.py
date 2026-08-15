#!/usr/bin/env python3
"""EAP-IG discovery on the B1 (heal-only, no-prune) model (2026-08-06).

R1 M3 / R2 B2, the decisive deflationary baseline: maybe ordinary distillation-based
post-training already makes circuits smaller, and the pruning loop contributes nothing.
Our existing B1 control answers this with the EAP-IG *participation ratio* -- a measure of
attribution spread -- which is not the quantity the paper reports anywhere else. The direct
experiment is to run discovery on the healed model and report the SIZE.

Protocol is identical to scripts/run_discovery_valselect.py so the numbers drop straight
into the same comparison: rank on train, sweep budgets, score every budget on the enlarged
validation split AND on the sealed test split, and select afterwards by whichever rule the
analysis uses. The only difference is the model: base weights plus the B1 heal-only adapter,
with every gate open, instead of the untouched base model.

Three outcomes, all publishable:
  size(B1 discovery) ~ size(frozen discovery)  -> healing alone does nothing; loop is the cause
  size(B1 discovery) ~ size(C1)                -> the loop is unnecessary; distillation suffices
  in between                                   -> report the split honestly

Writes results/discovery/head_b1/<task>/<model>.json
"""
import argparse
import json
from pathlib import Path

import torch

from ilm.data.dataset import PairCollator
from ilm.discovery.methods import EAPIG
from ilm.discovery.pipeline import _budgets, set_circuit
from ilm.training.setup import load_tokenizer
from scripts._task_eval import MPATH, ROOT, TASK_DIR, load_masked, make_ds, task_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--b1-ckpt", required=True, help="dir with lora_healed/ + gates_healed.pt")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="kl", help="which heal objective produced the checkpoint")
    ap.add_argument("--val-batches", type=int, default=250, help="250x8 = the 2000-example split")
    ap.add_argument("--test-batches", type=int, default=125, help="125x8 = the 1000-example split")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = Path(args.b1_ckpt)
    if not (ck / "gates_healed.pt").exists():
        print(f"no B1 checkpoint at {ck}; abort")
        return

    tok = load_tokenizer(str(ROOT / MPATH[args.model]))
    coll = PairCollator(tok.pad_token_id)
    model = load_masked(args.model, ck / "lora_healed", ck / "gates_healed.pt",
                        device, granularity="head")
    # B1 never pruned, so its saved gates are already all-open; force it regardless, because
    # the point of this arm is discovery over the FULL graph of the healed model.
    model.gate.log_alpha.data.fill_(20.0)

    E = len(model.edges)
    budgets = _budgets(E)
    train_ds = make_ds(args.task, args.model, tok, split="train")
    val_ds = make_ds(args.task, args.model, tok, split="eval_big")
    test_ds = make_ds(args.task, args.model, tok, split="test")

    full_val = task_acc(model, args.task, val_ds, coll, device, nb=args.val_batches)
    full_test = task_acc(model, args.task, test_ds, coll, device, nb=args.test_batches)
    print(f"{args.task}/{args.model}: {E} edges, B1-healed full graph "
          f"val={full_val:.3f} test={full_test:.3f}", flush=True)

    # rank_edges returns per-edge SCORES, not an ordering. Feeding the scores straight to
    # set_circuit() silently casts them to long (attributions are ~0), so ~one edge stays on
    # at every budget and the whole curve comes out flat. argsort is what makes it an order --
    # run_discovery_valselect.py does the same thing, and this script was missing it.
    scores = EAPIG().rank_edges(model, train_ds, coll, device)
    order = torch.argsort(scores, descending=True).tolist()

    pts = []
    for k in budgets:
        set_circuit(model, set(order[:k]))
        av = task_acc(model, args.task, val_ds, coll, device, nb=args.val_batches)
        at = task_acc(model, args.task, test_ds, coll, device, nb=args.test_batches)
        pts.append({"edges": k, "acc_val": round(av, 4), "acc_test": round(at, 4)})
        print(f"   {k:6d}e  val={av:.3f}  test={at:.3f}", flush=True)

    out = {"task": args.task, "model": args.model, "n_edges": E, "granularity": "head",
           "arm": f"B1-healed ({args.tag}, no-prune) + EAP-IG discovery", "seed": args.seed,
           "heal_objective": args.tag,
           "b1_ckpt": str(ck), "full_val": round(full_val, 4), "full_test": round(full_test, 4),
           "protocol": "rank=train select=val(2000) report=test(1000)",
           "curves": {"EAP-IG": {str(args.seed): pts}}}
    dst = ROOT / f"results/discovery/head_b1_{args.tag}/{args.task}/{args.model}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
