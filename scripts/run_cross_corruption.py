#!/usr/bin/env python3
"""Does C1 survive a corruption distribution it was never trained on? (2026-08-06)

The strongest objection to the whole paper: circuit size is defined under interchange
ablation on MINIMAL PAIRS, and the adapter is trained for hundreds of steps with the
corrupted-run activations present in every forward pass. Nothing in the objective
stops the adapter from reading the answer out of the counterfactual stream. If it
does, the small circuits are an artifact of the ruler, and the signature would be:
fine under the training corruption, collapse under any other.

This scores the SAME retained circuit under four corruption families, keeping the
clean prompt and the answer fixed and changing only what a closed edge reads:

  abc        the training family: acting subject replaced by a third name
  template   corrupted prompt uses a DIFFERENT template than the clean one
  resampled  corrupted prompt is another example's clean prompt (same task)
  unrelated  corrupted prompt is off-task text (out of distribution)

The frozen EAP-IG circuit at its own validation-selected size gets the identical
treatment, because the question is not "does condensed degrade" (every small circuit
degrades) but "does condensed degrade MORE than a frozen circuit of comparable size".
Reported as retention: accuracy under family F divided by accuracy under abc.

Writes results/transfer/cross_corruption_<task>_<model>_s<seed>.json
"""
import argparse
import json
import random
from pathlib import Path

import torch

from ilm.data.dataset import PairCollator
from ilm.discovery.methods import EAPIG
from ilm.discovery.pipeline import set_circuit
from ilm.masking.edge_masked_model import EdgeMaskedTransformer
from ilm.training.setup import load_tokenizer
from transformers import AutoModelForCausalLM
from scripts._task_eval import (ROOT, MPATH, TASK_DIR, condensed_ckpt, kept_edges,
                                load_condensed, make_ds, task_acc)

OFFTASK = [
    "The mitochondrion is an organelle found in most eukaryotic cells, and it",
    "Numerical integration approximates a definite integral by evaluating the",
    "In 1847 the railway between the two cities was completed, after which",
    "Photosynthesis converts light energy into chemical energy stored in",
    "The compiler emits an intermediate representation before it performs",
    "Rainfall in the region is highly seasonal, with most precipitation",
]


def rewrite_corruption(records, family, rng):
    """Return records with 'corrupted_prompt' replaced per family; clean/answer fixed."""
    out = []
    n = len(records)
    for i, r in enumerate(records):
        r = dict(r)
        if family == "abc":
            pass
        elif family == "resampled":
            # another example's CLEAN prompt: a valid same-task activation that has
            # nothing to do with this example's names/positions
            r["corrupted_prompt"] = records[(i + 1 + rng.randrange(n - 1)) % n]["clean_prompt"]
        elif family == "template":
            # a different example's corrupted prompt, which in this data means a
            # different template instantiation than this example's clean prompt
            j = (i + n // 2 + 1) % n
            r["corrupted_prompt"] = records[j]["corrupted_prompt"]
        elif family == "unrelated":
            r["corrupted_prompt"] = OFFTASK[i % len(OFFTASK)]
        else:
            raise ValueError(family)
        out.append(r)
    return out


def load_records(task, model, split="test"):
    f = ROOT / "data" / TASK_DIR[task] / model / f"{split}.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()]


def ds_from_records(task, model, tok, records, tmpdir):
    """make_ds reads a file, so materialize the rewritten split next to it."""
    p = Path(tmpdir) / f"{task}_{model}_tmp.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # temporarily point make_ds at our file by copying it into the expected location
    dst = ROOT / "data" / TASK_DIR[task] / model / "_xcorrupt.jsonl"
    dst.write_text(p.read_text())
    return make_ds(task, model, tok, split="_xcorrupt")


def frozen_selected_edges(task, model, seed):
    """Val-selected frozen EAP-IG budget for this cell, from the valsel curves."""
    f = ROOT / f"results/discovery/head_valsel/{task}/{model}.json"
    if not f.exists():
        return None
    curves = json.loads(f.read_text())["curves"]
    pts = (curves.get("EAP-IG") or {}).get(str(seed)) or []
    pts = [p for p in pts if p.get("acc_val") is not None]
    if not pts:
        return None
    full = max(pts, key=lambda p: p["edges"])
    bar = full["acc_val"] - 0.05
    ok = [p for p in pts if p["acc_val"] >= bar]
    return min(ok, key=lambda p: p["edges"])["edges"] if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batches", type=int, default=25)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    if condensed_ckpt(args.task, args.model) is None:
        print("no condensed checkpoint; abort")
        return
    tok = load_tokenizer(str(ROOT / MPATH[args.model]))
    coll = PairCollator(tok.pad_token_id)
    recs = load_records(args.task, args.model)

    m = load_condensed(args.task, args.model, device, granularity="head")
    circ = kept_edges(m)
    E = len(m.edges)
    fz_k = frozen_selected_edges(args.task, args.model, args.seed)
    print(f"{args.task}/{args.model} s{args.seed}: C1 {len(circ)}e of {E}; frozen budget {fz_k}",
          flush=True)

    # FROZEN control at the same budget, on the ORIGINAL weights. Every small
    # interchange circuit is expected to degrade off-convention; the question is only
    # whether the condensed one degrades MORE than a frozen circuit of the same size.
    fz = None
    if fz_k:
        base = AutoModelForCausalLM.from_pretrained(str(ROOT / MPATH[args.model]),
                                                    torch_dtype=torch.float32,
                                                    attn_implementation="eager")
        fz = EdgeMaskedTransformer(base, granularity="head").to(device)
        fz.eval()
        ds_rank = ds_from_records(args.task, args.model, tok,
                                  rewrite_corruption(recs, "abc", rng), ROOT / "results/tmp_xcorrupt")
        order = EAPIG().rank_edges(fz, ds_rank, coll, device)
        fz_circ = set(order[:fz_k].tolist() if hasattr(order, "tolist") else list(order)[:fz_k])
        print(f"   frozen circuit: {len(fz_circ)} edges", flush=True)

    out = {"task": args.task, "model": args.model, "seed": args.seed,
           "c1_edges": len(circ), "graph_edges": E, "frozen_edges": fz_k, "families": {}}
    tmp = ROOT / "results/tmp_xcorrupt"
    for fam in ("abc", "template", "resampled", "unrelated"):
        ds = ds_from_records(args.task, args.model, tok,
                             rewrite_corruption(recs, fam, rng), tmp)
        set_circuit(m, circ)
        acc_c1 = task_acc(m, args.task, ds, coll, device, nb=args.batches)
        rec = {"c1_acc": round(acc_c1, 4)}
        if fz is not None:
            set_circuit(fz, fz_circ)
            rec["frozen_acc"] = round(task_acc(fz, args.task, ds, coll, device, nb=args.batches), 4)
            fz.gate.log_alpha.data.fill_(20.0)
            rec["frozen_full_acc"] = round(task_acc(fz, args.task, ds, coll, device, nb=args.batches), 4)
        out["families"][fam] = rec
        print(f"   {fam:10s} C1 {acc_c1:.3f}"
              + (f"  frozen({fz_k}e) {rec['frozen_acc']:.3f}  frozen-full {rec['frozen_full_acc']:.3f}"
                 if fz is not None else ""), flush=True)
    dst = ROOT / f"results/transfer/cross_corruption_{args.task}_{args.model}_s{args.seed}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
