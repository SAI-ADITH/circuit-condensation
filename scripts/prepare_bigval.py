#!/usr/bin/env python3
"""Larger controller-validation split, disjoint from the existing splits (2026-08-06).

Why: the gate compared a point estimate on 200 examples against a 0.05 tolerance and was
consulted ~32 times per run keeping the best retry, so the accepted endpoint was optimistic
and a third of them missed the same tolerance on the sealed split. The noise on 200 examples
is +-0.021 to +-0.031, i.e. the tolerance is under two standard errors. At 2000 it is
+-0.007 to +-0.010, which leaves room for a confidence-bound accept rule
(prune_heal_gradual.py --accept-lcb).

This writes eval_big.jsonl per task/model WITHOUT touching train.jsonl or test.jsonl: the
existing three splits are loaded first and passed as the exclusion set, so the new file is
disjoint from all of them by construction. The sealed test split is therefore still sealed,
and every already-measured frozen curve stays comparable.

Splits are not a fixed pie here -- the data is generated from templates and we use a small
fraction of the available combinations (IOI alone admits ~536k), so enlarging validation
takes nothing from training.

Usage: python3 scripts/prepare_bigval.py [--n 2000] [--tasks ioi,agreement,induction,doc]
"""
import argparse
import json
import os
from pathlib import Path

from ilm.training.setup import load_tokenizer

ROOT = Path(os.environ.get("ILM_ROOT", Path(__file__).resolve().parent.parent))
MPATH = {"gpt2": "models/gpt2", "llama": "models/Llama-3.2-1B-Instruct",
         "llama3b": "models/Llama-3.2-3B-Instruct", "gemma1b": "models/gemma-3-1b-it",
         "gemma4b": "models/gemma-3-4b-it", "qwen3-0.6b": "models/Qwen3-0.6B",
         "qwen3-1.7b": "models/Qwen3-1.7B", "qwen3-4b": "models/Qwen3-4B"}
MODELS = list(MPATH)
TASK_DIR = {"ioi": "ioi", "agreement": "agreement", "induction": "induction", "doc": "docstring"}


def builder(task):
    if task == "ioi":
        from ilm.data.ioi import build_samples, save_samples
    elif task == "agreement":
        from ilm.data.agreement import build_samples, save_samples
    elif task == "induction":
        from ilm.data.induction import build_samples, save_samples
    else:
        from ilm.data.docstring import build_samples, save_samples
    return build_samples, save_samples


def existing_prompts(task, model):
    """Every clean prompt already used, so the new split can exclude all of them."""
    seen = set()
    for split in ("train", "eval", "test"):
        f = ROOT / "data" / TASK_DIR[task] / model / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            if line.strip():
                seen.add(json.loads(line)["clean_prompt"])
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tasks", default="ioi,agreement,induction,doc")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=777)   # not 0/11/22/33: distinct draw
    ap.add_argument("--name", default="eval_big.jsonl")
    args = ap.parse_args()

    for task in args.tasks.split(","):
        build_samples, save_samples = builder(task)
        for model in args.models.split(","):
            d = ROOT / "data" / TASK_DIR[task] / model
            if not (d / "train.jsonl").exists():
                print(f"  skip {task}/{model}: no existing splits")
                continue
            seen = existing_prompts(task, model)
            tok = load_tokenizer(str(ROOT / MPATH[model]))
            kw = {"repeats": 4} if task == "induction" else {}
            samples = build_samples(tok, args.n, seed=args.seed, exclude=seen, **kw)
            overlap = sum(1 for s in samples if s.clean_prompt in seen)
            out = d / args.name
            save_samples(samples, out)
            print(f"  {task}/{model}: {len(samples)} -> {out.relative_to(ROOT)} "
                  f"(overlap with train/eval/test: {overlap})", flush=True)


if __name__ == "__main__":
    main()
