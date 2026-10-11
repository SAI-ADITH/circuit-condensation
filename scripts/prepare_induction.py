#!/usr/bin/env python3
"""Generate induction task data (see ilm/data/induction.py for the task).

--repeats 2 plants the ``A B`` bigram twice before the final cue — the STRONG (canonical)
induction variant. The single-exposure variant (repeats=1) proved too weak as a base
behavior at top-1 (GPT-2 0.45, Qwen2.5-1.5B 0.30): with one exposure the model treats the
continuation as guessable filler; with two, copy-what-followed-A dominates and any
induction-capable model should be near ceiling. Splits are disjoint by construction
(later splits exclude all earlier prompts).

Data layout: data/induction/<model>/{train,eval,test}.jsonl -- pass
--out-dir data/induction/<model> (default: data/induction/gpt2).

    python scripts/prepare_induction.py --model models/Qwen2.5-1.5B-Instruct \
        --out-dir data/induction/qwen --n-train 2000 --n-eval 200 --n-test 200 --repeats 2
"""
import argparse
from pathlib import Path

from ilm.data.induction import build_samples, save_samples
from ilm.training.setup import load_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/induction/gpt2"),
                   help="writes <out-dir>/{train,eval,test}.jsonl")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    seen: set[str] = set()
    for split, n, seed in (("train", args.n_train, args.seed),
                           ("eval", args.n_eval, args.seed + 1),
                           ("test", args.n_test, args.seed + 2)):
        samples = build_samples(tok, n, seed=seed, exclude=seen, repeats=args.repeats)
        seen |= {s.clean_prompt for s in samples}
        path = args.out_dir / f"{split}.jsonl"
        save_samples(samples, path)
        print(f"{split}: {len(samples)} -> {path} (example: {samples[0].clean_prompt!r} "
              f"-> {samples[0].answer!r})")


if __name__ == "__main__":
    main()
