#!/usr/bin/env python3
"""Generate docstring clean/corrupted samples (GPT-2 single-token names) to JSONL.

Splits are disjoint BY CONSTRUCTION: each later split is built excluding every
clean_prompt already used by an earlier one (mirrors scripts/prepare_induction.py).

Data layout: data/docstring/<model>/{train,eval,test}.jsonl -- pass
--out-dir data/docstring/<model> (default: data/docstring/gpt2).

    python scripts/prepare_docstring.py --model-path models/gpt2 \
        --n-train 2000 --n-eval 200 --n-test 200 --out-dir data/docstring/gpt2
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from ilm.data.docstring import build_samples, save_samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/gpt2")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=Path("data/docstring/gpt2"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    seen: set[str] = set()
    for split, n, seed in (("train", args.n_train, args.seed),
                           ("eval", args.n_eval, args.seed + 1),
                           ("test", args.n_test, args.seed + 2)):
        samples = build_samples(tok, n, seed=seed, exclude=seen)
        overlap = sum(s.clean_prompt in seen for s in samples)
        seen |= {s.clean_prompt for s in samples}
        path = args.out_dir / f"{split}.jsonl"
        save_samples(samples, path)
        print(f"{split}: {len(samples)} -> {path} (overlap with prior splits: {overlap}, "
              f"should be 0; example: {samples[0].clean_prompt!r} -> {samples[0].answer!r})")


if __name__ == "__main__":
    main()
