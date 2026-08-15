#!/usr/bin/env python3
"""Generate subject-verb agreement clean/corrupted samples to JSONL.

Draws a MIX of three constructions per example (see ilm/data/agreement.py for the
literature grounding): "pp1" (one PP attractor, Linzen 2016), "pp2" (two chained PP
attractors, Linzen's attractor-count axis), "rc" (subject-relative-clause attractor,
Marvin & Linzen 2018 / Lakretz 2019). Splits are disjoint BY CONSTRUCTION: each later
split is built excluding every clean_prompt already used by an earlier one (mirrors
scripts/prepare_induction.py).

Data layout (2026-07-03): data/agreement/<model>/{train,eval,test}.jsonl -- pass
--out-dir data/agreement/<model> (default: data/agreement/gpt2).

    python scripts/prepare_agreement.py --model-path models/gpt2 \
        --n-train 2000 --n-eval 200 --n-test 200 --out-dir data/agreement/gpt2
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from ilm.data.agreement import FRAMES, build_samples, save_samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/gpt2")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=Path("data/agreement/gpt2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", default=",".join(FRAMES),
                   help="comma list of frames to mix (subset of pp1,pp2,rc).")
    args = p.parse_args()

    frames = tuple(args.frames.split(","))
    tok = AutoTokenizer.from_pretrained(args.model_path)
    seen: set[str] = set()
    for split, n, seed in (("train", args.n_train, args.seed),
                           ("eval", args.n_eval, args.seed + 1),
                           ("test", args.n_test, args.seed + 2)):
        samples = build_samples(tok, n, seed=seed, exclude=seen, frames=frames)
        overlap = sum(s.clean_prompt in seen for s in samples)
        seen |= {s.clean_prompt for s in samples}
        path = args.out_dir / f"{split}.jsonl"
        save_samples(samples, path)
        from collections import Counter
        counts = dict(Counter(s.frame for s in samples))
        print(f"{split}: {len(samples)} -> {path} (frame mix: {counts}, "
              f"overlap with prior splits: {overlap}, should be 0)")
        for fr in frames:
            ex = next((s for s in samples if s.frame == fr), None)
            if ex is not None:
                print(f"  example [{fr}]: {ex.clean_prompt!r} -> {ex.answer!r}  |  corr: {ex.corrupted_prompt!r}")


if __name__ == "__main__":
    main()
