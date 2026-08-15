#!/usr/bin/env python3
"""Generate IOI clean/corrupted samples (GPT-2 single-token names) to JSONL.

Splits are disjoint BY CONSTRUCTION: each later split is built excluding every
clean_prompt already used by an earlier one (mirrors scripts/prepare_induction.py).

Data layout (2026-07-03): data/ioi/<model>/{train,eval,test}.jsonl -- pass
--out-dir data/ioi/<model> (default: data/ioi/gpt2).

    python scripts/prepare_ioi.py --model-path models/gpt2 \
        --n-train 2000 --n-eval 200 --n-test 200 --out-dir data/ioi/gpt2
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from ilm.data.ioi import build_samples, save_samples, single_token_names, NAME_POOL, TEMPLATES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/gpt2",
                   help="Tokenizer to filter single-token names against.")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=Path("data/ioi/gpt2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--structure", default="", help="abba or baba to force first-clause order (structural transfer variant); empty=mix.")
    p.add_argument("--templates", default="",
                   help="comma list of TEMPLATES indices to restrict to (e.g. '0,1,2' for "
                        "family A). Empty = full 5-frame mix. Used for template-transfer families.")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    names = single_token_names(tok, NAME_POOL)
    print(f"{len(names)}/{len(NAME_POOL)} names are single GPT-2 tokens: {names}")
    templates = ([TEMPLATES[int(i)] for i in args.templates.split(",")]
                 if args.templates else None)
    if templates:
        print(f"restricting to {len(templates)} template frame(s): indices {args.templates}")

    seen: set[str] = set()
    for split, n, seed in (("train", args.n_train, args.seed),
                           ("eval", args.n_eval, args.seed + 1),
                           ("test", args.n_test, args.seed + 2)):
        samples = build_samples(tok, n, seed=seed, exclude=seen, templates=templates,
                                structure=(args.structure or None))
        overlap = sum(s.clean_prompt in seen for s in samples)
        seen |= {s.clean_prompt for s in samples}
        path = args.out_dir / f"{split}.jsonl"
        save_samples(samples, path)
        print(f"{split}: {len(samples)} -> {path} (overlap with prior splits: {overlap}, "
              f"should be 0; example: {samples[0].clean_prompt!r} -> {samples[0].io_name!r})")


if __name__ == "__main__":
    main()
