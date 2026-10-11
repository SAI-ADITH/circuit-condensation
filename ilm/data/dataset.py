"""Tokenised dataset + collator for clean/corrupted GSM8K pairs.

The disentangled residual stream blends the clean and corrupted contribution at
each position, so clean and corrupted sequences in a batch are padded to a single
common length. ``labels`` mark the clean answer span; training no longer uses them
(the loss is KL-to-original) -- they are kept only for the eval metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .gsm8k import Pair


def bos_prefix(tok) -> list[int]:
    """The tokenizer's BOS id as a 1-element list (empty if it has none). All task encoders
    build ids with add_special_tokens=False (for exact clean/corrupted token alignment and
    single-token answer control), which drops the BOS that instruct models (Llama/Gemma)
    expect -- evaluating them out-of-distribution. Prepending BOS restores in-distribution
    inputs: it fixes Gemma's IOI (0.19->0.72 argmax, 0.43->1.00 forced-choice) and lifts
    Llama induction (0.80->0.93), and is a no-op for gpt2/Qwen (no BOS id). Applied to BOTH
    clean and corrupted ids so their token alignment (only the intended slots differ) is
    preserved -- both just shift by one leading BOS."""
    b = getattr(tok, "bos_token_id", None)
    return [b] if b is not None else []


PROMPT_TMPL = (
    "Question: {q}\n"
    'Solve step by step, then give the final integer answer after "####".\n'
    "Answer:"
)
TARGET_TMPL = " {sol}"           # full chain-of-thought solution ending in "#### N"
FALLBACK_TMPL = " #### {a}"      # used only if a pair has no stored solution text


def save_pairs(pairs: list[Pair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")


def load_pairs(path: str | Path) -> list[Pair]:
    with open(path) as f:
        return [Pair(**json.loads(line)) for line in f if line.strip()]


class PairDataset(Dataset):
    """Tokenises each pair into clean (with answer labels) and corrupted ids."""

    def __init__(self, pairs: list[Pair], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._encode(p) for p in pairs]

    def _encode(self, p: Pair) -> dict:
        tok = self.tokenizer
        clean_prompt = PROMPT_TMPL.format(q=p.clean_question)
        corr_prompt = PROMPT_TMPL.format(q=p.corrupted_question)
        target = (TARGET_TMPL.format(sol=p.clean_solution) if p.clean_solution
                  else FALLBACK_TMPL.format(a=p.clean_answer))

        prompt_ids = tok(clean_prompt, add_special_tokens=True).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids
        clean_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]

        corr_ids = tok(corr_prompt, add_special_tokens=True).input_ids[: self.max_length]
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": p.clean_answer,
            "prompt": clean_prompt,      # for generative eval
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> dict:
        return self.examples[idx]


class PairCollator:
    """Pads clean and corrupted to one shared length so contributions align."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(
            max(len(b["clean_ids"]), len(b["corr_ids"])) for b in batch
        )

        def pad(seq, fill):
            return seq + [fill] * (max_len - len(seq))

        clean_ids, clean_mask, labels, corr_ids, corr_mask, answers, foil_ids = [], [], [], [], [], [], []
        for b in batch:
            n_c, n_x = len(b["clean_ids"]), len(b["corr_ids"])
            clean_ids.append(pad(b["clean_ids"], self.pad_token_id))
            clean_mask.append([1] * n_c + [0] * (max_len - n_c))
            labels.append(pad(b["labels"], -100))
            corr_ids.append(pad(b["corr_ids"], self.pad_token_id))
            corr_mask.append([1] * n_x + [0] * (max_len - n_x))
            answers.append(b["clean_answer"])
            # Per-example foil token for the CONTRASTIVE attribution metric (IOI: subject S;
            # agreement: wrong-number verb). -1 == no foil recorded -> metric falls back to
            # plain logp(answer). Tasks without a single-pair foil (induction/doc/gsm8k) omit
            # the key and land on -1. Kept order-aligned with the batch (no reordering).
            foil_ids.append(int(b.get("foil_id", -1)))

        t = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
        return {
            "clean_ids": t(clean_ids),
            "clean_mask": t(clean_mask),
            "labels": t(labels),
            "corr_ids": t(corr_ids),
            "corr_mask": t(corr_mask),
            "answers": answers,
            "foil_ids": t(foil_ids),
        }
