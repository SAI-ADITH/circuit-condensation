"""Induction task — the canonical induction-head behavior (Olsson et al. 2022).

A sequence contains a planted bigram ``A B`` early; the same token ``A`` reappears at
the end, and an induction-capable model must predict ``B`` (copy-what-followed-A-last-time).
This is THE textbook mechanistic-interpretability behavior, and induction heads appear in
Wang et al.'s published IOI circuit, giving a second ground-truth-overlap anchor.

clean:     [r1 .. rk]  A B  [s1 .. sm]  A   -> predict B
corrupted: [r1 .. rk]  A C  [s1 .. sm]  A   -> (would induce C != B)

The corrupted prompt is token-aligned with the clean one (only the early ``B`` slot
differs, swapped for a different single-token word ``C``), so EAP clean/corrupted patching
is well defined and the answer-determining edge is the induction path. Single-token answer,
so the IOI/docstring metrics (ioi_accuracy / circuit_token_accuracy) apply unchanged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import Dataset

from .dataset import bos_prefix

# Common, mostly-single-token English words; filtered per tokenizer at build time.
WORD_POOL = [
    "cat", "dog", "house", "tree", "book", "car", "road", "river", "stone", "cloud",
    "music", "paper", "table", "chair", "light", "water", "fire", "earth", "wind", "metal",
    "green", "blue", "red", "black", "white", "fast", "slow", "warm", "cold", "bright",
    "king", "queen", "city", "field", "horse", "bird", "fish", "snow", "rain", "star",
    "glass", "wood", "iron", "gold", "silver", "salt", "bread", "milk", "honey", "wine",
]

PREFIX_LEN = 6   # random tokens before the planted bigram
MIDDLE_LEN = 6   # random tokens between the bigram and the final cue token A


@dataclass
class InductionSample:
    clean_prompt: str       # "... A B ... A"
    corrupted_prompt: str   # "... A C ... A" (token-aligned)
    answer: str             # " B" (leading space, single token)


def single_token(tokenizer, word: str) -> bool:
    return len(tokenizer(" " + word, add_special_tokens=False).input_ids) == 1


def _filter(tokenizer, pool):
    return [w for w in pool if single_token(tokenizer, w)]


def _render(words: list[str]) -> str:
    """Join with leading spaces so every word is one token; prompt ends right after the
    final cue word A (the model then predicts the word that followed A earlier)."""
    return " " + " ".join(words)


def build_samples(tokenizer, n: int, seed: int = 0,
                  exclude: set[str] | None = None, repeats: int = 1) -> list[InductionSample]:
    """``repeats`` = how many times the planted bigram ``A B`` appears before the final cue.
    repeats=1 is the weak single-exposure variant (GPT-2 0.45, Qwen2.5-1.5B 0.30 top-1 —
    too weak a base behavior to condense). repeats>=2 is the canonical STRONG induction
    (Olsson et al. measure on repeated patterns): each extra exposure reinforces the same
    A->B pattern, so an induction-capable model should be near-ceiling at top-1."""
    rng = random.Random(seed)
    pool = _filter(tokenizer, WORD_POOL)
    if len(pool) < 8:
        raise RuntimeError(f"Need >= 8 single-token words, got {len(pool)}.")
    seen: set[str] = set(exclude or ())
    out: list[InductionSample] = []
    attempts, max_attempts = 0, 400 * n + 2000
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could only build {len(out)}/{n} induction samples.")
        A, B, C = rng.sample(pool, 3)            # cue, target, corrupt-target (all distinct)
        # filler must EXCLUDE A/B/C so the cue A appears exactly (repeats+1) times, always
        # followed by B (or by the final prediction slot) — a stray A in the filler would
        # plant a second, different A->? bigram and a real induction head can't tell which
        # continuation to copy.
        filler_pool = [w for w in pool if w not in (A, B, C)]
        prefix = [rng.choice(filler_pool) for _ in range(PREFIX_LEN)]
        clean_words, corr_words = list(prefix), list(prefix)
        for _ in range(max(1, repeats)):
            middle = [rng.choice(filler_pool) for _ in range(MIDDLE_LEN)]
            clean_words += [A, B] + middle
            corr_words += [A, C] + middle
        clean_words.append(A)
        corr_words.append(A)
        clean = _render(clean_words)
        if clean in seen:
            continue
        seen.add(clean)
        out.append(InductionSample(clean, _render(corr_words), " " + B))
    return out


def save_samples(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path):
    with open(path) as f:
        return [InductionSample(**json.loads(line)) for line in f if line.strip()]


class InductionDataset(Dataset):
    """Emits the IOI/PairDataset batch shape with a single-token answer, so the exact-
    top-token IOI metrics (ioi_accuracy / circuit_token_accuracy) apply unchanged."""

    def __init__(self, samples, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Candidate answer set = the single-token forms of the whole word pool. The induction
        # metric restricts the argmax to THIS set (induction_pool_accuracy): the model is
        # correct iff, among plausible copy targets, the gold word B is the top one -- so a
        # generic non-word continuation (" the", punctuation) no longer counts as a miss.
        self.pool_ids = sorted({tokenizer(" " + w, add_special_tokens=False).input_ids[0]
                                for w in WORD_POOL if single_token(tokenizer, w)})
        self.examples = [self._encode(s, tokenizer) for s in samples]

    def _encode(self, s, tok) -> dict:
        bos = bos_prefix(tok)
        prompt_ids = bos + tok(s.clean_prompt, add_special_tokens=False).input_ids
        ans_ids = tok(s.answer, add_special_tokens=False).input_ids   # one token
        clean_ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        corr_ids = (bos + tok(s.corrupted_prompt, add_special_tokens=False).input_ids)[: self.max_length]
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": s.answer,
            "answer_id": ans_ids[0],
            "pool_ids": self.pool_ids,
            "prompt": s.clean_prompt,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
