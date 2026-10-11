"""IOI (Indirect Object Identification) clean/corrupted data — the GPT-2 rig.

IOI is the task with a *published ground-truth circuit* (Wang 2023), so it is our
honesty anchor for the localization machinery. A sentence like

    "When Mary and John went to the store, John gave a drink to"   -> " Mary"

names two people; the **indirect object** (IO, here "Mary") is the name that
appears once, the **subject** (S, "John") the one that repeats and performs the
action. The model must predict the IO.

Two clean structures (as in Wang 2023 / Edge Pruning's multi-template IOI):
    ABBA: "When [IO] and [S] ... [S] gave ... to"   -> IO
    BABA: "When [S] and [IO] ... [S] gave ... to"   -> IO

**Corruption (the "ABC" scheme):** replace the acting subject (the repeated name)
with a random third name C, breaking the duplicate-name cue the IOI heads rely on
— the activation that OFF edges read. C is a single token, so the corrupted prompt
is token-aligned with the clean one position-for-position.

Names are filtered to those that are a *single* GPT-2 token (with a leading
space), so the answer is one prediction step.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import bos_prefix

# Templates: "{n1}" and "{n2}" are the first-clause pair; "{actor}" performs the
# action (always the subject S). The prompt ends right before the IO answer.
TEMPLATES = [
    "When {n1} and {n2} went to the store, {actor} gave a drink to",
    "Then {n1} and {n2} went to the park, and {actor} gave the ball to",
    "After {n1} and {n2} left the school, {actor} handed the book to",
    "While {n1} and {n2} were at the office, {actor} passed the note to",
    "One day {n1} and {n2} went shopping, and {actor} sold a bag to",
]

# Candidate single-token first names (filtered against the real tokenizer at build).
NAME_POOL = [
    "John", "Mary", "Tom", "James", "Dan", "Paul", "Mark", "Anna", "Kate", "Sarah",
    "Mike", "Steve", "Laura", "Emma", "Jack", "Alice", "Henry", "David", "Peter",
    "Sam", "Grace", "Lucy", "Frank", "Carl", "Nick", "Rose", "Adam", "Eric", "Luke",
]


@dataclass
class IOISample:
    clean_prompt: str       # ends right before the answer; next token should be the IO
    corrupted_prompt: str   # ABC-corrupted, token-aligned with the clean prompt
    io_name: str            # the answer, with leading space (single token)
    s_name: str             # the subject (repeated) name


def single_token_names(tokenizer, pool: list[str]) -> list[str]:
    """Keep only names that encode to exactly one token with a leading space."""
    ok = []
    for n in pool:
        ids = tokenizer(" " + n, add_special_tokens=False).input_ids
        if len(ids) == 1:
            ok.append(n)
    return ok


def build_samples(tokenizer, n: int, seed: int = 0,
                  exclude: set[str] | None = None,
                  templates: list[str] | None = None,
                  structure: str | None = None) -> list[IOISample]:
    """Generate ``n`` IOI samples. ``exclude`` is a set of ``clean_prompt`` strings
    to avoid (e.g. the train split), so an eval set can be guaranteed *string-disjoint*
    from train rather than merely independently resampled. Within a build the clean
    prompts are also kept unique. The clean-prompt space (~6.5k for this pool) far
    exceeds typical n, so the rejection loop terminates quickly.

    ``templates`` restricts generation to a subset of ``TEMPLATES`` (the lexical sentence
    frames), used to build disjoint template FAMILIES for the transfer experiment
    Default ``None`` = the full 5-frame mix.

    ``structure`` forces the first-clause order: ``"abba"`` = IO first ([IO] and [S] ... [S]),
    ``"baba"`` = S first ([S] and [IO] ... [S]). Default ``None`` = 50/50 mix. Used to build the
    STRUCTURAL transfer variants (the DCD-style dataset-specificity test), which is where the
    circuit actually differs across datasets, unlike the mild lexical-frame split."""
    if structure not in (None, "abba", "baba"):
        raise ValueError(f"structure must be None/'abba'/'baba', got {structure!r}")
    rng = random.Random(seed)
    pool = templates if templates is not None else TEMPLATES
    names = single_token_names(tokenizer, NAME_POOL)
    if len(names) < 3:
        raise RuntimeError(f"Need >=3 single-token names, got {len(names)}: {names}")
    seen: set[str] = set(exclude or ())
    samples: list[IOISample] = []
    attempts = 0
    max_attempts = 200 * n + 1000
    while len(samples) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not draw {n} unique clean prompts (got {len(samples)}); "
                               "the template/name space may be too small for n + exclude.")
        io, s, c = rng.sample(names, 3)          # IO, subject, corrupting third name
        template = rng.choice(pool)
        abba = (rng.random() < 0.5) if structure is None else (structure == "abba")
        n1, n2 = (io, s) if abba else (s, io)    # first-clause order
        clean = template.format(n1=n1, n2=n2, actor=s)
        if clean in seen:
            continue
        seen.add(clean)
        corrupted = template.format(n1=n1, n2=n2, actor=c)   # S -> C only
        samples.append(IOISample(clean, corrupted, " " + io, " " + s))
    return samples


def save_samples(samples: list[IOISample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path: str | Path) -> list[IOISample]:
    with open(path) as f:
        return [IOISample(**json.loads(line)) for line in f if line.strip()]


class IOIDataset(Dataset):
    """Tokenises IOI samples into the same batch dict shape as ``PairDataset``,
    so the existing collator / trainer / eval work unchanged. The training target
    is the single IO-name token appended to the clean prompt.
    """

    def __init__(self, samples: list[IOISample], tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._encode(s, tokenizer) for s in samples]

    def _encode(self, s: IOISample, tok) -> dict:
        bos = bos_prefix(tok)
        prompt_ids = bos + tok(s.clean_prompt, add_special_tokens=False).input_ids
        ans_ids = tok(s.io_name, add_special_tokens=False).input_ids   # one token
        clean_ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        corr_ids = (bos + tok(s.corrupted_prompt, add_special_tokens=False).input_ids)[: self.max_length]
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": s.io_name,
            # Forced-choice ids for the canonical Wang IO-vs-S logit-difference metric
            # (ioi_forced_choice_accuracy): the model is correct iff it puts more mass on the
            # indirect object than on the subject, NOT iff the IO is argmax over the whole
            # vocab (which penalised models that prefer a non-name top token -- qwen3 scored
            # 0.69 by argmax but 1.00 by forced choice).
            "answer_id": ans_ids[0],
            "s_id": tok(s.s_name, add_special_tokens=False).input_ids[0],
            # Unified foil for the CONTRASTIVE attribution metric (make_logit_diff_metric):
            # IOI's foil is the subject S, so attribution targets logit(IO)-logit(S) -- the
            # same Wang IO-vs-S decision the eval scores, not generic "boost IO".
            "foil_id": tok(s.s_name, add_special_tokens=False).input_ids[0],
            "prompt": s.clean_prompt,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> dict:
        return self.examples[idx]
