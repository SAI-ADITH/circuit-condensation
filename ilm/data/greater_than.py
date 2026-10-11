"""Greater-than clean/corrupted data — the second atomic GPT-2 rig (Hanna 2023).

The **greater-than** task is the cleanest published companion to IOI: a single,
atomic behavior with a known circuit in GPT-2 small. A prompt like

    "The war lasted from the year 1742 to the year 17"   -> a 2-digit number > 42

names a START year (1742) and ends mid-way through the END year ("...year 17"),
so the model must complete the end year's last two digits with a number STRICTLY
GREATER than 42. The behavior = "respect the > constraint."

**Why it fits our rig:** the answer is a *single token* (a 2-digit number like
"98"), so — exactly as for IOI — teacher-forced argmax == real accuracy, no
free-generation gap, no compounding error. That makes it the ideal second
"atomic" rung in the behavior-breadth ladder (see the 2026-06-27 explore doc).

**Corruption (Hanna's "01" scheme):** replace the start year's last two digits
with "01", i.e. "...from the year 1701 to the year 17". This makes almost every
2-digit completion valid, destroying the > cue the greater-than heads rely on —
the activation the OFF edges read. The corrupted prompt is token-aligned with the
clean one (only the YY token differs), so positions match for the disentangled
forward.

**Accuracy metric (Hanna's argmax-valid):** the answer is not one fixed token but
a *set* — any YY' > start_yy is correct. So accuracy = "is the model's top
predicted 2-digit token a year strictly greater than the start year?" Each example
therefore carries its own ``valid_ids`` (the token ids of all valid completions).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import Dataset

# Events whose duration spans years — the noun in "The {noun} lasted from ...".
NOUN_POOL = [
    "war", "famine", "drought", "feud", "siege", "reign", "voyage", "plague",
    "boom", "trial", "tour", "study", "project", "march", "boycott", "revolt",
    "blockade", "uprising", "conflict", "epidemic",
]

# Two-clause template: a START year (century+YY) and an END year cut off after the
# century, so the model predicts the end YY under the > constraint.
TEMPLATE = "The {noun} lasted from the year {start} to the year {century}"

CORRUPT_YY = 1   # Hanna's corruption: start year's YY -> "01"


@dataclass
class GreaterThanSample:
    clean_prompt: str       # "...from the year 1742 to the year 17"
    corrupted_prompt: str   # "...from the year 1701 to the year 17" (token-aligned)
    century: int            # e.g. 17 (the shared century of both years)
    start_yy: int           # the threshold YY; valid completions are > this


def two_digit_token_id(tokenizer, yy: int) -> int | None:
    """Token id for the 2-digit string ``f"{yy:02d}"`` IFF it is a single token
    (no leading space — it directly follows the century digits). Else None."""
    ids = tokenizer(f"{yy:02d}", add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def _valid_ids(tokenizer, start_yy: int) -> list[int]:
    """Token ids of every single-token YY' with start_yy < YY' <= 99."""
    out = []
    for yy in range(start_yy + 1, 100):
        tid = two_digit_token_id(tokenizer, yy)
        if tid is not None:
            out.append(tid)
    return out


def build_samples(tokenizer, n: int, seed: int = 0,
                  centuries: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18),
                  min_valid: int = 8,
                  exclude: set[str] | None = None) -> list[GreaterThanSample]:
    """Generate ``n`` greater-than samples. ``start_yy`` is drawn so there are at
    least ``min_valid`` single-token completions above it (a non-trivial > test),
    and so the clean and corrupted prompts stay TOKEN-ALIGNED (same length) — both
    requirements are checked against the real tokenizer. ``exclude`` is a set of
    ``clean_prompt`` strings to avoid (so eval is string-disjoint from train)."""
    rng = random.Random(seed)
    seen: set[str] = set(exclude or ())
    samples: list[GreaterThanSample] = []
    attempts = 0
    max_attempts = 400 * n + 2000
    while len(samples) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could only build {len(samples)}/{n} greater-than samples; "
                               "loosen min_valid or widen centuries.")
        century = rng.choice(centuries)
        # Keep room above (need >= min_valid valid completions) and below (so the
        # constraint is real, not "> 02"); 12..(98-min_valid) is a safe window.
        start_yy = rng.randint(12, 98 - min_valid)
        if two_digit_token_id(tokenizer, start_yy) is None:
            continue
        if two_digit_token_id(tokenizer, CORRUPT_YY) is None:
            continue
        if len(_valid_ids(tokenizer, start_yy)) < min_valid:
            continue
        clean = TEMPLATE.format(noun=rng.choice(NOUN_POOL),
                                start=f"{century}{start_yy:02d}", century=century)
        if clean in seen:
            continue
        corrupted = clean.replace(f"{century}{start_yy:02d}",
                                  f"{century}{CORRUPT_YY:02d}", 1)
        # Hard alignment requirement (2026-07-03, tightened): clean and corrupted must
        # tokenize to the same length AND differ at EXACTLY ONE token position -- not just
        # "same total length", which a BPE tokenizer can satisfy while splitting the
        # century/YY boundary differently than expected (confirmed empirically misaligned
        # on Llama/Qwen: same-length pairs still differing at 2 positions). The
        # disentangled forward needs a genuine single-token counterfactual.
        clean_ids = tokenizer(clean, add_special_tokens=False).input_ids
        corrupted_ids = tokenizer(corrupted, add_special_tokens=False).input_ids
        if len(clean_ids) != len(corrupted_ids):
            continue
        if sum(1 for a, b in zip(clean_ids, corrupted_ids) if a != b) != 1:
            continue
        seen.add(clean)
        samples.append(GreaterThanSample(clean, corrupted, century, start_yy))
    return samples


def save_samples(samples: list[GreaterThanSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path: str | Path) -> list[GreaterThanSample]:
    with open(path) as f:
        return [GreaterThanSample(**json.loads(line)) for line in f if line.strip()]


class GreaterThanDataset(Dataset):
    """Tokenises greater-than samples into the IOI/PairDataset batch shape. The
    teacher-forcing answer slot uses a *representative* valid year (so labels mark
    one supervised position); correctness is judged by ``valid_ids`` set-membership
    in ``greater_than_accuracy``, not exact match. ``valid_ids`` is kept on each
    example for that metric (the collator ignores extra keys)."""

    def __init__(self, samples: list[GreaterThanSample], tokenizer, max_length: int = 64,
                 generative: bool = False):
        # generative=True: tokenizer-robust path — cut the prompt at "...to the year "
        #   and have the model GENERATE the full end year (parsed + compared > start in
        #   ``greater_than_generative_accuracy``). Works on Llama/Qwen, which do NOT pack
        #   a 2-digit year into one token. generative=False: the original GPT-2 single-
        #   token argmax-valid path (Hanna), guarded by an assertion below.
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generative = generative
        self.examples = [self._encode(s, tokenizer) for s in samples]

    def _encode(self, s: GreaterThanSample, tok) -> dict:
        if self.generative:
            return self._encode_generative(s, tok)
        return self._encode_token(s, tok)

    def _encode_token(self, s: GreaterThanSample, tok) -> dict:
        prompt_ids = tok(s.clean_prompt, add_special_tokens=False).input_ids
        # Representative answer token = the smallest valid YY (start+1) if single-token,
        # else the next single-token YY above it. Used only to mark the answer slot.
        rep_yy = next((yy for yy in range(s.start_yy + 1, 100)
                       if two_digit_token_id(tok, yy) is not None), None)
        if rep_yy is None:
            raise ValueError(
                "GreaterThanDataset(generative=False) needs single-token 2-digit answers, "
                f"but tokenizer {tok.name_or_path!r} has none above start_yy={s.start_yy}. "
                "This tokenizer splits 2-digit numbers (e.g. Llama/Qwen) — use "
                "generative=True (the tokenizer-robust generated-year metric).")
        ans_ids = [two_digit_token_id(tok, rep_yy)]
        clean_ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        corr_ids = tok(s.corrupted_prompt, add_special_tokens=False).input_ids[: self.max_length]
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": f"{rep_yy:02d}",
            "prompt": s.clean_prompt,
            "valid_ids": _valid_ids(tok, s.start_yy),   # set of correct token ids
            "start_yy": s.start_yy,
            "century": s.century,
        }

    def _encode_generative(self, s: GreaterThanSample, tok) -> dict:
        # Cut the prompt at "...to the year " (drop the dangling century digits) so the
        # model writes the WHOLE end year — robust to how the tokenizer chunks numbers.
        year_prompt = re.sub(r"\d+\s*$", "", s.clean_prompt)
        corr_prompt = re.sub(r"\d+\s*$", "", s.corrupted_prompt)
        prompt_ids = tok(year_prompt, add_special_tokens=False).input_ids
        corr_ids = tok(corr_prompt, add_special_tokens=False).input_ids[: self.max_length]
        # Representative full end year (smallest valid) — only marks the answer slot;
        # correctness is judged generatively, not by these tokens.
        rep_year = s.century * 100 + (s.start_yy + 1)
        ans_ids = tok(str(rep_year), add_special_tokens=False).input_ids
        clean_ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": str(rep_year),
            "prompt": year_prompt,
            "start_yy": s.start_yy,
            "century": s.century,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> dict:
        return self.examples[idx]
