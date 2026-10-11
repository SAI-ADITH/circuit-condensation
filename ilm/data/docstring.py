"""Docstring clean/corrupted data — a candidate third atomic GPT-2 rig.

The **docstring** task (Heimersheim & Janiak 2023) is an in-order *copy* behavior:
given a Python function whose docstring documents its parameters in signature
order, predict the NEXT parameter name. A prompt like

    def process(load, size, files, mode):
        \"\"\"Run the job.

        :param load: the input.
        :param size: the value.
        :param

ends right after the third ``:param`` and the model must emit ``files`` — the
third parameter in the SIGNATURE, copied in order. There is a single correct
answer token, so this reuses IOI's exact-top-token accuracy path unchanged.

**Corruption:** replace the signature's parameter names with a different set of
single-token names (token-aligned), so the correct copy target changes — breaking
the name-copying cue the docstring heads rely on (the activation OFF edges read).

Parameter and filler names are filtered to single GPT-2 tokens (with a leading
space) so the answer is one prediction step — the IOI story, no generation gap.

NOTE: unlike IOI/greater-than this has NO published circuit on GPT-2 small (the
paper used a 4-layer attn-only model), so KL-faithfulness is the only honesty
anchor here. It is also a *code* behavior, so GPT-2 small may be near chance —
run ``scripts/docstring_ceiling.py`` first to confirm the capability anchor.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import Dataset

from .dataset import bos_prefix

# Candidate single-token parameter names (variable-like; filtered at build time).
PARAM_POOL = [
    "load", "size", "files", "mode", "data", "value", "count", "index", "name",
    "item", "node", "key", "val", "text", "flag", "path", "temp", "num", "rate",
    "type", "code", "step", "time", "user", "task", "line", "page", "byte", "port",
]
# Filler nouns for the parameter descriptions (single token, leading space).
DESC_POOL = ["input", "result", "output", "number", "string", "object", "value", "list", "buffer"]
FN_POOL = ["process", "handle", "build", "update", "render", "parse", "create", "compute"]
N_PARAMS = 4        # signature length
N_DOCUMENTED = 2    # params already written in the docstring; predict param #N_DOCUMENTED


@dataclass
class DocstringSample:
    clean_prompt: str       # ends right after the (N_DOCUMENTED+1)-th ":param"
    corrupted_prompt: str   # renamed params, token-aligned
    answer: str             # the next param name (signature order), leading space, single token


def single_token(tokenizer, word: str) -> bool:
    return len(tokenizer(" " + word, add_special_tokens=False).input_ids) == 1


def _filter(tokenizer, pool):
    return [w for w in pool if single_token(tokenizer, w)]


def _render(fn: str, params: list[str], descs: list[str]) -> str:
    """Build the prompt that ends right after the target ':param ' (the model must
    then emit params[N_DOCUMENTED])."""
    sig = f"def {fn}({', '.join(params)}):"
    lines = [sig, '    """Run the job.', ""]
    for i in range(N_DOCUMENTED):
        lines.append(f"    :param {params[i]}: the {descs[i]}.")
    lines.append("    :param")          # next param name predicted right here
    return "\n".join(lines)


def build_samples(tokenizer, n: int, seed: int = 0,
                  exclude: set[str] | None = None) -> list[DocstringSample]:
    rng = random.Random(seed)
    params = _filter(tokenizer, PARAM_POOL)
    descs = _filter(tokenizer, DESC_POOL)
    fns = _filter(tokenizer, FN_POOL)
    if len(params) < 2 * N_PARAMS:
        raise RuntimeError(f"Need >= {2*N_PARAMS} single-token params, got {len(params)}.")
    seen: set[str] = set(exclude or ())
    out: list[DocstringSample] = []
    attempts, max_attempts = 0, 400 * n + 2000
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could only build {len(out)}/{n} docstring samples.")
        clean_params = rng.sample(params, N_PARAMS)
        corr_params = rng.sample([p for p in params if p not in clean_params], N_PARAMS)
        d = [rng.choice(descs) for _ in range(N_DOCUMENTED)]
        fn = rng.choice(fns)
        clean = _render(fn, clean_params, d)
        if clean in seen:
            continue
        corrupted = _render(fn, corr_params, d)
        # token-aligned check (each name single token, same structure -> same length)
        if (len(tokenizer(clean, add_special_tokens=False).input_ids)
                != len(tokenizer(corrupted, add_special_tokens=False).input_ids)):
            continue
        seen.add(clean)
        out.append(DocstringSample(clean, corrupted, " " + clean_params[N_DOCUMENTED]))
    return out


def save_samples(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path):
    with open(path) as f:
        return [DocstringSample(**json.loads(line)) for line in f if line.strip()]


class DocstringDataset(Dataset):
    """Emits the IOI/PairDataset batch shape with a single-token answer, so the
    exact-top-token IOI metrics (ioi_accuracy / circuit_token_accuracy) apply."""

    def __init__(self, samples, tokenizer, max_length: int = 96):
        self.tokenizer = tokenizer
        self.max_length = max_length
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
            "prompt": s.clean_prompt,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
