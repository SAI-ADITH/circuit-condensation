"""GSM8K clean/corrupted pair construction with answer-flip verification.

Edge Pruning needs, per problem, a *clean* version and a minimally *corrupted*
twin whose answer differs (the corrupted activations are what OFF edges read).

Corruption strategy (doc 5 §1): change one integer in the question and recompute
the gold answer by re-evaluating the solution's ``<<a op b = c>>`` calculator
annotations with the substituted value propagated through the dependency chain.
A pair is kept only if the corrupted answer is a valid integer that *differs*
from the clean answer; otherwise it is discarded. This guarantees a genuine
answer flip rather than relying on the model's own output.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

ANS_RE = re.compile(r"####\s*(-?[\d,]+)")
CALC_RE = re.compile(r"<<([^>]+)>>")
EXPR_RE = re.compile(r"^\s*([\d.\s+\-*/()]+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class Pair:
    clean_question: str
    corrupted_question: str
    clean_answer: int
    corrupted_answer: int
    clean_solution: str = ""  # full reasoning + "#### N", calc annotations stripped


def extract_answer(answer_field: str) -> int | None:
    m = ANS_RE.search(answer_field)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _safe_eval(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression with no names/calls."""
    if not re.fullmatch(r"[\d.\s+\-*/()]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - expr is arithmetic-only
    except (SyntaxError, ZeroDivisionError, ValueError):
        return None


def recompute_answer(answer_field: str, old: int, new: int) -> int | None:
    """Re-evaluate the calculator steps after substituting ``old`` -> ``new``.

    The substitution is applied to the first calc step whose expression contains
    ``old`` as a standalone integer; each step's recomputed result then replaces
    that step's old result in later steps, so the change propagates. Returns the
    new final answer, or None if it cannot be done unambiguously.
    """
    steps = CALC_RE.findall(answer_field)
    if not steps:
        return None

    parsed: list[tuple[str, float]] = []
    for step in steps:
        m = EXPR_RE.match(step)
        if not m:
            return None
        parsed.append((m.group(1), float(m.group(2))))

    substituted = False
    value_map: dict[float, float] = {}  # old result value -> new result value
    final = None
    for expr, old_result in parsed:
        new_expr = expr
        # 1. Substitute the changed input wherever it appears as a literal: the
        #    same input quantity can be referenced by several calc steps.
        pat = rf"(?<!\d){old}(?!\d)"
        if re.search(pat, new_expr):
            new_expr = re.sub(pat, str(new), new_expr)
            substituted = True
        # 2. Propagate earlier recomputed results into this expression.
        for ov, nv in value_map.items():
            new_expr = re.sub(rf"(?<!\d){_num(ov)}(?!\d)", _num(nv), new_expr)
        val = _safe_eval(new_expr)
        if val is None:
            return None
        if val != old_result:
            value_map[old_result] = val
        final = val

    if not substituted or final is None or final != int(final):
        return None
    return int(final)


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def clean_solution_text(answer_field: str) -> str:
    """The GSM8K solution with ``<<calc>>`` annotations stripped.

    This is the supervised target: full chain-of-thought reasoning ending in
    ``#### N``. Training on it (rather than the bare answer) keeps the model
    able to actually solve the problem, so there is real accuracy to trade off
    against circuit sparsity on the knee curve.
    """
    text = CALC_RE.sub("", answer_field)          # drop <<48/2=24>> spans
    text = re.sub(r"[ \t]+", " ", text).strip()   # tidy whitespace
    return text


def make_pair(question: str, answer_field: str, rng: random.Random, max_tries: int = 8) -> Pair | None:
    """Build a verified clean/corrupted pair from one GSM8K example."""
    clean_ans = extract_answer(answer_field)
    if clean_ans is None:
        return None
    ints = [int(t) for t in re.findall(r"(?<!\d)(\d+)(?!\d)", question)]
    ints = [v for v in ints if v > 1]  # avoid 0/1 which often don't change the answer
    if not ints:
        return None

    for _ in range(max_tries):
        old = rng.choice(ints)
        new = old + rng.choice([-3, -2, -1, 1, 2, 3, 5])
        if new <= 0 or new == old:
            continue
        corr_ans = recompute_answer(answer_field, old, new)
        if corr_ans is None or corr_ans == clean_ans:
            continue
        corrupted_q = re.sub(rf"(?<!\d){old}(?!\d)", str(new), question, count=1)
        if corrupted_q == question:
            continue
        return Pair(question, corrupted_q, clean_ans, corr_ans,
                    clean_solution=clean_solution_text(answer_field))
    return None


def build_pairs(dataset, limit: int | None, seed: int = 0) -> list[Pair]:
    """Build verified pairs from a HF GSM8K split (with 'question','answer')."""
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for ex in dataset:
        pair = make_pair(ex["question"], ex["answer"], rng)
        if pair is not None:
            pairs.append(pair)
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
