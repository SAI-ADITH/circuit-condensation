"""Capability and circuit-size measurement for the eval suite.

- ``generative_accuracy``: generate a solution and match the final number (the
  real GSM8K capability metric -- full model, gate-independent).
- ``circuit_token_accuracy``: teacher-forced answer-token accuracy through the
  masked circuit (gate-dependent -- the knee y-axis).
- ``circuit_faithfulness``: mean token KL(full || masked) -- how well the small
  circuit reproduces the full model.
- ``circuit_size``: number of edges the hard-concrete gates leave ON.
- ``make_logit_diff_metric``: the scalar metric EAP differentiates (log-prob of
  the answer tokens), the behaviour signal for attribution.
"""

from __future__ import annotations

import re

import torch
from torch.utils.data import DataLoader

# A number: optional sign/$, digits with thousands separators, optional decimal.
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_int(token: str) -> int | None:
    try:
        val = float(token.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None
    if val != val or val in (float("inf"), float("-inf")) or abs(val) > 1e15:
        return None  # garbage / runaway generation -> not a valid answer
    return int(round(val))


def _extract_pred(text: str) -> int | None:
    """Predicted integer from a generation. Prefer the number right after the last
    ``####``; otherwise the last number in the text. Tolerates $, commas, decimals."""
    if "####" in text:
        nums = _NUM_RE.findall(text.rsplit("####", 1)[-1])
        if nums:
            return _to_int(nums[0])
    nums = _NUM_RE.findall(text)
    return _to_int(nums[-1]) if nums else None


@torch.no_grad()
def generative_accuracy(peft_model, tokenizer, examples, device,
                        max_new_tokens: int = 512, batch_size: int = 8,
                        max_examples: int = 64, prompt_prefix: str = "",
                        return_details: bool = False, keep_samples: int | None = None):
    """Real GSM8K accuracy: generate from each prompt with the full model and
    compare the final number to the gold answer.

    ``examples`` is a list of dicts with ``prompt`` (str) and ``clean_answer``
    (int). ``prompt_prefix`` is prepended to every prompt (e.g. few-shot examples).
    With ``return_details`` returns a dict (accuracy, n, unparseable rate, and a few
    sample generations) for observability; otherwise just the accuracy float.

    Generation uses the full model (LoRA on, masks off) — autoregressive decoding
    through the masked circuit is impractical, so the masked circuit is tracked
    separately via the faithfulness KL.
    """
    examples = examples[:max_examples]
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    correct = total = unparseable = 0
    samples = []
    try:
        for i in range(0, len(examples), batch_size):
            chunk = examples[i: i + batch_size]
            enc = tokenizer([prompt_prefix + e["prompt"] for e in chunk],
                            return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            out = peft_model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for e, text in zip(chunk, texts):
                total += 1
                pred = _extract_pred(text)
                gold = int(e["clean_answer"])
                ok = pred == gold
                correct += int(ok)
                unparseable += int(pred is None)
                if keep_samples is None or len(samples) < keep_samples:
                    samples.append({"prompt": e["prompt"], "generation": text,
                                    "pred": pred, "gold": gold, "correct": ok})
    finally:
        tokenizer.padding_side = side
    acc = correct / max(total, 1)
    if return_details:
        return {"accuracy": acc, "n": total,
                "unparseable_rate": unparseable / max(total, 1), "samples": samples}
    return acc


def circuit_size(masked_model, threshold: float = 0.5) -> int:
    """Number of ON edges in the snapped 0/1 circuit."""
    return masked_model.gate.num_open(threshold)


@torch.no_grad()
def circuit_token_accuracy(masked_model, dataset, collator, device,
                           batch_size=8, num_batches=8) -> float:
    """Teacher-forced answer-token accuracy through the *masked circuit*.

    Unlike ``generative_accuracy`` (full model, gate-independent), this runs the
    trimmed circuit (wires OFF feed the corrupted contribution) and reports the
    fraction of supervised answer tokens it predicts correctly. It therefore
    depends on the edges -- the right y-axis for the knee curve.
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        corr_ids = batch["corr_ids"].to(device)
        corr_mask = batch["corr_mask"].to(device)
        corrupted = masked_model.capture_contributions(corr_ids, corr_mask)
        logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        preds = logits[:, :-1, :].argmax(dim=-1)
        tgt = labels[:, 1:]
        sup = tgt != -100
        correct += int(((preds == tgt) & sup).sum().item())
        total += int(sup.sum().item())
    return correct / max(total, 1)


@torch.no_grad()
def ioi_accuracy(masked_model, dataset, collator, device, batch_size=8, num_batches=8) -> float:
    """Full-model IOI capability: fraction where the IO name is the top next-token
    prediction (teacher-forced at the answer position, masks OFF).

    This is the IOI analogue of ``generative_accuracy`` — the capability anchor
    that must stay flat while edges drop. The answer is a single token, so
    teacher-forced argmax == real accuracy (no free-generation gap).
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        ids = batch["clean_ids"].to(device)
        mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        preds = masked_model(ids, mask, masked=False)[:, :-1, :].argmax(dim=-1)
        tgt = labels[:, 1:]
        sup = tgt != -100
        correct += int(((preds == tgt) & sup).sum().item())
        total += int(sup.sum().item())
    return correct / max(total, 1)


@torch.no_grad()
def ioi_forced_choice_accuracy(masked_model, dataset, collator, device, masked: bool,
                               batch_size=8, num_batches=8) -> float:
    """IOI via the canonical Wang et al. IO-vs-S logit difference (forced choice): correct
    iff the model puts more mass on the indirect object (answer_id) than on the subject
    (s_id) at the answer slot. Replaces argmax-over-vocab, which penalised models that do
    the IOI computation perfectly but prefer a non-name top token -- e.g. qwen3-4b scored
    0.69 by argmax yet 1.00 by forced choice; every model is >=0.98 here. masked=True runs
    the trimmed circuit (OFF edges read the corrupted activation); masked=False the full model."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        if masked:
            corrupted = masked_model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        else:
            logits = masked_model(clean_ids, clean_mask, masked=False)
        sup = labels[:, 1:] != -100
        for r in range(clean_ids.size(0)):
            ex = dataset.examples[bi * batch_size + r]
            j = int(sup[r].nonzero()[0].item())
            correct += int(float(logits[r, j, ex["answer_id"]]) > float(logits[r, j, ex["s_id"]]))
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def induction_pool_accuracy(masked_model, dataset, collator, device, masked: bool,
                            batch_size=8, num_batches=8) -> float:
    """Induction restricted to the answer pool: correct iff the gold copy target (answer_id)
    is the top-scoring token among the fixed word pool (pool_ids) at the answer slot. Removes
    the argmax-over-vocab bias where a generic continuation (" the", punctuation) outranks
    the correct word even when the model clearly prefers it among plausible copy targets
    (e.g. gemma induction 0.63->0.77, llama with BOS 0.80->0.93). masked as above."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        if masked:
            corrupted = masked_model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        else:
            logits = masked_model(clean_ids, clean_mask, masked=False)
        sup = labels[:, 1:] != -100
        for r in range(clean_ids.size(0)):
            ex = dataset.examples[bi * batch_size + r]
            j = int(sup[r].nonzero()[0].item())
            pool = torch.tensor(ex["pool_ids"], device=device)
            best = int(pool[int(logits[r, j, pool].argmax())])
            correct += int(best == ex["answer_id"])
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def greater_than_accuracy(masked_model, dataset, collator, device, masked: bool,
                          batch_size=8, num_batches=8) -> float:
    """Greater-than capability via Hanna's *argmax-valid* metric: the model is
    correct at the answer slot iff its top predicted 2-digit token is a year
    STRICTLY GREATER than the example's start year (i.e. the argmax token id is in
    that example's ``valid_ids`` set). The answer is a single token, so this is the
    real accuracy with no generation gap (the IOI story).

    ``masked=True``  -> run the trimmed circuit (OFF edges read corrupted activations):
                         the gate-dependent **knee y-axis**.
    ``masked=False`` -> run the full model (gates OFF): the **capability anchor**
                         that must stay flat as edges drop.

    ``valid_ids`` are read from ``dataset.examples`` in order; the DataLoader is
    unshuffled so row r of batch b is example ``b*batch_size + r``.
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        if masked:
            corr_ids = batch["corr_ids"].to(device)
            corr_mask = batch["corr_mask"].to(device)
            corrupted = masked_model.capture_contributions(corr_ids, corr_mask)
            logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        else:
            logits = masked_model(clean_ids, clean_mask, masked=False)
        preds = logits[:, :-1, :].argmax(dim=-1)   # predicted next token at each pos
        tgt = labels[:, 1:]
        sup = tgt != -100                           # the single answer slot per row
        rows = clean_ids.size(0)
        for r in range(rows):
            ex = dataset.examples[bi * batch_size + r]
            valid = set(ex["valid_ids"])
            j = int(sup[r].nonzero()[0].item())     # supervised column for this row
            correct += int(int(preds[r, j].item()) in valid)
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def agreement_accuracy(masked_model, dataset, collator, device, masked: bool,
                       batch_size=8, num_batches=8) -> float:
    """Agreement capability via Linzen 2016 FORCED-CHOICE scoring: at the verb slot,
    correct iff the model assigns MORE probability to the correct-number verb than to the
    SAME verb in the wrong number (``answer_id`` vs ``contrast_id``). This is the standard
    number-agreement metric and, unlike argmax-over-vocabulary, does not penalise the model
    for preferring a natural function-word continuation over the narrow content-verb pool —
    that argmax-valid metric scored even base models at ~0.3 (below the 0.5 minimal-pair
    chance line) because the prompt does not *force* a content verb; the minimal-pair
    contrast isolates the grammatical-number decision the circuit actually makes.

    ``masked=True``  -> through the trimmed circuit (OFF edges read corrupted activation).
    ``masked=False`` -> through the full model (the capability anchor)."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        if masked:
            corr_ids = batch["corr_ids"].to(device)
            corr_mask = batch["corr_mask"].to(device)
            corrupted = masked_model.capture_contributions(corr_ids, corr_mask)
            logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        else:
            logits = masked_model(clean_ids, clean_mask, masked=False)
        tgt = labels[:, 1:]
        sup = tgt != -100
        for r in range(clean_ids.size(0)):
            ex = dataset.examples[bi * batch_size + r]
            aid, cid = ex["answer_id"], ex["contrast_id"]
            if cid < 0:
                continue  # no minimal-pair foil recorded (pre-Linzen data) -> skip
            j = int(sup[r].nonzero()[0].item())  # column that PREDICTS the answer token
            correct += int(float(logits[r, j, aid]) > float(logits[r, j, cid]))
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def forced_choice_margins(masked_model, dataset, collator, device, task: str, masked: bool,
                          batch_size=8, num_batches=8):
    """Per-example forced-choice MARGIN = logit(answer) - logit(foil) at the answer slot,
    for the FAILURE-PREDICTION analysis (Option 2). Same scoring rule as the *_accuracy
    functions above, but emits the scalar margin (not a boolean) so it can serve as a
    confidence/failure signal. A small or negative margin means the deciding computation is
    not firing cleanly -> the model is likely wrong.

    Foil per task: agreement -> contrast_id (same verb, wrong number); ioi -> s_id (subject);
    induction -> the best competing token in the answer pool (pool_ids minus the answer).
    Returns (margins, correct) as float lists aligned by example. masked as elsewhere."""
    foil_key = {"agreement": "contrast_id", "ioi": "s_id"}.get(task)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    margins, correct = [], []
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        labels = batch["labels"].to(device)
        if masked:
            corrupted = masked_model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        else:
            logits = masked_model(clean_ids, clean_mask, masked=False)
        sup = labels[:, 1:] != -100
        for r in range(clean_ids.size(0)):
            ex = dataset.examples[bi * batch_size + r]
            aid = ex["answer_id"]
            j = int(sup[r].nonzero()[0].item())
            if task == "induction":                       # answer vs best competing pool token
                pool = torch.tensor(ex["pool_ids"], device=device)
                comp = pool[pool != aid]
                if comp.numel() == 0:
                    continue
                m = float(logits[r, j, aid]) - float(logits[r, j, comp].max())
            else:
                fid = ex[foil_key]
                if fid is None or fid < 0:
                    continue
                m = float(logits[r, j, aid]) - float(logits[r, j, fid])
            margins.append(m)
            correct.append(int(m > 0))
    return margins, correct


def _parse_end_year(text: str, century: int) -> int | None:
    """Parse the END year a model wrote after a '...to the year ' prompt (the
    generative greater-than path, tokenizer-agnostic). Collapse spaced digits
    ('8 0'->'80'), take the first 3-4 digit run as the full year. None if absent."""
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    m = re.search(r"\d+", txt)
    if not m:
        return None
    d = m.group(0)
    return int(d[:4]) if len(d) >= 3 else None


@torch.no_grad()
def greater_than_generative_accuracy(masked_model, tokenizer, examples, device, masked: bool,
                                     max_new_tokens: int = 8, max_examples: int = 80) -> float:
    """Greater-than accuracy when the model GENERATES the full end year itself
    (the tokenizer-robust path for models that don't pack a 2-digit year into one
    token — Llama/Qwen). The prompt was cut at '...to the year ' (see
    ``GreaterThanDataset(generative=True)``); the model writes a 4-digit end year,
    which we decode-and-parse and check ``> start_year``. No single-token assumption.

    ``masked=True``  -> decode greedily through the trimmed circuit (OFF edges read
                        the corrupted activation), the gate-dependent knee y-axis.
    ``masked=False`` -> decode through the full model (gates OFF), the capability anchor.
    Examples are dataset ``.examples`` dicts carrying ``clean_ids``, ``labels``
    (-100 over the prompt), ``corr_ids``, ``century`` and ``start_yy``.
    """
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    correct = total = 0
    for e in examples[:max_examples]:
        labels = e["labels"]
        prompt_len = sum(1 for x in labels if x == -100) or 1
        clean = list(e["clean_ids"][:prompt_len])
        corr = list(e["corr_ids"])
        for _ in range(max_new_tokens):
            if masked:
                L = max(len(clean), len(corr))
                ci = torch.tensor([clean + [pad] * (L - len(clean))], device=device)
                cm = torch.tensor([[1] * len(clean) + [0] * (L - len(clean))], device=device)
                xi = torch.tensor([corr + [pad] * (L - len(corr))], device=device)
                xm = torch.tensor([[1] * len(corr) + [0] * (L - len(corr))], device=device)
                corrupted = masked_model.capture_contributions(xi, xm)
                logits = masked_model(ci, cm, corrupted=corrupted, masked=True)
            else:
                ci = torch.tensor([clean], device=device)
                cm = torch.ones_like(ci)
                logits = masked_model(ci, cm, masked=False)
            nxt = int(logits[0, len(clean) - 1].argmax())
            if nxt == eos:
                break
            clean.append(nxt)
            corr.append(nxt)
        text = tokenizer.decode(clean[prompt_len:], skip_special_tokens=True)
        end_year = _parse_end_year(text, e["century"])
        start_year = e["century"] * 100 + e["start_yy"]
        total += 1
        correct += int(end_year is not None and end_year > start_year)
    return correct / max(total, 1)


@torch.no_grad()
def circuit_faithfulness(masked_model, dataset, collator, device, batch_size=8,
                         num_batches: int = 1) -> float:
    """Mean token KL( full model || masked circuit ) over ``num_batches`` batches.

    With num_batches=1 this is an n=8 estimate with a ~+-0.2 spread, while the accuracy beside it
    is measured on the full 1000-example sealed split. Pass n_test_batches to make the two
    like-for-like.

    Low = the small circuit reproduces the full model (the core circuit-quality
    claim). Reported alongside accuracy at each knee point.
    """
    import torch.nn.functional as F
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    tot, den = 0.0, 0.0
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        clean_ids = batch["clean_ids"].to(device)
        clean_mask = batch["clean_mask"].to(device)
        corr_ids = batch["corr_ids"].to(device)
        corr_mask = batch["corr_mask"].to(device)
        corrupted = masked_model.capture_contributions(corr_ids, corr_mask)
        masked_logits = masked_model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        full_logits = masked_model(clean_ids, clean_mask, masked=False)
        full = F.log_softmax(full_logits, dim=-1)
        msk = F.log_softmax(masked_logits, dim=-1)
        kl = (full.exp() * (full - msk)).sum(dim=-1)
        m = clean_mask.to(kl.dtype)
        tot += float((kl * m).sum()); den += float(m.sum())
    return tot / max(den, 1.0)


@torch.no_grad()
def generative_circuit_accuracy(masked_model, tokenizer, examples, device,
                                max_new_tokens: int = 200, max_examples: int = 40,
                                return_details: bool = False, verbose: bool = False,
                                keep_samples: int | None = 5):
    """GSM8K accuracy when the **masked circuit generates the answer itself**.

    Unlike ``circuit_token_accuracy`` (teacher-forced: gold prefix fed token by token)
    and ``generative_accuracy`` (full model, gates OFF), this decodes greedily where
    every next-token logit comes from the *disentangled masked forward* — gates applied,
    OFF edges reading the corrupted activation. It answers the question: does the trimmed circuit actually SOLVE the problem on its own, not just predict
    the next gold token?

    Alignment: the clean and corrupted sequences grow by the SAME generated token each
    step, so they stay position-aligned for the per-edge blend (the corruption lives only
    in the prompt, exactly as in training). Sequences are right-padded to a common length
    so absolute/rotary positions of the real tokens are correct (no left-pad offset). There
    is no KV cache (the custom forward recomputes), so cost is O(n^2): run on a modest
    sample. ``examples`` are dataset ``.examples`` dicts carrying ``clean_ids``, ``labels``
    (-100 over the prompt), ``corr_ids`` and ``clean_answer``.
    """
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    import time as _time
    correct = total = unparseable = 0
    samples = []
    _t0 = _time.time()
    for e in examples[:max_examples]:
        labels = e["labels"]
        prompt_len = sum(1 for x in labels if x == -100) or 1   # leading -100 == prompt span
        clean = list(e["clean_ids"][:prompt_len])               # clean prompt (no gold answer)
        corr = list(e["corr_ids"])                              # corrupted prompt
        hit_eos = False
        for step_i in range(max_new_tokens):
            L = max(len(clean), len(corr))
            ci = torch.tensor([clean + [pad] * (L - len(clean))], device=device)
            cm = torch.tensor([[1] * len(clean) + [0] * (L - len(clean))], device=device)
            xi = torch.tensor([corr + [pad] * (L - len(corr))], device=device)
            xm = torch.tensor([[1] * len(corr) + [0] * (L - len(corr))], device=device)
            corrupted = masked_model.capture_contributions(xi, xm)
            logits = masked_model(ci, cm, corrupted=corrupted, masked=True)
            nxt = int(logits[0, len(clean) - 1].argmax())
            if nxt == eos:
                hit_eos = True
                break
            clean.append(nxt)
            corr.append(nxt)
            # Early stop once the answer is complete: GSM8K ends "#### <number>". Greedy
            # decode through a pruned circuit rarely emits a clean EOS (it rambles), which
            # wastes tokens and lets _extract_pred grab a stale number. Stop as soon as a
            # COMPLETE number has been written after "####" (a non-digit follows it).
            if step_i % 4 == 0:
                tail = tokenizer.decode(clean[prompt_len:][-40:], skip_special_tokens=True)
                if "####" in tail and re.search(r"####\s*-?\$?\d[\d,]*(?:\.\d+)?\D", tail):
                    hit_eos = True   # treated as a clean, intentional stop
                    break
        gen_ids = clean[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = _extract_pred(text)
        gold = int(e["clean_answer"])
        total += 1
        correct += int(pred == gold)
        unparseable += int(pred is None)
        # Failure taxonomy so a deep-dive can separate REAL reasoning errors from
        # truncation / no-#### / degeneration artifacts.
        if keep_samples is None or len(samples) < keep_samples:
            samples.append({"generation": text, "pred": pred, "gold": gold,
                            "correct": pred == gold, "n_gen_tokens": len(gen_ids),
                            "hit_eos": hit_eos, "truncated": not hit_eos,
                            "has_marker": "####" in text})
        if verbose and total % 5 == 0:
            print(f"    [gen-circuit] {total}/{min(max_examples, len(examples))} "
                  f"running_acc={correct/total:.3f} ({_time.time()-_t0:.0f}s)", flush=True)
    acc = correct / max(total, 1)
    if return_details:
        return {"accuracy": acc, "n": total,
                "unparseable_rate": unparseable / max(total, 1), "samples": samples}
    return acc


def make_logit_diff_metric(labels: torch.Tensor, foil_ids: torch.Tensor | None = None):
    """Return metric_fn(logits)->scalar: the behaviour EAP/EAP-IG attributes to edges.

    CONTRASTIVE when ``foil_ids`` is given: at each supervised
    answer slot the metric is ``logp(answer) - logp(foil)``. Because both share the same
    ``log_softmax`` normaliser, this equals ``logit(answer) - logit(foil)`` EXACTLY (the
    logsumexp cancels) -- so the name is literal, and the attribution targets the
    SAME answer-vs-foil decision the forced-choice eval scores (IOI: IO-vs-S; agreement:
    correct-vs-wrong-number verb), not "boost the answer token generically". Without a
    contrastive metric, attribution over-credits generic answer-boosting edges and
    under-credits the suppress-the-foil edges that are the heart of e.g. the Wang circuit.

    ``foil_ids`` is a LongTensor[B] (from PairCollator); an entry < 0 means "no foil" and
    that row falls back to plain ``logp(answer)`` (induction/doc/gsm8k, whose eval is a
    pool/vocab argmax rather than a single-pair choice). ``foil_ids=None`` reproduces the
    original summed-log-prob behaviour for callers that don't thread it.
    """

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = labels[:, 1:]
        sup = tgt != -100
        gathered = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        out = gathered
        if foil_ids is not None:
            # logp(foil) at every position for each row; subtract only where foil >= 0.
            fid = foil_ids.to(logp.device).view(-1, 1, 1).expand(-1, logp.size(1), 1).clamp(min=0)
            logp_foil = logp.gather(-1, fid).squeeze(-1)
            has_foil = (foil_ids.to(logp.device) >= 0).view(-1, 1).to(logp.dtype)
            out = gathered - has_foil * logp_foil
        return (out * sup.to(logp.dtype)).sum()

    return metric_fn
