"""Shared helpers for the verification scripts: cell resolution, dataset
construction, per-task accuracy dispatch, contrastive logit margins, and circuit loading.

Conventions (mirroring the repo):
  task keys      : "ioi" | "agreement" | "induction" | "doc"  (doc -> data/docstring)
  full model     : all gates ON through the masked forward (verified exactly equal to the
                   unmasked forward: every edge reads its clean activation)
  contrast foil  : ioi -> foil_id (subject S); agreement -> contrast_id (same verb, wrong
                   number); induction/doc -> None (plain logit(answer)), matching the
                   discovery module's make_logit_diff_metric fallback.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from peft import PeftModel

from ilm.masking.edge_masked_model import EdgeMaskedTransformer
from ilm.data.dataset import PairCollator
from ilm.data.ioi import IOIDataset, load_samples as load_ioi
from ilm.data.agreement import AgreementDataset, load_samples as load_agr
from ilm.data.induction import InductionDataset, load_samples as load_ind
from ilm.data.docstring import DocstringDataset, load_samples as load_doc
from ilm.training.setup import load_tokenizer
from ilm.eval import circuit as C

ROOT = Path(os.environ.get("ILM_ROOT", Path(__file__).resolve().parent.parent))

MODELS = ["gpt2", "llama", "llama3b", "gemma1b", "gemma4b", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]
TASKS = ["ioi", "agreement", "induction", "doc"]
MPATH = {"gpt2": "models/gpt2", "llama": "models/Llama-3.2-1B-Instruct",
         "llama3b": "models/Llama-3.2-3B-Instruct", "gemma1b": "models/gemma-3-1b-it",
         "gemma4b": "models/gemma-3-4b-it", "qwen3-0.6b": "models/Qwen3-0.6B",
         "qwen3-1.7b": "models/Qwen3-1.7B", "qwen3-4b": "models/Qwen3-4B"}
PREFIX = {"ioi": "ioi", "agreement": "agr", "induction": "ind", "doc": "doc"}
TASK_DIR = {"ioi": "ioi", "agreement": "agreement", "induction": "induction", "doc": "docstring"}
MAXLEN = {"ioi": 64, "agreement": 64, "induction": 64, "doc": 96}


# Env overrides so the verification scripts (necessity/minimality) can target a given arm and
# granularity without editing their logic:
#   COND_CKPT_ROOT  e.g. checkpoints/headsave/c1gate/seed11 -> condensed = {root}/{PREFIX}_{model}
#   FRESH_SUFFIX    default "_fresh"; set "_head_fresh" for the head grid's frozen-arm base
#   GRAN            default "block"; set "head" for the head grid
def fresh_ckpt(task, model) -> Path:
    suf = os.environ.get("FRESH_SUFFIX", "_fresh")
    return ROOT / f"checkpoints/{PREFIX[task]}_{model}{suf}"


def condensed_ckpt(task, model):
    """Resolve a saved condensed circuit (lora_healed + gates_healed.pt), newest scheme first."""
    root = os.environ.get("COND_CKPT_ROOT")
    cands = []
    if root:
        cands.append((Path(root) if Path(root).is_absolute() else ROOT / root) / f"{PREFIX[task]}_{model}")
    cands += [ROOT / f"checkpoints/gridsave/{PREFIX[task]}_{model}",
              ROOT / f"checkpoints/seedstab/{TASK_DIR[task]}_{model}_s0"]
    if task == "ioi":
        cands.append(ROOT / f"checkpoints/ioifull/ioi_{model}")
    for c in cands:
        if (c / "gates_healed.pt").exists():
            return c
    return None


def make_ds(task, model, tok, split="test"):
    d = ROOT / "data" / TASK_DIR[task] / model / f"{split}.jsonl"
    ml = MAXLEN[task]
    return {"ioi": lambda: IOIDataset(load_ioi(d), tok, ml),
            "agreement": lambda: AgreementDataset(load_agr(d), tok, ml),
            "induction": lambda: InductionDataset(load_ind(d), tok, ml),
            "doc": lambda: DocstringDataset(load_doc(d), tok, ml)}[task]()


def load_masked(model, lora_dir, gates_file, device, granularity="block"):
    base = AutoModelForCausalLM.from_pretrained(str(ROOT / MPATH[model]), torch_dtype=torch.float32,
                                                attn_implementation="eager")
    m = EdgeMaskedTransformer(PeftModel.from_pretrained(base, str(lora_dir), is_trainable=False),
                              granularity=granularity).to(device)
    m.gate.load_state_dict(torch.load(gates_file, map_location=device))
    m.eval()
    return m


def load_condensed(task, model, device, granularity=None):
    granularity = granularity or os.environ.get("GRAN", "block")
    ck = condensed_ckpt(task, model)
    if ck is None:
        return None
    return load_masked(model, ck / "lora_healed", ck / "gates_healed.pt", device, granularity)


def load_frozen(task, model, device, granularity=None, all_on=True):
    granularity = granularity or os.environ.get("GRAN", "block")
    ck = fresh_ckpt(task, model)
    m = load_masked(model, ck / "lora", ck / "gates.pt", device, granularity)
    if all_on:
        m.gate.log_alpha.data.fill_(20.0)
    return m


def kept_edges(m) -> set[int]:
    hm = m.gate.hard_mask().bool()
    return {i for i in range(len(m.edges)) if hm[i]}


def task_acc(m, task, ds, coll, device, nb=25, bs=8) -> float:
    """Accuracy through the CURRENT gate mask (all-on mask == full model)."""
    if task == "ioi":
        return C.ioi_forced_choice_accuracy(m, ds, coll, device, masked=True, batch_size=bs, num_batches=nb)
    if task == "agreement":
        return C.agreement_accuracy(m, ds, coll, device, masked=True, batch_size=bs, num_batches=nb)
    if task == "induction":
        return C.induction_pool_accuracy(m, ds, coll, device, masked=True, batch_size=bs, num_batches=nb)
    return C.circuit_token_accuracy(m, ds, coll, device, batch_size=bs, num_batches=nb)


def foil_of(task, ex):
    if task == "ioi":
        return ex.get("foil_id", ex.get("s_id"))
    if task == "agreement":
        cid = ex.get("contrast_id", -1)
        return cid if cid is not None and cid >= 0 else None
    return None   # induction / doc: plain logit(answer)


@torch.no_grad()
def logit_margin(m, task, ds, coll, device, nb=25, bs=8) -> float:
    """Mean logit(answer) - logit(foil) (or plain logit(answer) where no foil) through the
    CURRENT gate mask. OFF edges read the corrupted-run activation (the repo's standard)."""
    tot, n = 0.0, 0
    for bi, b in enumerate(DataLoader(ds, batch_size=bs, collate_fn=coll)):
        if bi >= nb:
            break
        ci, cm = b["clean_ids"].to(device), b["clean_mask"].to(device)
        corr = m.capture_contributions(b["corr_ids"].to(device), b["corr_mask"].to(device))
        logits = m(ci, cm, corrupted=corr, masked=True)
        tgt = b["labels"].to(device)[:, 1:]
        sup = (tgt != -100)
        for r in range(ci.size(0)):
            ex = ds.examples[bi * bs + r]
            nz = sup[r].nonzero()
            if nz.numel() == 0:
                continue
            j = int(nz[0].item())
            # answer token read from labels (dataset-agnostic; docstring examples carry no
            # answer_id key, but the supervised label position IS the answer token)
            v = float(logits[r, j, int(tgt[r, j])])
            f = foil_of(task, ex)
            tot += v - (float(logits[r, j, f]) if f is not None else 0.0)
            n += 1
    return tot / max(n, 1)
