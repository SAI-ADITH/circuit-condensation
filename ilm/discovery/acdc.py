"""Faithful ACDC (Conmy et al. 2023, "Towards Automated Circuit Discovery").

The real ACDC is ITERATIVE and greedy -- NOT a one-shot ranking. It walks edges in
reverse-topological order (from the output back toward the input) and, for each edge,
tentatively removes it (feeds the corrupted-run activation instead). If removing the edge
raises the circuit's KL divergence from the full model by less than a threshold tau, the
removal is kept; otherwise the edge is restored. Because earlier removals change the graph
that later edges are judged against, the process is genuinely sequential -- this is why ACDC
is slow (O(edges) forward passes per tau) and why gradient methods (EAP/EAP-IG) were later
introduced to approximate it in one pass.

Matches the reference implementation's contract (github.com/ArthurConmy/Automatic-Circuit-Discovery):
  - metric = KL(full model clean-run || pruned circuit), the ACDC default;
  - reverse-topological order over readers (logits first, then last layer ... embedding);
  - threshold tau controls sparsity (larger tau -> more removed -> smaller circuit);
  - sweep tau to trace the accuracy/faithfulness-vs-edges knee.

We diverge from the reference ONLY in mechanics that don't change the algorithm: our disentangled
edge-masked model already exposes per-edge on/off gates, so "remove an edge" = set its gate to 0
(feed the corrupted activation), exactly the ablation ACDC prescribes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.eval import circuit as circuit_eval
from .methods import BIN
from .pipeline import set_circuit, circuit_accuracy


def _reader_rank(reader: str) -> int:
    """Topological position of a reader node (higher = later / closer to the output)."""
    if reader == "embed":
        return -1
    if reader == "logits":
        return 1 << 30
    kind, i = reader.split("_")  # "attn"/"mlp", layer index
    return 2 * int(i) + (0 if kind == "attn" else 1)


def _cache_batches(model, dataset, collator, device, batch_size, rank_batches):
    """Precompute, per batch: clean ids/mask, corrupted activations, the FULL model's clean-run
    log-probs (the KL reference ACDC prunes against), and the SUPERVISED-SLOT mask.

    The supervised mask keeps this faithful to the ACDC paper. Conmy et al.
    define H(x_i, x'_i) as "the output probability distribution of the subgraph" and D_KL as "the
    average KL divergence over a set of DATAPOINTS" -- one distribution per example, at the slot
    where the task is answered. Averaging over every token position instead would divide
    each edge's measured effect by roughly the sequence length, making a nominal tau behave
    like a tau ~20x larger (the paper's own IOI runs use tau=0.0575 and 0.00398). It would also
    disagree with EAP/EAP-IG in this repo, which attribute through make_logit_diff_metric restricted to the same slots.
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    cached = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= rank_batches:
                break
            clean_ids = batch["clean_ids"].to(device)
            clean_mask = batch["clean_mask"].to(device)
            corrupted = model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            full = model(clean_ids, clean_mask, masked=False)
            # Answer slots, shifted the same way make_logit_diff_metric shifts them: logits[:, :-1]
            # predicts labels[:, 1:]. Fall back to all real tokens only if a collator gives no
            # labels, which no current task does.
            if "labels" in batch:
                sup = (batch["labels"].to(device)[:, 1:] != -100)
            else:
                sup = clean_mask[:, 1:].bool()
            cached.append((clean_ids, clean_mask, corrupted,
                           F.log_softmax(full[:, :-1, :], dim=-1), sup))
    return cached


@torch.no_grad()
def _circuit_kl(model, cached) -> float:
    """Mean KL(full model || current masked circuit) at the SUPERVISED ANSWER SLOTS.

    Per Conmy et al.'s definition: one output distribution per datapoint, averaged over datapoints.
    See _cache_batches for why averaging over all token positions was wrong."""
    tot, w = 0.0, 0.0
    for clean_ids, clean_mask, corrupted, ref_lp, sup in cached:
        logits = model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        kl = (ref_lp.exp() * (ref_lp - lp)).sum(dim=-1)
        m = sup.to(kl.dtype)
        tot += float((kl * m).sum()); w += float(m.sum())
    return tot / max(w, 1.0)


@torch.no_grad()
def acdc_circuit(model, dataset, collator, device, tau: float, *,
                 rank_batches: int = 2, batch_size: int = 8, max_edges: int = 3600):
    """Run one greedy ACDC pass at threshold ``tau``; return (kept_edge_set, final_kl)."""
    E = len(model.edges)
    if E > max_edges:
        raise ValueError(f"ACDC: {E} edges > max_edges={max_edges} (O(E) forwards/tau is too slow).")
    la = model.gate.log_alpha
    la.data.fill_(BIN)  # start from the full circuit (every edge ON)
    cached = _cache_batches(model, dataset, collator, device, batch_size, rank_batches)
    order = sorted(range(E), key=lambda e: _reader_rank(model.edges[e][1]), reverse=True)
    kept = set(range(E))
    cur_kl = _circuit_kl(model, cached)  # ~0: all-on circuit == full model
    for e in order:
        la.data[e] = -BIN                 # tentatively remove edge e
        new_kl = _circuit_kl(model, cached)
        if new_kl - cur_kl < tau:         # removal barely hurts -> keep it removed
            cur_kl = new_kl
            kept.discard(e)
        else:
            la.data[e] = BIN              # restore
    return kept, cur_kl


def acdc_knee(model, dataset, collator, device, task, *,
              taus=(0.0003, 0.0007, 0.0015, 0.003, 0.006, 0.012, 0.025, 0.05), eval_ds=None,
              rank_batches: int = 2, eval_batches: int = 25, batch_size: int = 8) -> dict:
    """Sweep tau; for each, run greedy ACDC and record (edges, accuracy, faithfulness). ``dataset``
    is what ACDC prunes on; ``eval_ds`` (default = dataset) is the sealed split it's scored on."""
    eval_ds = eval_ds or dataset
    la = model.gate.log_alpha
    saved = la.data.clone()
    curve = []
    try:
        for tau in taus:
            kept, _ = acdc_circuit(model, dataset, collator, device, tau,
                                   rank_batches=rank_batches, batch_size=batch_size)
            realised = set_circuit(model, kept)
            acc = circuit_accuracy(model, eval_ds, collator, device, task,
                                   batch_size=batch_size, num_batches=eval_batches)
            faith = circuit_eval.circuit_faithfulness(model, eval_ds, collator, device,
                                                      batch_size=batch_size)
            curve.append({"tau": tau, "edges": realised, "acc": round(acc, 4), "faith": round(faith, 4)})
    finally:
        la.data.copy_(saved)
    return {"method": "ACDC", "task": task, "n_edges": len(model.edges), "curve": curve}
