"""Edge-ranking circuit-discovery methods, unified behind one interface.

Each method scores every edge (higher = more important to the behavior). The pipeline then
sweeps a keep-budget over that ranking to draw a knee curve. All methods use the SAME
contrastive behavior metric so the only thing that differs is HOW they score.

Methods:
  EAP              - single-point Edge Attribution Patching (Syed et al.): one gradient x delta.
  EAPIG            - EAP with Integrated Gradients (Hanna et al.): gradient averaged over the
                     corrupted->clean path (fixes EAP's zero-gradient/saturation failure). Our
                     production ranker.
  EAPIGPathDelta   - EAP-GP-STYLE approximation (not a verbatim reimpl of arXiv 2502.06852):
                     integrate with the delta recomputed per-alpha so both delta and gradient
                     follow the model-dependent path.
  ActivationPatching - EXACT single-edge ablation KL (the ACDC scoring primitive / ground-truth
                     importance): turn one edge OFF, measure KL from the full circuit. O(E)
                     forward passes -- for small graphs and as the anchor the cheap methods are
                     correlated against.
  RandomRank       - control: random scores (the "random subset" baseline).
  AntiRank(base)   - control: negate a base method's scores (keep the LEAST important edges).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.eval import circuit as circuit_eval

BIN = 20.0  # log_alpha magnitude for a hard 0/1 gate (sigmoid(20) ~= 1)


def _metric(batch, device):
    """The shared contrastive behavior metric (logit(answer) - logit(foil) where a foil exists)."""
    return circuit_eval.make_logit_diff_metric(
        batch["labels"].to(device),
        batch["foil_ids"].to(device) if "foil_ids" in batch else None)


def _batches(dataset, collator, batch_size, rank_batches):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    for bi, batch in enumerate(loader):
        if bi >= rank_batches:
            break
        yield batch


class DiscoveryMethod:
    """Base: return a (num_edges,) importance score, higher = keep first."""
    name = "base"

    def rank_edges(self, model, dataset, collator, device, *,
                   rank_batches: int = 8, batch_size: int = 8) -> torch.Tensor:
        raise NotImplementedError


class EAP(DiscoveryMethod):
    name = "EAP"

    def rank_edges(self, model, dataset, collator, device, *, rank_batches=8, batch_size=8):
        model.eval()
        scores = torch.zeros(len(model.edges))
        n = 0
        for batch in _batches(dataset, collator, batch_size, rank_batches):
            corrupted = model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            scores += model.edge_attributions(
                batch["clean_ids"].to(device), batch["clean_mask"].to(device),
                corrupted, _metric(batch, device))
            n += 1
        return scores / max(n, 1)


class EAPIG(DiscoveryMethod):
    name = "EAP-IG"

    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps

    def rank_edges(self, model, dataset, collator, device, *, rank_batches=8, batch_size=8):
        model.eval()
        scores = torch.zeros(len(model.edges))
        n = 0
        for batch in _batches(dataset, collator, batch_size, rank_batches):
            corrupted = model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            scores += model.edge_attributions_ig(
                batch["clean_ids"].to(device), batch["clean_mask"].to(device),
                corrupted, _metric(batch, device), n_steps=self.n_steps)
            n += 1
        return scores / max(n, 1)


class EAPGP(DiscoveryMethod):
    """EAP-GP (Zhang et al. 2025, arXiv 2502.06852) -- from-equations reimplementation (no
    official code released). Adaptive GradPath integration; see edge_attributions_gp."""
    name = "EAP-GP"

    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps

    def rank_edges(self, model, dataset, collator, device, *, rank_batches=8, batch_size=8):
        model.eval()
        scores = torch.zeros(len(model.edges))
        n = 0
        for batch in _batches(dataset, collator, batch_size, rank_batches):
            corrupted = model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            scores += model.edge_attributions_gp(
                batch["clean_ids"].to(device), batch["clean_mask"].to(device),
                corrupted, _metric(batch, device), n_steps=self.n_steps)
            n += 1
        return scores / max(n, 1)


class ActivationPatching(DiscoveryMethod):
    """Exact per-edge ablation importance = KL(full circuit || circuit with that one edge OFF),
    averaged over positions and batches. The ACDC scoring primitive: expensive (one masked
    forward per edge) but exact, so it doubles as the ground truth the cheap attribution
    methods are correlated against. ``max_edges`` guards against running it on huge graphs."""
    name = "ActPatch"

    def __init__(self, max_edges: int = 400):
        self.max_edges = max_edges

    @torch.no_grad()
    def rank_edges(self, model, dataset, collator, device, *, rank_batches=2, batch_size=8):
        model.eval()
        E = len(model.edges)
        if E > self.max_edges:
            raise ValueError(f"ActivationPatching: {E} edges > max_edges={self.max_edges}; "
                             f"raise max_edges to run it on this graph (it's O(E) forwards).")
        batches = list(_batches(dataset, collator, batch_size, rank_batches))
        la = model.gate.log_alpha
        saved = la.data.clone()

        # Baseline: full circuit (all edges ON) log-probs per batch.
        la.data.fill_(BIN)
        base = []
        for batch in batches:
            corrupted = model.capture_contributions(
                batch["corr_ids"].to(device), batch["corr_mask"].to(device))
            logits = model(batch["clean_ids"].to(device), batch["clean_mask"].to(device),
                           corrupted=corrupted, masked=True)
            base.append((F.log_softmax(logits, dim=-1), batch["clean_mask"].to(device), corrupted,
                         batch["clean_ids"].to(device)))

        scores = torch.zeros(E)
        for e in range(E):
            la.data.fill_(BIN)
            la.data[e] = -BIN  # this edge OFF, everything else ON
            kl_sum, w_sum = 0.0, 0.0
            for full_lp, cmask, corrupted, clean_ids in base:
                logits = model(clean_ids, cmask, corrupted=corrupted, masked=True)
                lp = F.log_softmax(logits, dim=-1)
                kl = (full_lp.exp() * (full_lp - lp)).sum(dim=-1)  # KL(full || ablated) per token
                m = cmask.to(kl.dtype)
                kl_sum += float((kl * m).sum())
                w_sum += float(m.sum())
            scores[e] = kl_sum / max(w_sum, 1.0)
        la.data.copy_(saved)
        return scores


class RandomRank(DiscoveryMethod):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def rank_edges(self, model, dataset, collator, device, *, rank_batches=8, batch_size=8):
        g = torch.Generator().manual_seed(self.seed)
        return torch.rand(len(model.edges), generator=g)


class AntiRank(DiscoveryMethod):
    """Keep the LEAST important edges (negate a base method) -- the B2 anti-rank control."""

    def __init__(self, base: DiscoveryMethod):
        self.base = base
        self.name = f"Anti-{base.name}"

    def rank_edges(self, model, dataset, collator, device, *, rank_batches=8, batch_size=8):
        return -self.base.rank_edges(model, dataset, collator, device,
                                     rank_batches=rank_batches, batch_size=batch_size)


REGISTRY: dict[str, DiscoveryMethod] = {
    "EAP": EAP(),
    "EAP-IG": EAPIG(),
    "EAP-GP": EAPGP(),
    "ActPatch": ActivationPatching(),
    "Random": RandomRank(),
}
