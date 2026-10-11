"""The EAP concentration panel (evaluation only).

Given per-edge attribution scores, collapse them into the three metrics reported
side-by-side: normalised entropy, participation ratio, kurtosis. Zhang et al.
(2025) report entropy and kurtosis; we add participation ratio ("effective number
of edges"). Agreement across the three is the robust result; we never blend them
into a weighted score, and we never optimise them (sparsity is the train signal).

Directions (Zhang et al. 2025 report RL fine-tuning moving these toward diffuse):
    entropy            lower  = more concentrated
    participation_ratio lower = more concentrated (fewer effective edges)
    kurtosis           higher = more concentrated (peakier)
"""

from __future__ import annotations

import torch


def concentration_panel(scores: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    scores = scores.detach().float().abs()
    total = scores.sum()
    if total <= eps:
        n = scores.numel()
        return {"normalized_entropy": 1.0, "participation_ratio": float(n), "kurtosis": 0.0}

    p = scores / total
    n = scores.numel()

    entropy = -(p * (p + eps).log()).sum()
    normalized_entropy = (entropy / torch.log(torch.tensor(float(n)))).item()

    participation_ratio = (1.0 / (p * p).sum()).item()

    # Kurtosis is scale-invariant in theory, but an *additive* eps in the
    # denominator is not: when the raw attribution scores are tiny (e.g. ~1e-3 for
    # a sharply pruned circuit) var**2 underflows below a fixed eps and kurtosis is
    # wrongly crushed toward 0. Normalise to unit mean first so the moments are O(1)
    # and eps is genuinely negligible; treat an (essentially) constant vector -- the
    # maximally-diffuse case -- as kurtosis 0.
    s = scores / (scores.mean() + eps)
    var = s.var(unbiased=False)
    kurtosis = 0.0 if var <= eps else (((s - s.mean()) ** 4).mean() / (var ** 2)).item()

    return {
        "normalized_entropy": normalized_entropy,
        "participation_ratio": participation_ratio,
        "kurtosis": kurtosis,
    }
