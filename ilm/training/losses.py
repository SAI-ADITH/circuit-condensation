"""Loss terms of the preserve-and-sparsify objective.

    L = alpha * KL(original || masked)   # sparse circuit reproduces the original
      + delta * KL(original || full)     # full model stays equal to the original
      + lambda * L0(gates)               # use as few edges as possible

All first-order; no double-backprop. There is NO ground-truth task loss -- we hold
capability fixed (via the KL-to-original terms) and only push sparsity. ``L0`` is
the expected-L0 of the edge gates (the *only* concentration signal we optimise);
EAP concentration is evaluation-only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean token KL( teacher || student ); teacher is detached (the target).

    ``mask`` (b, s) selects real (non-pad) positions.
    """
    teacher = F.log_softmax(teacher_logits.detach(), dim=-1)
    student = F.log_softmax(student_logits, dim=-1)
    # KL(teacher || student) = sum_p exp(teacher) * (teacher - student)
    kl = (teacher.exp() * (teacher - student)).sum(dim=-1)  # (b, s)
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp(min=1.0)


def fraction_open(expected_l0: torch.Tensor, num_edges: int) -> torch.Tensor:
    """Expected fraction of edges left ON (the interpretable sparsity reading)."""
    return expected_l0 / max(num_edges, 1)
