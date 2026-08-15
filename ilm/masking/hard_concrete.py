"""Hard-concrete gates for differentiable L0 sparsity.

Each edge in the circuit graph carries one gate. The gate is a smooth stand-in
for a hard 0/1 switch: a learnable log-alpha parameter is stretched through a
sigmoid and clamped to [0, 1], so it usually sits very close to OFF or ON but
stays differentiable for gradient descent. The *expected* number of open gates
(expected-L0) is a closed-form function of log-alpha and is what we penalise to
push edges OFF.

Reference: Louizos et al. 2018, "Learning Sparse Neural Networks through L0
Regularization"; reused as the edge mask in Bhaskar et al. 2024 (Edge Pruning).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Stretch interval (gamma, zeta): the sigmoid output is mapped from [0, 1] to
# (gamma, zeta) before clamping, which lets the gate reach exact 0 / 1.
GAMMA = -0.1
ZETA = 1.1
BETA = 0.66  # temperature of the concrete distribution
EPS = 1e-6


class HardConcreteGate(nn.Module):
    """A vector of independent hard-concrete gates, one per edge.

    Args:
        num_gates: number of edges to gate.
        init_open_prob: probability a gate starts ON (gates open ~= keep edge).
            Initialising near 1.0 means training starts close to the full model
            and sparsity is ramped in via the L0 penalty.
    """

    def __init__(self, num_gates: int, init_open_prob: float = 0.95):
        super().__init__()
        self.num_gates = num_gates
        # Solve for the log_alpha whose deterministic gate value equals the
        # requested open probability, so the ramp starts from a faithful model.
        init = math.log(init_open_prob + EPS) - math.log(1.0 - init_open_prob + EPS)
        self.log_alpha = nn.Parameter(torch.full((num_gates,), float(init)))

    def sample(self, training: bool) -> torch.Tensor:
        """Return gate values in [0, 1], shape (num_gates,).

        Training: sample the concrete distribution (adds noise so the gate
        explores). Eval: use the deterministic value (no noise).
        """
        if training:
            u = torch.rand_like(self.log_alpha).clamp(EPS, 1 - EPS)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        s = s * (ZETA - GAMMA) + GAMMA
        return s.clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        """Expected number of open gates (the differentiable sparsity term)."""
        # P(gate > 0) under the hard-concrete distribution.
        prob_open = torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA))
        return prob_open.sum()

    @torch.no_grad()
    def hard_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Snap to a 0/1 circuit for evaluation: deterministic gate > threshold."""
        s = torch.sigmoid(self.log_alpha) * (ZETA - GAMMA) + GAMMA
        return (s.clamp(0.0, 1.0) > threshold).float()

    @torch.no_grad()
    def num_open(self, threshold: float = 0.5) -> int:
        return int(self.hard_mask(threshold).sum().item())
