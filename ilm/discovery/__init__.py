"""Frozen-model circuit-discovery baselines on the SAME ruler as our condensation arms.

The paper's baseline is the field's paradigm -- discover a behavior's circuit in the FROZEN
model -- instantiated by a family of methods that all (a) rank edges by importance, then
(b) get evaluated by an identical knee curve (accuracy + faithfulness vs #edges). Our method
(C1) differs by RESHAPING the weights so a smaller faithful circuit exists; C0/C0.5 are the
frozen members of this same family. Every method here shares the contrastive attribution
metric (make_logit_diff_metric with foil_ids), so comparisons are apples-to-apples.

See plans/2026-07-06-contrastive-metric-and-discovery-baselines.md and QA_V2 (2026-07-06).
"""
from .methods import (
    DiscoveryMethod, EAP, EAPIG, EAPGP, ActivationPatching,
    RandomRank, AntiRank, REGISTRY,
)
from .pipeline import discovery_knee, set_circuit, circuit_accuracy
from .acdc import acdc_circuit, acdc_knee

__all__ = [
    "DiscoveryMethod", "EAP", "EAPIG", "EAPGP", "ActivationPatching",
    "RandomRank", "AntiRank", "REGISTRY",
    "discovery_knee", "set_circuit", "circuit_accuracy",
    "acdc_circuit", "acdc_knee",
]
