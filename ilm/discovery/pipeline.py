"""Turn any edge ranking into a comparable knee curve (accuracy + faithfulness vs #edges).

Frozen weights only -- these are the discovery baselines. The SAME eval dispatch as
prune_heal_gradual.measure() is used (forced-choice per task) so a discovery-baseline knee is
directly comparable to the C0/C1 curves.
"""
from __future__ import annotations

import torch

from ilm.eval import circuit as circuit_eval
from .methods import BIN, DiscoveryMethod

_FLOOR = 8  # smallest circuit we bother evaluating (matches the grid's schedule floor)


def set_circuit(model, keep_idx) -> int:
    """Force exactly ``keep_idx`` edges ON (hard 0/1 gates); returns the ON count."""
    la = model.gate.log_alpha
    new = torch.full_like(la.data, -BIN)
    keep_idx = list(keep_idx)
    if keep_idx:
        idx = torch.tensor(sorted(keep_idx), device=la.device, dtype=torch.long)
        new[idx] = BIN
    la.data.copy_(new)
    return circuit_eval.circuit_size(model)


def circuit_accuracy(model, eval_ds, collator, device, task, *, batch_size=8, num_batches=25) -> float:
    """Task accuracy through the CURRENT masked circuit -- same dispatch as measure()."""
    if task == "agreement":
        return circuit_eval.agreement_accuracy(model, eval_ds, collator, device, masked=True,
                                               batch_size=batch_size, num_batches=num_batches)
    if task == "ioi":
        return circuit_eval.ioi_forced_choice_accuracy(model, eval_ds, collator, device, masked=True,
                                                       batch_size=batch_size, num_batches=num_batches)
    if task == "induction":
        return circuit_eval.induction_pool_accuracy(model, eval_ds, collator, device, masked=True,
                                                    batch_size=batch_size, num_batches=num_batches)
    # doc (and any single-token argmax task): teacher-forced answer-token accuracy.
    return circuit_eval.circuit_token_accuracy(model, eval_ds, collator, device,
                                               batch_size=batch_size, num_batches=num_batches)


def _budgets(n_edges: int, factor: float = 0.7, floor: int = _FLOOR) -> list[int]:
    """Geometric decay from n_edges down to floor -- same shape as the grid schedule."""
    out, x = [n_edges], n_edges
    while x > floor:
        x = max(int(round(x * factor)), floor)
        out.append(x)
    return sorted(set(out), reverse=True)


def discovery_knee(model, method: DiscoveryMethod, eval_ds, collator, device, task, *,
                   budgets: list[int] | None = None, rank_dataset=None,
                   rank_batches: int = 8, eval_batches: int = 25, batch_size: int = 8) -> dict:
    """Rank once (on the FULL frozen model), then sweep keep-budgets and record the knee.

    ``rank_dataset`` is what the method ranks on; the curve is measured on ``eval_ds``. Gate state
    is snapshotted and restored.

    Pass BOTH explicitly. ``rank_dataset`` defaults to ``eval_ds`` for backwards compatibility, but
    that default means the method picks its edges on the same examples it is then graded on --
    in-sample ranking, which flatters the baseline. C1 ranks on train and reports on test, so for
    any comparison against C1 pass ``rank_dataset=<train>`` and ``eval_ds=<test>``.
    (This docstring previously called eval_ds "the sealed split". It is not: eval is the
    controller's decision split; test is the sealed one.)
    """
    model.eval()
    la = model.gate.log_alpha
    saved = la.data.clone()
    try:
        la.data.fill_(BIN)  # rank on the full model (all edges ON)
        ranking = method.rank_edges(model, rank_dataset or eval_ds, collator, device,
                                    rank_batches=rank_batches, batch_size=batch_size)
        order = torch.argsort(ranking, descending=True).tolist()
        n_edges = len(model.edges)
        budgets = budgets or _budgets(n_edges)
        curve = []
        for n in budgets:
            n = min(n, n_edges)
            realised = set_circuit(model, order[:n])
            acc = circuit_accuracy(model, eval_ds, collator, device, task,
                                   batch_size=batch_size, num_batches=eval_batches)
            faith = circuit_eval.circuit_faithfulness(model, eval_ds, collator, device,
                                                      batch_size=batch_size)
            curve.append({"edges": realised, "target": n,
                          "acc": round(acc, 4), "faith": round(faith, 4)})
    finally:
        la.data.copy_(saved)
    return {"method": method.name, "task": task, "n_edges": len(model.edges),
            "curve": curve, "ranking": ranking.tolist()}
