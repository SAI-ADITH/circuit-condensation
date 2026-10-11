#!/usr/bin/env python3
"""Exhaustive minimality (per-edge necessity) for every cell with a saved circuit (2026-07-10).

Wang et al.'s third circuit criterion: MINIMAL -- every component earns its place. For each edge e
in the circuit: remove just e, measure the task-accuracy drop; e is necessary if drop >= TOL (0.05).
Comparison: the frozen EAP-IG circuit at MATCHED accuracy. Frozen circuits > SAMPLE_CAP edges get a
random-sample estimate (flagged 'sampled') -- itself part of the point: the diffuse circuit is too
big to check exhaustively.

  python scripts/run_minimality_all.py --task ioi     (one task per job to parallelize)
-> results/transfer/minimality_all_<task>.json
"""
import argparse
import json
import random

import torch

from ilm.data.dataset import PairCollator
from ilm.training.setup import load_tokenizer
from ilm.discovery import EAPIG
from ilm.discovery.pipeline import set_circuit
from scripts._task_eval import (ROOT, MODELS, MPATH, condensed_ckpt, load_condensed,
                                load_frozen, kept_edges, make_ds, task_acc)

TOL = 0.05
SAMPLE_CAP = 120     # exhaustive up to this many edges; beyond it, sample this many
NB = 12              # eval batches per ablation (~100 examples)


def minimality(m, circuit, task, ds, coll, device, tag):
    circuit = sorted(circuit)
    set_circuit(m, set(circuit))
    base = task_acc(m, task, ds, coll, device, nb=NB)
    edges = circuit
    sampled = len(circuit) > SAMPLE_CAP
    if sampled:
        edges = random.Random(0).sample(circuit, SAMPLE_CAP)
    drops = []
    for e in edges:
        set_circuit(m, set(c for c in circuit if c != e))
        drops.append(round(base - task_acc(m, task, ds, coll, device, nb=NB), 4))
    nec = sum(d >= TOL for d in drops)
    return {"n_edges": len(circuit), "tested": len(edges), "sampled": sampled,
            "base_acc": round(base, 3), "n_necessary": nec,
            "pct_necessary": round(100 * nec / max(len(edges), 1), 1),
            "mean_drop": round(sum(drops) / max(len(drops), 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for model in MODELS:
        cell = f"{a.task}/{model}"
        if condensed_ckpt(a.task, model) is None:
            print(f"{cell}: no circuit, skip", flush=True); continue
        try:
            tok = load_tokenizer(str(ROOT / MPATH[model]))
            coll = PairCollator(tok.pad_token_id)
            ds = make_ds(a.task, model, tok, split="test")

            m = load_condensed(a.task, model, device)
            cond = minimality(m, kept_edges(m), a.task, ds, coll, device, "condensed")
            del m; torch.cuda.empty_cache()

            m = load_frozen(a.task, model, device)
            order = torch.argsort(EAPIG(n_steps=5).rank_edges(m, ds, coll, device, rank_batches=8),
                                  descending=True).tolist()
            E = len(m.edges)
            chosen = order[:E]
            for k in [8, 16, 32, 64, 128, 256, 512, E]:
                if k > E:
                    k = E
                set_circuit(m, set(order[:k]))
                if task_acc(m, a.task, ds, coll, device, nb=NB) >= cond["base_acc"] - 0.03:
                    chosen = order[:k]; break
            froz = minimality(m, chosen, a.task, ds, coll, device, "frozen")
            del m; torch.cuda.empty_cache()

            out[cell] = {"condensed": cond, "frozen": froz}
            print(f"{cell:<22} condensed {cond['pct_necessary']:>5}% of {cond['n_edges']:<4} | "
                  f"frozen {froz['pct_necessary']:>5}% of {froz['n_edges']}"
                  f"{' (sampled)' if froz['sampled'] else ''}", flush=True)
        except Exception as e:
            print(f"{cell:<22} ERROR {type(e).__name__}: {e}", flush=True)
            out[cell] = {"error": str(e)}
    import os
    tag = os.environ.get("OUT_TAG", "")
    dst = ROOT / f"results/transfer/minimality_all_{a.task}{tag}.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nMINIMALITY_ALL_DONE ({a.task}) -> {dst}")


if __name__ == "__main__":
    main()
