#!/usr/bin/env python3
"""Completeness/necessity for every cell with a saved circuit.

Wang et al.'s second criterion: the circuit should carry the behavior -- knocking it OUT of the
full graph should destroy the task where an equal-size RANDOM knockout does not. Generalizes the
ioi/gpt2 pilot (circuit-cut 0.46 vs random-cut 0.73) to the whole grid. Corrupted-activation
knockout (the repo standard), 3 random seeds.
-> results/transfer/necessity_all.json
"""
import json

import torch

from ilm.data.dataset import PairCollator
from ilm.training.setup import load_tokenizer
from ilm.discovery.pipeline import set_circuit
from scripts._task_eval import (ROOT, MODELS, TASKS, MPATH, condensed_ckpt, load_condensed,
                                kept_edges, make_ds, task_acc)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for task in TASKS:
        for model in MODELS:
            cell = f"{task}/{model}"
            if condensed_ckpt(task, model) is None:
                continue
            try:
                tok = load_tokenizer(str(ROOT / MPATH[model]))
                coll = PairCollator(tok.pad_token_id)
                ds = make_ds(task, model, tok, split="test")
                m = load_condensed(task, model, device)
                circ = kept_edges(m)
                E, k = len(m.edges), len(circ)
                all_e = set(range(E))
                m.gate.log_alpha.data.fill_(20.0)
                full = task_acc(m, task, ds, coll, device)
                set_circuit(m, all_e - circ)
                cut = task_acc(m, task, ds, coll, device)
                rnd = []
                for s in range(3):
                    g = torch.Generator().manual_seed(s)
                    rs = set(torch.randperm(E, generator=g)[:k].tolist())
                    set_circuit(m, all_e - rs)
                    rnd.append(round(task_acc(m, task, ds, coll, device), 3))
                out[cell] = {"n_edges": k, "graph_edges": E, "full_acc": round(full, 3),
                             "circuit_cut_acc": round(cut, 3), "random_cut_accs": rnd,
                             "random_cut_mean": round(sum(rnd) / 3, 3)}
                print(f"{cell:<22} full={full:.2f} cut-circuit({k}e)={cut:.2f} "
                      f"cut-random={sum(rnd)/3:.2f}", flush=True)
                del m; torch.cuda.empty_cache()
            except Exception as e:
                print(f"{cell:<22} ERROR {type(e).__name__}: {e}", flush=True)
                out[cell] = {"error": str(e)}
    import os
    tag = os.environ.get("OUT_TAG", "")
    dst = ROOT / f"results/transfer/necessity_all{tag}.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nNECESSITY_ALL_DONE -> {dst}")


if __name__ == "__main__":
    main()
