#!/usr/bin/env python3
"""Build a FRESH C1 starting checkpoint: LoRA attached but NEVER trained, gates all open
-- no Stage 1 joint training at all. This is the way to produce a C1 starting
point: attach LoRA+gates fresh, then hand straight to
scripts/prune_heal_gradual.py --adaptive, which does 100% of the reshaping and pruning
with its own safety machinery (backtracking, capability gate, self-calibration).

Why: a controlled warm-start-vs-fresh-start comparison across 4 (task, model) pairs found
Stage 1's joint gate+weight training has no equivalent of Stage 2's safety net, so a long
or unmonitored Stage-1 run can silently bake in collateral drift that eats into Stage 2's
own safety margin -- confirmed directly (IOI: warm-start got stuck 142 edges, never cut
further, vs fresh-start reaching 33 edges at better accuracy; docstring/Qwen: warm-start's
C1 was outright broken, 0.04 test accuracy, vs fresh-start's clean 69e@0.98). Skipping
Stage 1 for C1 removes that failure mode entirely and simplifies the method to one stage.

C0 (frozen-weights, gates-only) is UNCHANGED by this -- it's already safe by construction
(weights can't drift if they never move) and still goes through Stage 1 (scripts/train.py
--freeze-weights) as before.

    python scripts/build_fresh_checkpoint.py --out-dir checkpoints/ioi_fresh \
        --model-path models/gpt2 --task ioi --graph-granularity head --max-length 64
"""
import argparse
import json
from pathlib import Path

import torch

from ilm.config import TrainConfig
from ilm.training.setup import build_masked_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model-path", required=True)
    p.add_argument("--task", required=True, choices=["ioi", "gt", "doc", "induction", "agreement", "gsm8k"])
    p.add_argument("--graph-granularity", default="block", choices=["block", "head"],
                   help="edge-graph granularity: 'head' (attention heads, used for the paper's "
                        "grid) or 'block' (whole attention/MLP blocks).")
    p.add_argument("--max-length", type=int, default=64)
    args = p.parse_args()

    cfg = TrainConfig(model_path=args.model_path, task=args.task, max_length=args.max_length,
                      graph_granularity=args.graph_granularity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_masked_model(cfg, device)  # fresh LoRA (near-zero init), gates ~all open, NO training

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.peft_model.save_pretrained(str(args.out_dir / "lora"))
    torch.save(model.gate.state_dict(), args.out_dir / "gates.pt")
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({
            "model_path": args.model_path, "task": args.task,
            "graph_granularity": args.graph_granularity, "max_length": args.max_length,
            "gt_mode": "generative" if args.task == "gt" else None,
            "note": "FRESH C1 checkpoint -- LoRA attached but NEVER trained, no Stage 1. "
                    "Feed directly into prune_heal_gradual.py --adaptive.",
        }, f, indent=2)
    print(f"Wrote fresh (untrained) C1 checkpoint -> {args.out_dir}")
    print(f"edges: {len(model.edges)}, gate mean open: {model.gate.hard_mask().float().mean().item():.3f}")


if __name__ == "__main__":
    main()
