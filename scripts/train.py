#!/usr/bin/env python3
"""Run ILM Edge-Pruning co-training on a 1B model.

    python scripts/train.py --model-path .../Llama-3.2-1B-Instruct \
        --data .../gsm8k_train.jsonl --eval-data .../gsm8k_eval.jsonl \
        --output-dir .../checkpoints/pilot_1b --num-steps 1000
"""

import argparse

import torch

from ilm.config import TrainConfig
from ilm.data.dataset import PairCollator, PairDataset, load_pairs
from ilm.data.ioi import IOIDataset, load_samples
from ilm.data.greater_than import GreaterThanDataset, load_samples as load_gt_samples
from ilm.data.docstring import DocstringDataset, load_samples as load_doc_samples
from ilm.data.induction import InductionDataset, load_samples as load_ind_samples
from ilm.data.agreement import AgreementDataset, load_samples as load_agr_samples
from ilm.eval.knee import plot_knee
from ilm.training.setup import build_masked_model, load_tokenizer
from ilm.training.trainer import Trainer


def build_datasets(cfg, tokenizer):
    """Load train/eval datasets for the configured task (both emit the same batch
    dict shape, so the trainer is task-agnostic)."""
    if cfg.task == "ioi":
        train = IOIDataset(load_samples(cfg.data_path), tokenizer, cfg.max_length)
        eval_ = IOIDataset(load_samples(cfg.eval_data_path), tokenizer, cfg.max_length)
    elif cfg.task == "gt":
        gen = cfg.gt_mode == "generative"
        train = GreaterThanDataset(load_gt_samples(cfg.data_path), tokenizer, cfg.max_length, gen)
        eval_ = GreaterThanDataset(load_gt_samples(cfg.eval_data_path), tokenizer, cfg.max_length, gen)
    elif cfg.task == "doc":
        train = DocstringDataset(load_doc_samples(cfg.data_path), tokenizer, cfg.max_length)
        eval_ = DocstringDataset(load_doc_samples(cfg.eval_data_path), tokenizer, cfg.max_length)
    elif cfg.task == "induction":
        train = InductionDataset(load_ind_samples(cfg.data_path), tokenizer, cfg.max_length)
        eval_ = InductionDataset(load_ind_samples(cfg.eval_data_path), tokenizer, cfg.max_length)
    elif cfg.task == "agreement":
        train = AgreementDataset(load_agr_samples(cfg.data_path), tokenizer, cfg.max_length)
        eval_ = AgreementDataset(load_agr_samples(cfg.eval_data_path), tokenizer, cfg.max_length)
    else:
        train = PairDataset(load_pairs(cfg.data_path), tokenizer, cfg.max_length)
        eval_ = PairDataset(load_pairs(cfg.eval_data_path), tokenizer, cfg.max_length)
    return train, eval_


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--eval-data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lambda-sparse", type=float, default=None,
                   help="Fixed per-edge L0 cost (swept across runs to trace the knee).")
    p.add_argument("--alpha-circuit", type=float, default=None,
                   help="Weight on KL(original || masked) — circuit faithful to original.")
    p.add_argument("--delta-preserve", type=float, default=None,
                   help="Weight on KL(original || full) — keep the model == original.")
    p.add_argument("--freeze-weights", action="store_true",
                   help="Baseline mode: lock the model, train ONLY the gates.")
    p.add_argument("--model-lr", type=float, default=None,
                   help="LoRA (weight) learning rate. Lower it to keep the capability "
                        "leash from slipping when the circuit can't fit (e.g. docstring).")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=None,
                   help="Eval batches per checkpoint (×batch_size = #problems scored).")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--task", default=None, choices=["gsm8k", "ioi", "gt", "doc", "induction", "agreement"],
                   help="gsm8k (generative), ioi (top-token name), gt (greater-than argmax-valid), or doc (docstring next-param, exact-token).")
    p.add_argument("--gt-mode", default=None, choices=["generative", "token"],
                   help="greater-than scoring: generative (gen+parse full year; robust to "
                        "Llama/Qwen) or token (GPT-2 single-token argmax-valid).")
    p.add_argument("--graph-granularity", default=None, choices=["block", "head"],
                   help="block (one attn writer/layer) or head (one writer/head).")
    # --- Dual-variable (Lagrangian) auto-tuning ---
    p.add_argument("--lambda-mode", default=None, choices=["fixed", "budget"],
                   help="fixed lambda, or 'budget' = auto-tune lambda to hit --target-frac-open.")
    p.add_argument("--target-frac-open", type=float, default=None,
                   help="budget mode: target fraction of edges left ON (graph-size invariant).")
    p.add_argument("--dual-lr-lambda", type=float, default=None,
                   help="dual-ascent step size on log-lambda (budget mode).")
    p.add_argument("--delta-mode", default=None, choices=["fixed", "tolerance"],
                   help="fixed delta, or 'tolerance' = auto-tune delta to hold KL(orig||full) <= --capability-tol.")
    p.add_argument("--capability-tol", type=float, default=None,
                   help="tolerance mode: max allowed KL(original || full) (capability you'll give up).")
    p.add_argument("--dual-lr-delta", type=float, default=None,
                   help="dual-ascent step size on log-delta (tolerance mode).")
    p.add_argument("--delta-min", type=float, default=None, help="clamp floor for the delta multiplier.")
    p.add_argument("--delta-max", type=float, default=None, help="clamp ceiling for the delta multiplier.")
    p.add_argument("--lambda-min", type=float, default=None, help="clamp floor for the lambda multiplier.")
    p.add_argument("--lambda-max", type=float, default=None, help="clamp ceiling for the lambda multiplier.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_config(args) -> TrainConfig:
    cfg = TrainConfig()
    overrides = {
        "model_path": args.model_path, "data_path": args.data,
        "eval_data_path": args.eval_data, "output_dir": args.output_dir,
        "num_steps": args.num_steps, "batch_size": args.batch_size,
        "eval_every": args.eval_every, "eval_batches": args.eval_batches,
        "max_length": args.max_length, "task": args.task, "seed": args.seed,
        "gt_mode": args.gt_mode, "graph_granularity": args.graph_granularity,
        "lambda_mode": args.lambda_mode, "target_frac_open": args.target_frac_open,
        "dual_lr_lambda": args.dual_lr_lambda, "delta_mode": args.delta_mode,
        "capability_tol": args.capability_tol, "dual_lr_delta": args.dual_lr_delta,
        "delta_min": args.delta_min, "delta_max": args.delta_max,
        "lambda_min": args.lambda_min, "lambda_max": args.lambda_max,
        "model_lr": args.model_lr,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.lambda_sparse is not None:
        cfg.loss.lambda_sparse = args.lambda_sparse
    if args.alpha_circuit is not None:
        cfg.loss.alpha_circuit = args.alpha_circuit
    if args.delta_preserve is not None:
        cfg.loss.delta_preserve = args.delta_preserve
    if args.freeze_weights:
        cfg.freeze_weights = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {cfg.model_path}")

    tokenizer = load_tokenizer(cfg.model_path)
    masked_model = build_masked_model(cfg, device)
    print(f"Circuit graph: {masked_model.graph.num_layers} layers, "
          f"{masked_model.graph.num_edges} edges")

    train_ds, eval_ds = build_datasets(cfg, tokenizer)
    collator = PairCollator(tokenizer.pad_token_id)
    print(f"Task: {cfg.task} | train: {len(train_ds)} | eval: {len(eval_ds)}")

    trainer = Trainer(cfg, masked_model, train_ds, eval_ds, collator, device, tokenizer)
    trainer.train()

    knee_path = f"{cfg.output_dir}/knee_curve.png"
    plot_knee(trainer.eval_log, knee_path)
    print(f"Wrote knee curve -> {knee_path}")


if __name__ == "__main__":
    main()
