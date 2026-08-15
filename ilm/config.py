"""Configuration dataclasses for ILM training and evaluation.

Defaults are the doc 5 §2 starting hyper-parameters for the 1B pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class LossConfig:
    # PRESERVE-AND-SPARSIFY objective (2026-06-23 pivot). The model already does the
    # task; we ONLY (a) keep it equal to its frozen original self and (b) shrink the
    # faithful circuit. There is NO ground-truth "get it right" loss -- that was an
    # "improve capability" signal pushed through the shrinking circuit, and it was
    # what corrupted the shared weights (40% -> 3% collapse).
    #   L = alpha * KL(original || masked)   # sparse circuit must say what original said
    #     + delta * KL(original || full)     # model itself must stay equal to original
    #     + lambda * L0                      # use as few edges as possible
    alpha_circuit: float = 1.0    # circuit faithful to the ORIGINAL model
    delta_preserve: float = 1.0   # full model preserved == ORIGINAL (capability leash)
    # Sparsity = standard Edge-Pruning LINEAR L0 cost per open edge (expected-L0). Flat
    # per-edge price -> each gate commits to 0/1 (binarises). lambda is FIXED per run
    # (no controller); the knee comes from SWEEPING lambda (small=large circuit).
    lambda_sparse: float = 0.1    # per-edge L0 cost (swept across runs)


@dataclass
class TrainConfig:
    model_path: str = "models/Llama-3.2-1B-Instruct"
    data_path: str = "data/gsm8k_train.jsonl"
    output_dir: str = "checkpoints/pilot_1b"

    task: str = "gsm8k"      # "gsm8k" (generative number match) or "ioi" (top-token name)
    gt_mode: str = "generative"  # greater-than scoring: "generative" (cut at "...to the
    # year ", generate+parse the full year — robust to Llama/Qwen number tokenization)
    # or "token" (GPT-2 single-token argmax-valid, Hanna's original).
    graph_granularity: str = "block"  # "block" (attn block writers) or "head" (per-head writers)
    max_length: int = 1024   # room for the full chain-of-thought solution target
    batch_size: int = 8
    num_steps: int = 1000
    model_lr: float = 1e-4
    mask_lr: float = 1e-2
    freeze_weights: bool = False   # baseline mode: lock the model, train ONLY the gates
    init_open_prob: float = 0.95
    sparsity_warmup_frac: float = 0.5   # fraction of steps to ramp lambda (or the budget) over
    grad_clip: float = 1.0
    seed: int = 0

    # --- Dual-variable (Lagrangian) auto-tuning of the penalty weights -------------
    # Instead of a FIXED lambda/delta, optionally treat them as Lagrange multipliers
    # updated by DUAL ASCENT so the run HITS a target you specify (constrained optim):
    #   lambda_mode="budget":   you set target_frac_open (edge budget); lambda auto-found.
    #   delta_mode="tolerance": you set capability_tol (max KL(orig||full)); delta auto-found.
    # The multiplier is updated multiplicatively in log-space (exponentiated gradient):
    #   log_mult += dual_lr * (constraint_value - target)
    # which keeps it positive and scale-free, so an edge BUDGET is graph-size invariant
    # (the SAME target frac works for a 325-edge or a 2041-edge graph -- fixes the
    # head-vs-block mismatch where a fixed lambda priced ~6x harder on the bigger graph).
    lambda_mode: str = "fixed"        # "fixed" | "budget"
    target_frac_open: float = 0.10    # budget: target fraction of edges left ON
    dual_lr_lambda: float = 0.05      # dual-ascent step on log-lambda
    lambda_min: float = 1e-5          # clamp range for the lambda multiplier
    lambda_max: float = 10.0
    delta_mode: str = "fixed"         # "fixed" | "tolerance"
    capability_tol: float = 0.02      # tolerance: max allowed KL(orig||full)
    dual_lr_delta: float = 0.05       # dual-ascent step on log-delta
    delta_min: float = 1e-3           # clamp range for the delta multiplier
    delta_max: float = 1e3

    eval_every: int = 100
    eval_data_path: str = "data/gsm8k_eval.jsonl"
    eval_batches: int = 8
    eval_max_new_tokens: int = 512   # generation budget for generative_accuracy

    dtype: str = "bfloat16"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> dict:
        return asdict(self)
