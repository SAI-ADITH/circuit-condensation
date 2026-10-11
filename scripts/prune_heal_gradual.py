#!/usr/bin/env python3
"""Gradual prune-and-heal: shrink a circuit in SMALL steps, healing between each.

The one-shot probe (scripts/prune_heal_probe.py) showed: healing recovers capability,
but cutting 142->35 edges in ONE shock either diverges (no leash) or only matches the
joint-loss knee (with leash). This script tests the fix the result pointed to: prune
GRADUALLY. Each round removes a small fraction of the lowest-importance ON edges, then
does a short leashed heal so the weights CO-ADAPT to the smaller circuit before the next
cut. Edge importance is RE-RANKED (EAP) each round under the current circuit, and the
LoRA is NOT reset between rounds (the reshaping accumulates) -- the opposite of the
one-shot probe, which restored LoRA every time.

Hypothesis: gradual + leashed heal should reach SMALLER edge counts at higher accuracy
than the one-shot, and ideally BEAT the joint-loss knee (35e->0.75, 16e->0.688) it only
tied before.

Records accuracy / faithfulness / full-capability at every scheduled edge count; flags
the milestones {35,16,8} for direct comparison to the one-shot + knee. Heal on the TRAIN
split, evaluate on the EVAL split.

    python scripts/prune_heal_gradual.py \
        --checkpoint checkpoints/ioi_head_c1_d20/lam_0.002 \
        --schedule 142,100,70,50,35,24,16,11,8 \
        --heal-steps 120 --heal-lr 1e-4 --heal-delta 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from itertools import cycle
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from ilm.data.ioi import IOIDataset, load_samples
from ilm.data.greater_than import GreaterThanDataset, load_samples as load_gt_samples
from ilm.data.docstring import DocstringDataset, load_samples as load_doc_samples
from ilm.data.induction import InductionDataset, load_samples as load_ind_samples
from ilm.data.agreement import AgreementDataset, load_samples as load_agr_samples
from ilm.data.dataset import PairCollator
from ilm.eval import circuit as circuit_eval
from ilm.eval.eap import concentration_panel
from ilm.masking.edge_masked_model import EdgeMaskedTransformer
from ilm.training.losses import kl_loss
from ilm.training.setup import DTYPES, load_tokenizer, split_parameters

def _example_keys(path: str) -> set:
    """Stable per-example identity for overlap checks, independent of task schema:
    try common prompt fields first, fall back to the whole row (order-independent)."""
    keys = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for field in ("clean_prompt", "prompt", "clean_question", "question", "text"):
            if field in row:
                keys.add(row[field])
                break
        else:
            keys.add(json.dumps(row, sort_keys=True))
    return keys


def assert_disjoint_splits(train_path: str, eval_path: str, test_path: str) -> None:
    """MANDATORY split hygiene: the controller's accept/reject decisions must never see the
    same examples as the sealed test split (otherwise re-rolling until eval passes would leak
    test information), and eval/test must not silently be training data.
    Raises if train/eval/test overlap on the same example."""
    train_k, eval_k, test_k = (_example_keys(p) for p in (train_path, eval_path, test_path))
    te, tt, et = train_k & eval_k, train_k & test_k, eval_k & test_k
    if te or tt or et:
        raise ValueError(
            f"train/eval/test split overlap detected -- refusing to run with leaked data.\n"
            f"  train={train_path} eval={eval_path} test={test_path}\n"
            f"  train&eval overlap={len(te)} train&test overlap={len(tt)} eval&test overlap={len(et)}\n"
            f"  Every split must be disjoint: the controller's accept/reject decisions (eval) must "
            f"never see the sealed test examples, and neither may repeat training data.")
    print(f"[split-check] OK: train({len(train_k)}) / eval({len(eval_k)}) / test({len(test_k)}) "
          f"are pairwise disjoint.", flush=True)


def assert_probe_disjoint(probe_path: str, eval_path: str, test_path: str) -> None:
    """The capability probe gates accept/reject, so it is part of the controller's decision path.

    A probe built from test prompts would condition the controller's per-round decisions on the
    sealed test set, and assert_disjoint_splits() only sees train/eval/test. A probe drawn from
    train is fine (the healer trains on it anyway); a probe drawn from eval or test is not.
    """
    if not probe_path:
        return
    probe_k = _example_keys(probe_path)
    pe, pt = probe_k & _example_keys(eval_path), probe_k & _example_keys(test_path)
    if pe or pt:
        raise ValueError(
            f"capability-probe corpus overlaps a decision/sealed split -- refusing to run.\n"
            f"  probe={probe_path} eval={eval_path} test={test_path}\n"
            f"  probe&eval overlap={len(pe)}  probe&test overlap={len(pt)}\n"
            f"  The probe drives the capability gate, so these prompts would condition the "
            f"controller's accept/reject on data it must never see. Rebuild with "
            f"scripts/build_probe_corpus.py (which now refuses to emit a leaky corpus).")
    print(f"[probe-check] OK: probe({len(probe_k)}) disjoint from eval and sealed test.", flush=True)


JOINT_LOSS_KNEE = {142: 0.844, 35: 0.750, 16: 0.688, 5: 0.625}
MILESTONES = {35, 16, 8}
BIN = 10.0  # log_alpha magnitude that snaps a deterministic gate to a clean 0/1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/ioi_head_c1_d20/lam_0.002")
    p.add_argument("--train-data", default="data/ioi/gpt2/train.jsonl",
                   help="data layout: data/<task_dir>/<model>/{train,eval,test}.jsonl "
                        "(task_dir is 'docstring' for --task doc, else same as --task).")
    p.add_argument("--eval-data", default="data/ioi/gpt2/eval.jsonl")
    p.add_argument("--test-data", default=None, required=True,
                   help="SEALED test split, REQUIRED. The controller uses eval for all its decisions; the "
                        "FINAL reported faithfulness + edit-eval are computed on THIS split (never seen by "
                        "the controller), so the headline numbers aren't selected on the set they're graded "
                        "on. Must be disjoint from --train-data and --eval-data (checked at startup).")
    p.add_argument("--model-path", default=None)
    p.add_argument("--task", default=None, choices=[None, "ioi", "gt", "doc", "induction", "agreement"],
                   help="default: from checkpoint config.json")
    p.add_argument("--start", default="trained", choices=["trained", "full"],
                   help="start the descent from the checkpoint's trained circuit, or from "
                        "ALL edges ON (the full model) — use 'full' for tasks whose joint-loss "
                        "circuit thrashed (docstring).")
    p.add_argument("--freeze-weights", action="store_true",
                   help="C0 arm: run the FULL adaptive controller (EAP-IG ranking, cut, "
                        "backtracking, cap-gate) but never let LoRA weights move -- heal() "
                        "becomes a no-op. Isolates the effect of reshaping (C1 vs C0, same "
                        "search machinery).")
    p.add_argument("--schedule", default="142,100,70,50,35,24,16,11,8",
                   help="descending edge counts to step through (first <= start ON count)")
    p.add_argument("--rerank", default="each", choices=["each", "once"],
                   help="re-rank edge importance after every heal (each) or rank once at the start")
    p.add_argument("--heal-steps", type=int, default=500, help="heal steps PER round (a CEILING in "
                   "adaptive mode — early-stop usually stops well before this)")
    p.add_argument("--heal-lr", type=float, default=1e-4)
    p.add_argument("--heal-alpha", type=float, default=1.0,
                   help="weight on the circuit-fitting term α·KL(orig‖masked) during heal "
                        "(held at 1.0 historically; the untested circuit-pressure lever).")
    p.add_argument("--heal-mode", default="standard", choices=["standard", "mixed", "taskce"],
                   help="'mixed' = MIXED-CUT-EDGE heal: "
                        "one objective KL(out‖ref) where each step feeds CORRUPTED (prob --mix-p) "
                        "or CLEAN cut-edge activations. Dissolves the α/δ ratio and preserves "
                        "full_cap for free; ignores --heal-delta and the leash machinery. "
                        "'taskce' = the C1-taskCE ABLATION: heal on TASK cross-entropy against the "
                        "answer token instead of KL(orig‖masked). Ablates the design choice 'why heal "
                        "toward the frozen original rather than fine-tune on the task' -- expected to "
                        "drift capability / stop matching the original. No leash/ref term.")
    p.add_argument("--mix-p", type=float, default=0.5,
                   help="mixed heal: fraction of steps trained on the CORRUPTED (faithfulness) "
                        "case; the rest on CLEAN (full-model preservation). 0.5 = equal.")
    p.add_argument("--adaptive-mixed-switch", action="store_true",
                   help="CONDITIONAL-MIXED SWITCH: in "
                        "--adaptive mode, heal STANDARD by default, but when a round is rejected "
                        "SOLELY because full_cap fell below its floor (circuit_acc + cap_ratio both "
                        "fine = full_cap is the binding constraint), re-heal the SAME cut with the "
                        "mixed heal (--mix-p) from the same last-good weights before backtracking. If "
                        "that clears every gate, LATCH mixed on for the rest of the run. Fires only "
                        "where full_cap binds -- the exact regime where mixed is well-behaved -- so "
                        "cells that run away under global mixed never enter it. No-op with "
                        "--heal-mode mixed (already mixed) or --adaptive-fullcap-tol 0 (no gate).")
    p.add_argument("--heal-delta", type=float, default=10.0,
                   help="leash weight during heal. With --heal-cap-tol=0 this is FIXED; with "
                        "--heal-cap-tol>0 it is only the INITIAL delta (then auto-tuned).")
    p.add_argument("--heal-cap-tol", type=float, default=0.0,
                   help="δ-TOLERANCE: capability budget = max allowed KL(orig||full) during heal. "
                        "0 = fixed delta; >0 = auto-tune delta by dual ascent to hold drift at this "
                        "budget (loosen when there's slack, tighten when the model drifts).")
    p.add_argument("--heal-dual-lr", type=float, default=0.05, help="dual-ascent step on log-delta")
    p.add_argument("--heal-acc-floor", type=float, default=0.0,
                   help="ACCURACY-GATED δ (the fix): allowed full-model accuracy drop. >0 auto-tunes "
                        "delta on the REAL capability drop (not the KL proxy); pair with a low "
                        "--heal-delta-max. Takes precedence over --heal-cap-tol.")
    p.add_argument("--heal-acc-every", type=int, default=30, help="measure full-model acc every N heal steps")
    p.add_argument("--heal-circ-floor", type=float, default=0.0,
                   help="Q2: gate δ on the CIRCUIT's accuracy — raise δ toward this target when the "
                        "masked circuit underperforms (tests whether a strong leash auto-helps the circuit).")
    p.add_argument("--retry-floor", type=float, default=0.0,
                   help="Q3 (adaptive prune-rate): if circuit-acc after a round is below this, do extra "
                        "heal rounds at the SAME edge count (slow the schedule) before the next cut.")
    p.add_argument("--retry-max", type=int, default=0, help="max extra heal rounds per cut (Q3)")
    p.add_argument("--delta-ramp", default=None,
                   help="DYNAMIC δ ceiling test: 'lo,hi' — ramp δ from lo (full circuit) to hi (floor) "
                        "linearly with the pruned fraction, so the leash tightens as the circuit shrinks "
                        "(motivated by the optimum moving up toward the floor). Overrides --heal-delta per round.")
    p.add_argument("--heal-delta-max", type=float, default=1e3, help="clamp ceiling for delta (use ~15 for auto-δ)")
    p.add_argument("--heal-batch", type=int, default=16)
    p.add_argument("--rank-batches", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lazy-heal", action="store_true",
                   help="heal only on rounds where the raw (unhealed) cut drops the circuit below "
                        "the accuracy floor. Skips wasted healing on cuts that already pass, which "
                        "avoids the capability drift that stops C1 early on drift-sensitive tasks.")
    p.add_argument("--heal-weight-decay", type=float, default=0.01,
                   help="AdamW weight decay on the LoRA params during healing. Decays the adapter "
                        "toward zero, i.e. toward the ORIGINAL model -- a structural drift leash.")
    p.add_argument("--ig-steps", type=int, default=5,
                   help="integrated-gradient points for EAP-IG ranking. Higher = better edge "
                        "ranking (better cut choices), linear cost in ranking time.")
    p.add_argument("--no-prune", action="store_true",
                   help="B1 distillation control: heal at the FULL mask each round, never cut. "
                        "Tests whether distillation alone concentrates (it should not). Works "
                        "standalone (fixed-schedule loop) or combined with --adaptive "
                        "(--adaptive-b1-rounds heal-only rounds at the full mask).")
    p.add_argument("--adaptive-b1-rounds", type=int, default=0,
                   help="B1 control under --adaptive --no-prune: number of heal-only rounds at "
                        "the full starting mask (never cutting). 0 (with --no-prune set) falls "
                        "back to 10. Ignored unless --adaptive --no-prune are both set.")
    p.add_argument("--rank-mode", default="eap", choices=["eap", "random", "anti"],
                   help="B2 control: edge ranking for pruning. eap=causal importance (default); "
                        "random=arbitrary subset; anti=keep the LEAST important edges. Respected "
                        "by both the fixed-schedule loop AND --adaptive.")
    p.add_argument("--adaptive", action="store_true",
                   help="ADAPTIVE schedule: ignore --schedule; auto-cut a fraction each round and "
                        "STOP at the smallest circuit still faithful (acc >= baseline - tol). Removes "
                        "the last manual knob (the schedule + stopping point).")
    p.add_argument("--adaptive-tol", type=float, default=0.05,
                   help="faithful = circuit_acc >= baseline_acc - this tolerance.")
    p.add_argument("--adaptive-fullcap-tol", type=float, default=0.05,
                   help="TASK-CAPABILITY gate: reject a round if the FULL model's accuracy on this "
                        "same task drops more than this below baseline (full_cap >= base_full - tol). "
                        "Same 0.05 tolerance as the circuit-acc floor -- both measure task accuracy vs "
                        "the same baseline. A no-op for frozen arms (full_cap can't move); it only "
                        "constrains C1, preventing a 'faithful circuit on a broken model' over-"
                        "specialisation (pilot: full_cap collapsed 0.80->0.21 while circuit_acc stayed "
                        "0.75). 0 = off.")
    p.add_argument("--adaptive-cut", type=float, default=0.30, help="initial relative cut per round.")
    p.add_argument("--adaptive-min-cut", type=float, default=0.05, help="floor on the relative cut (gentleness near the cliff).")
    p.add_argument("--adaptive-min", type=int, default=4, help="never shrink below this many edges.")
    # ---- conservative accept rule --------------------------------------------------------
    # A POINT estimate on 200 eval examples compared against the floor is consulted ~32 times
    # per run, keeping the best retry. Selecting the max of noisy estimates biases the winner
    # upward, so endpoints can miss the same tolerance on the sealed split. --accept-lcb requires the LOWER
    # end of a one-sided binomial confidence interval to clear the floor, so an accepted
    # circuit is conservative by construction. Pair it with a larger --eval-batches: noise
    # shrinks as 1/sqrt(n), and at n=200, p=0.75 the bound eats 0.051 -- the whole tolerance.
    p.add_argument("--accept-lcb", type=float, default=0.0,
                   help="z for a one-sided lower confidence bound in the accept test "
                        "(0 = off, point-estimate rule; 1.645 = one-sided 95%%).")
    # Per-round states, so the size/accuracy trajectory can be re-scored offline (Pareto
    # curves) or walked back to a larger circuit without re-running the cell. Only the last
    # K accepted states are kept: a full history runs 10-70MB per round per run.
    p.add_argument("--save-rounds", default=None,
                   help="directory to save the last K accepted (adapter, gates) states.")
    p.add_argument("--save-rounds-keep", type=int, default=6,
                   help="how many of the most recent accepted states to retain.")
    p.add_argument("--adaptive-reheal", type=int, default=3,
                   help="extra fresh heal attempts (each restarted from the last-GOOD weights, keeping "
                        "the best) to recover a below-floor cut before declaring the cliff. Escapes a "
                        "single unlucky heal / local minimum instead of quitting early.")
    p.add_argument("--adaptive-heal-buffer-frac", type=float, default=0.5,
                   help="SEED margin above the floor (as a fraction of the tolerance) used only for the "
                        "first rounds before we have a drop estimate. Default 0.5 = half the tolerance.")
    p.add_argument("--adaptive-buffer-beta", type=float, default=0.6,
                   help="smoothing for the auto-sized heal buffer: buffer = running average of the "
                        "measured accuracy-drop-per-cut (beta = weight on the old estimate). Bigger cuts "
                        "near the cliff -> bigger buffer automatically. Clipped to [0.5%%, tol].")
    p.add_argument("--adaptive-heal-buffer", type=float, default=0.0,
                   help="absolute manual override for the heal margin above the floor. 0 = use the "
                        "auto-sized (drop-tracking) buffer instead.")
    p.add_argument("--adaptive-prop", action="store_true",
                   help="margin-PROPORTIONAL cut: cut = clip(gain*(acc-floor), min_cut, max_cut) each "
                        "round instead of the fixed 30%%->halve ladder. Big cuts when accuracy has "
                        "headroom, gentle near the cliff — self-scaling, no fixed initial cut.")
    p.add_argument("--adaptive-max-cut", type=float, default=0.30, help="cap on the per-round cut (margin-prop).")
    p.add_argument("--adaptive-cut-decay", type=float, default=1.0,
                   help="ENFORCED gradual schedule (fixed-cut mode): multiply the cut by this each "
                        "accepted round (floored at min_cut), so cuts shrink over time even when "
                        "accuracy gives no warning (saturated tasks). 1.0 = off; 0.9 = gentle decay.")
    p.add_argument("--adaptive-prop-gain", type=float, default=0.0,
                   help="cut-per-unit-margin. 0 = auto (max_cut / tol, so cut hits max_cut at full headroom).")
    p.add_argument("--adaptive-early-every", type=int, default=40,
                   help="in adaptive mode, check circuit accuracy every N heal steps and STOP the "
                        "heal early once it reaches the faithful floor — `--heal-steps` becomes a "
                        "ceiling, not a fixed spend. Set 0 to always run the full budget.")
    p.add_argument("--adaptive-overshoot", type=int, default=4,
                   help="after the cliff, keep pruning this many MORE rounds (restarting from the best "
                        "faithful weights) to record how accuracy drops past the knee — the knee-curve "
                        "tail. The model is rolled back to the best faithful checkpoint before edit-eval.")
    p.add_argument("--diagnostic-sweep", action="store_true",
                   help="DIAGNOSTIC mode, not the real headline controller: keep pruning past where "
                        "--adaptive would stop, no retries/backtracking, recording circuit_acc/full_cap/"
                        "cap_ratio every round, until --diagnostic-sweep-cap-ratio or --diagnostic-acc-floor "
                        "trips. Mutually exclusive with --adaptive -- run separately, never used for real "
                        "C0/C1 headline numbers.")
    p.add_argument("--diagnostic-sweep-cap-ratio", type=float, default=1.2,
                   help="stop the diagnostic sweep once cap_ratio reaches this ceiling.")
    p.add_argument("--diagnostic-acc-floor", type=float, default=0.4,
                   help="safety stop: end the diagnostic sweep early if circuit_acc or full_cap drops "
                        "below this (real collapse, not informative for the tradeoff curve).")
    p.add_argument("--edit-eval", action="store_true",
                   help="C2 editing/unlearning: after condensing, ablate the circuit's edges on "
                        "the full model and report target-behavior drop + collateral LM loss vs a "
                        "random equal-size ablation (specificity + low-collateral demonstration).")
    p.add_argument("--capability-probe", action="store_true",
                   help="after condensing, probe GENERAL capability: perplexity of the full healed "
                        "model vs the original on UNRELATED text (--probe-data), so 'capability "
                        "preserved' becomes measured, not assumed.")
    p.add_argument("--probe-data", default=None,
                   help="jsonl of an UNRELATED task (uses its clean_prompt strings as the generic-text "
                        "corpus for the capability probe).")
    p.add_argument("--probe-every-round", action="store_true",
                   help="Drift shape: measure the general-capability probe (ppl ratio vs "
                        "original on --probe-data) after EVERY round, recorded per row. Decides whether "
                        "OOD damage arrives as a JUMP (reject-gate fixes it) or a slow RAMP (needs the "
                        "replay-mix leash).")
    p.add_argument("--cap-gate", type=float, default=0.0,
                   help="Capability-gated ACCEPT: a round is only accepted if the probe ppl "
                        "ratio <= this (e.g. 1.08) IN ADDITION to the accuracy floor. A capability failure "
                        "backtracks/re-rolls exactly like an accuracy failure. 0 = off, overridden by "
                        "--cap-gate-auto if set.")
    p.add_argument("--probe-max-length", type=int, default=0,
                   help="token cap for capability-probe passages. 0 = use the task's --max-length "
                        "(64), which TRUNCATES WikiText passages to a fraction of their length. "
                        "Set 256 so the gate reads real prose, not prose fragments.")
    p.add_argument("--cap-gate-batches", type=int, default=8,
                   help="probe batches for the per-round gate/probe measurement.")
    p.add_argument("--cap-gate-auto", action="store_true",
                   help="SELF-CALIBRATE the gate instead of requiring a pre-measured number (needed the "
                        "first time a NEW model/task runs, with no prior checkpoint to calibrate on). Before "
                        "the descent starts, the circuit is still at its STARTING point (uncut, by "
                        "definition healthy) -- bootstrap-resample --cap-gate-batches-sized subsets of the "
                        "probe corpus --cap-gate-boot times, take sigma(log-ratio), set the gate to "
                        "exp(3*sigma) against the true-healthy baseline of 1.0. Costs one extra forward pass "
                        "over the probe corpus (bootstrap resampling itself is free, CPU-only). Overrides "
                        "--cap-gate.")
    p.add_argument("--cap-gate-boot", type=int, default=30,
                   help="bootstrap resamples for --cap-gate-auto (free -- CPU-only over cached per-prompt CE).")
    p.add_argument("--cap-gate-calib-rounds", type=int, default=1,
                   help="on a DEGENERATE start (fresh/untrained checkpoint), how many initial "
                        "rounds to run gate-off and recalibrate from before locking in the "
                        "(averaged) gate for the rest of the run. 1 = calibrate from just the "
                        "first heal; >1 averages across that many, for a less noise-sensitive "
                        "estimate of what capability-probe noise looks like once training has "
                        "actually progressed a bit, not just the tiniest possible first nudge.")
    p.add_argument("--cap-gate-sigma-mult", type=float, default=3.0,
                   help="gate = exp(sigma_mult * sigma). Default 3.0 = a fixed ~0.13%% false-alarm "
                        "rate PER CHECK. A run makes many checks (every round/backtrack/cliff-probe, "
                        "often 20-30+), so the whole-RUN false-alarm rate is well above the nominal "
                        "per-check rate -- raise this (a Bonferroni-style correction) to hold the "
                        "run-level rate near the original target instead, e.g. ~3.6 for a 0.3%% "
                        "run-level target over ~20 checks. A named statistical convention, not an "
                        "arbitrary loosening -- pick and document deliberately.")
    p.add_argument("--cap-abort", type=float, default=0.0,
                   help="per-heal runaway abort: every --adaptive-early-every heal steps, check the probe "
                        "ratio on a few batches; if it exceeds this (e.g. 2.0), abort the heal early so a "
                        "runaway trajectory is caught in minutes, not after the full heal. 0 = off.")
    p.add_argument("--gamma-sweep", default=None,
                   help="WiSE-FT-style post-hoc rescue: comma list of gammas (e.g. '1.0,0.75,0.5,0.25,0.0'). "
                        "After the descent, scale the healed LoRA update by each gamma (gamma=0 == original "
                        "weights) and record circuit-acc + capability-probe ratio at each — tests whether "
                        "general-capability damage can be dialed away before the condensed circuit breaks.")
    p.add_argument("--save-healed", default=None,
                   help="directory to save the healed LoRA adapter + final gate mask (the descent otherwise "
                        "keeps weights only in memory).")
    p.add_argument("--measure-panel", action="store_true",
                   help="also record the EAP concentration panel (PR/entropy/kurtosis) at the "
                        "baseline and each condensed round.")
    p.add_argument("--panel-batches", type=int, default=8, help="batches averaged for the EAP panel")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_model(model_path, checkpoint, granularity, device) -> EdgeMaskedTransformer:
    dtype = DTYPES["bfloat16"] if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager")
    peft_model = PeftModel.from_pretrained(base, str(Path(checkpoint) / "lora"), is_trainable=True)
    masked = EdgeMaskedTransformer(peft_model, granularity=granularity).to(device)
    masked.gate.load_state_dict(torch.load(Path(checkpoint) / "gates.pt", map_location=device))
    return masked


def set_binary_mask(model, keep_idx) -> int:
    la = model.gate.log_alpha
    new = torch.full_like(la.data, -BIN)
    if len(keep_idx):
        idx = torch.tensor(sorted(keep_idx), device=la.device, dtype=torch.long)
        new[idx] = BIN
    la.data.copy_(new)
    return circuit_eval.circuit_size(model)


def snapshot_lora(model) -> list:
    """Clone the trainable LoRA weights so we can roll back after a divergent heal. The gate
    mask is reconstructed separately from the kept-edge set (set_binary_mask), so we only need
    the LoRA tensors here."""
    from ilm.training.setup import split_parameters
    lora_params, _ = split_parameters(model)
    return [p.detach().clone() for p in lora_params]


def restore_lora(model, snap) -> None:
    """Copy a snapshot back into the LoRA weights in place."""
    from ilm.training.setup import split_parameters
    lora_params, _ = split_parameters(model)
    with torch.no_grad():
        for p, s in zip(lora_params, snap):
            p.data.copy_(s)


def measure_panel(model, eval_ds, collator, device, panel_batches) -> dict:
    """EAP-IG concentration panel (PR / entropy / kurtosis) under the CURRENT mask — i.e.
    the condensed circuit, measured identically
    to scripts/task_finetune.py's 'diffuse' axis (both now use edge_attributions_ig, not
    plain single-point EAP) so the two remain directly comparable."""
    model.eval()
    loader = DataLoader(eval_ds, batch_size=8, collate_fn=collator)
    keys = ("normalized_entropy", "participation_ratio", "kurtosis")
    acc = {k: 0.0 for k in keys}
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= panel_batches:
            break
        corrupted = model.capture_contributions(
            batch["corr_ids"].to(device), batch["corr_mask"].to(device))
        metric_fn = circuit_eval.make_logit_diff_metric(
            batch["labels"].to(device),
            batch["foil_ids"].to(device) if "foil_ids" in batch else None)
        scores = model.edge_attributions_ig(
            batch["clean_ids"].to(device), batch["clean_mask"].to(device), corrupted, metric_fn,
            n_steps=5)
        panel = concentration_panel(scores)
        for k in keys:
            acc[k] += panel[k]
        n += 1
    return {k: round(acc[k] / max(n, 1), 4) for k in keys}


def eap_ranking(model, train_ds, collator, device, rank_batches, ig_steps: int = 5,
                seed: int | None = None) -> torch.Tensor:
    """Mean |EAP-IG attribution| per edge under the CURRENT circuit (mask already set).
    Uses integrated gradients (average over ig_steps points from fully-corrupted to the
    current shaped circuit) instead of a single gradient sample -- plain EAP's single point
    can badly under-score an edge if the gradient happens to be near-zero right there (a
    documented failure mode that gets worse at high sparsity, sometimes producing literally
    empty/broken circuits). Confirmed empirically on IOI/GPT-2 (rank-once, no training):
    EAP-IG >= plain EAP at every edge cutoff tested, with the largest gain (+8.5 points)
    right at the accuracy cliff. ig_steps=1 recovers plain EAP's single-point behavior at
    alpha=1 (not identical to the old edge_attributions() call, but the same idea) if ever
    needed for a cheaper ranking pass.

    SEEDED DATA DRAW. An unshuffled loader would rank on the SAME first `rank_batches` batches
    regardless of --seed, so seeds would vary healing noise only, and in the C0 arm (frozen
    weights = no healing) nothing at all. Data ORDER alone swings EAP-IG circuit size ~2x, so
    unshuffled ranking would undercount seed variance. Shuffling with a per-run generator
    makes --seed control the data draw, which is what "3 seeds" is taken to mean."""
    g = None
    if seed is not None:
        # ROUND-VARYING DRAW. A fixed `seed` re-shuffles to the SAME order on every call, so a
        # ~20-round descent would make every cut decision from one 64-example draw -- and EAP-IG
        # circuit size swings ~2x on data order alone, so the controller could ride whatever quirk that single draw had. Advancing a per-run call
        # counter means the descent sees ~rounds x rank_batches x 8 distinct examples instead,
        # at zero extra cost. Still fully deterministic given --seed.
        eap_ranking._call = getattr(eap_ranking, "_call", 0) + 1
        g = torch.Generator(); g.manual_seed(seed * 100003 + eap_ranking._call)
    loader = DataLoader(train_ds, batch_size=8, collate_fn=collator,
                        shuffle=seed is not None, generator=g)
    scores = torch.zeros(len(model.edges))
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= rank_batches:
            break
        corrupted = model.capture_contributions(
            batch["corr_ids"].to(device), batch["corr_mask"].to(device))
        metric_fn = circuit_eval.make_logit_diff_metric(
            batch["labels"].to(device),
            batch["foil_ids"].to(device) if "foil_ids" in batch else None)
        scores += model.edge_attributions_ig(
            batch["clean_ids"].to(device), batch["clean_mask"].to(device), corrupted, metric_fn,
            n_steps=ig_steps)
        n += 1
    return scores / max(n, 1)


@torch.no_grad()
def _lm_ce(model, eval_ds, collator, device, eb, nb):
    """Generic next-token cross-entropy on the clean prompts under the CURRENT mask —
    a task-agnostic 'collateral' probe (does ablating the circuit hurt general language?)."""
    import torch.nn.functional as F
    loader = DataLoader(eval_ds, batch_size=eb, collate_fn=collator)
    tot, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= nb:
            break
        cids = batch["clean_ids"].to(device); cmask = batch["clean_mask"].to(device)
        corr = model.capture_contributions(batch["corr_ids"].to(device), batch["corr_mask"].to(device))
        logits = model(cids, cmask, corrupted=corr, masked=True)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                             cids[:, 1:].reshape(-1), reduction="mean")
        tot += ce.item(); n += 1
    return tot / max(n, 1)




def capability_probe(model, prompts, tok, device, max_length, eb=8, nb=16):
    """GENERAL-capability probe: has healing damaged the model beyond the target task? Measures
    next-token cross-entropy / perplexity of the FULL model (all edges on) on UNRELATED text,
    with the LoRA ON (healed) vs OFF (original). A small delta = general capability preserved;
    a large delta = the healing collaterally hurt the model. Task-agnostic (reads raw prompts)."""
    import torch.nn.functional as F
    import contextlib, math
    model.eval()

    def ce_pass(adapter_on: bool) -> float:
        tot, n = 0.0, 0
        for i in range(0, min(len(prompts), nb * eb), eb):
            batch = prompts[i:i + eb]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device); am = enc.attention_mask.to(device)
            ctx = contextlib.nullcontext() if adapter_on else model.peft_model.disable_adapter()
            with torch.no_grad(), ctx:
                logits = model(ids, am, masked=False)          # full model, no ablation
            lp = logits[:, :-1, :].float(); tgt = ids[:, 1:]; m = am[:, 1:].bool()
            ce = F.cross_entropy(lp.reshape(-1, lp.size(-1)), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
            ce = (ce * m).sum() / m.sum().clamp(min=1)
            tot += ce.item(); n += 1
        return tot / max(n, 1)

    ce_o, ce_h = ce_pass(False), ce_pass(True)
    return {"probe_ce_orig": round(ce_o, 4), "probe_ce_healed": round(ce_h, 4),
            "probe_delta_ce": round(ce_h - ce_o, 4),
            "probe_ppl_orig": round(math.exp(ce_o), 3), "probe_ppl_healed": round(math.exp(ce_h), 3),
            "probe_ppl_ratio": round(math.exp(ce_h - ce_o), 4)}


def calibrate_cap_gate(model, prompts, tok, device, max_length, gate_n, n_boot=30, batch=8,
                       sigma_mult: float = 3.0) -> dict:
    """SELF-CALIBRATE the capability gate on THIS model/task, no prior run required. Call once,
    before any cutting, while the circuit is still at its starting (uncut, by-construction-healthy)
    state. Computes per-prompt CE once (one forward pass over the whole probe corpus, adapter on
    vs off), then bootstrap-resamples --cap-gate-boot subsets of size gate_n (matching the real
    per-round gate's sample size) -- resampling is pure CPU arithmetic over the cached per-prompt
    numbers, so more resamples cost nothing extra. sigma(log-ratio) at this healthy point estimates
    the gate's natural noise floor; the threshold is exp(sigma_mult*sigma) against the TRUE-healthy
    baseline of 1.0 (not against this run's own mean, which may not be exactly 1.0 even when
    healthy).

    sigma_mult (default 3.0, i.e. "3-sigma"): a fixed per-check false-alarm rate on ONE check
    (~0.13% one-sided). A single run doesn't check once, though -- it checks at every round,
    backtrack, and cliff-reheal probe (often 20-30+ checks per run), so the chance that AT LEAST
    ONE check trips on pure noise (not real drift) across the whole run is well above the nominal
    per-check rate -- the standard multiple-comparisons problem. To hold the RUN-level false-alarm
    rate near the original per-check target given M checks, raise sigma_mult so the per-check rate
    is ~target/M (a Bonferroni correction) -- e.g. target 0.3% over M=20 checks needs a per-check
    rate of ~0.015%, which is sigma_mult~3.6 rather than 3.0. This is a named, standard statistical
    technique, not an arbitrary loosening -- pick sigma_mult deliberately and document why."""
    import torch.nn.functional as F
    import contextlib, math, random
    model.eval()

    def per_prompt_ce(adapter_on: bool) -> list:
        out = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device); am = enc.attention_mask.to(device)
            ctx = contextlib.nullcontext() if adapter_on else model.peft_model.disable_adapter()
            with torch.no_grad(), ctx:
                logits = model(ids, am, masked=False)
            lp = logits[:, :-1, :].float(); tgt = ids[:, 1:]; m = am[:, 1:].bool()
            ce = F.cross_entropy(lp.reshape(-1, lp.size(-1)), tgt.reshape(-1),
                                 reduction="none").reshape(tgt.shape)
            ce = (ce * m).sum(1) / m.sum(1).clamp(min=1)
            out += ce.tolist()
        return out

    ce_o, ce_h = per_prompt_ce(False), per_prompt_ce(True)
    rng = random.Random(0)
    idx_all = list(range(len(prompts)))
    n = min(gate_n, len(idx_all))

    # Treatment-effect noise: sigma of the adapter-on-vs-off ratio across resamples, PAIRED
    # per prompt (ce_h[i]-ce_o[i] for the SAME i), so prompt-to-prompt difficulty cancels out
    # and what's left is genuine adapter-effect noise. This degenerates to EXACTLY 0 on a
    # fresh/untrained checkpoint -- not approximately small, but provably zero: the forward
    # pass is fully deterministic (no dropout/sampling) and an untrained LoRA contributes
    # nothing, so ce_h[i] == ce_o[i] bit-for-bit and no amount of resampling reveals variance
    # that genuinely isn't there. When degenerate, the CALLER defers calibration to after the
    # first real heal round instead of substituting a borrowed number here (see main()) -- an
    # untrained model simply has no signal to calibrate from yet, and manufacturing one (either
    # from an unpaired estimator, tried and reverted -- inflated the gate ~20x even on real
    # checkpoints -- or from a fixed floor borrowed from a DIFFERENT checkpoint's calibration)
    # isn't defensible: it would quietly break the "self-calibrating, no prior run needed"
    # property the whole point of this function is to provide.
    logratios = []
    for _ in range(n_boot):
        sub = rng.sample(idx_all, n)
        logratios.append(sum(ce_h[i] - ce_o[i] for i in sub) / n)
    mu = sum(logratios) / len(logratios)
    sd = (sum((x - mu) ** 2 for x in logratios) / max(len(logratios) - 1, 1)) ** 0.5
    gate = math.exp(sigma_mult * sd)   # against the TRUE baseline of 1.0, not exp(mu + sigma_mult*sd)
    degenerate = sd < 1e-9
    print(f"[cap-gate-auto] self-calibrated on {len(prompts)} prompts, {n_boot} resamples of {n}: "
          f"pooled ratio {math.exp(sum(ce_h) / len(ce_h) - sum(ce_o) / len(ce_o)):.4f}, "
          f"sigma(log)={sd:.5f}{' (DEGENERATE -- no real signal yet, deferring)' if degenerate else ''} "
          f"-> gate=exp({sigma_mult}*sigma)={gate:.4f}", flush=True)
    return {"gate": round(gate, 4), "sigma_log": round(sd, 5), "n_boot": n_boot, "gate_n": n,
            "degenerate": degenerate, "sigma_mult": sigma_mult}


def edit_eval(model, eval_ds, collator, device, circuit, eb, nb, seed,
              task="ioi", gt_generative=False, tok=None):
    """C2 editing/unlearning: on the FULL model (all edges ON), ablate ONLY the condensed
    circuit's edges (surgical) and measure (a) target-behavior drop and (b) collateral LM
    loss, vs ablating an equal number of RANDOM edges (specificity control). Target accuracy
    is dispatched PER TASK (greater-than uses its 'year>start' metric, not exact-token match)."""
    all_edges = set(range(len(model.edges)))
    k = len(circuit)
    torch.manual_seed(seed)
    rand_set = set(torch.randperm(len(all_edges))[:k].tolist())

    def task_acc():
        if task == "gt" and gt_generative:
            return circuit_eval.greater_than_generative_accuracy(
                model, tok, eval_ds.examples, device, masked=True, max_examples=eb * nb)
        if task == "gt":
            return circuit_eval.greater_than_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        if task == "agreement":
            return circuit_eval.agreement_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        return circuit_eval.circuit_token_accuracy(model, eval_ds, collator, device,
                                                   batch_size=eb, num_batches=nb)

    def probe(keep):
        set_binary_mask(model, keep)
        acc = task_acc()
        ce = _lm_ce(model, eval_ds, collator, device, eb, nb)
        return round(acc, 4), round(ce, 4)

    full_acc, full_ce = probe(all_edges)                       # normal full model
    cut_acc, cut_ce = probe(all_edges - circuit)               # ablate the circuit (surgical)
    rnd_acc, rnd_ce = probe(all_edges - rand_set)              # ablate equal-size random edges
    return {"circuit_edges": k,
            "target_full": full_acc, "target_circuit_ablated": cut_acc,
            "target_random_ablated": rnd_acc,
            "collateral_ce_full": full_ce, "collateral_ce_circuit_ablated": cut_ce,
            "collateral_ce_random_ablated": rnd_ce}


def build_dataset(task, path, tok, max_length, gt_generative):
    if task == "ioi":
        return IOIDataset(load_samples(path), tok, max_length)
    if task == "gt":
        return GreaterThanDataset(load_gt_samples(path), tok, max_length, generative=gt_generative)
    if task == "doc":
        return DocstringDataset(load_doc_samples(path), tok, max_length)
    if task == "induction":
        return InductionDataset(load_ind_samples(path), tok, max_length)
    if task == "agreement":
        return AgreementDataset(load_agr_samples(path), tok, max_length)
    raise ValueError(f"unsupported task {task}")


def measure(model, eval_ds, collator, device, eb, nb, task, gt_generative, tok) -> dict:
    """Circuit-acc (through the masked circuit), faithfulness KL, and full-model
    capability — dispatched per task exactly like scripts/evaluate.py."""
    model.eval()
    faith = round(circuit_eval.circuit_faithfulness(
        model, eval_ds, collator, device, batch_size=eb, num_batches=nb), 4)
    if task == "gt" and gt_generative:
        n = eb * nb
        circ = circuit_eval.greater_than_generative_accuracy(
            model, tok, eval_ds.examples, device, masked=True, max_examples=n)
        cap = circuit_eval.greater_than_generative_accuracy(
            model, tok, eval_ds.examples, device, masked=False, max_examples=n)
    elif task == "gt":
        circ = circuit_eval.greater_than_accuracy(
            model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        cap = circuit_eval.greater_than_accuracy(
            model, eval_ds, collator, device, masked=False, batch_size=eb, num_batches=nb)
    elif task == "agreement":
        # argmax-VALID (same-number verb set), not exact match -- see agreement_accuracy.
        circ = circuit_eval.agreement_accuracy(
            model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        cap = circuit_eval.agreement_accuracy(
            model, eval_ds, collator, device, masked=False, batch_size=eb, num_batches=nb)
    elif task == "ioi":
        # Canonical Wang IO-vs-S forced choice (not argmax-over-vocab -- see
        # ioi_forced_choice_accuracy): removes the metric bias that made bigger/newer models
        # look worse than gpt2 at IOI when they actually do it perfectly.
        circ = circuit_eval.ioi_forced_choice_accuracy(
            model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        cap = circuit_eval.ioi_forced_choice_accuracy(
            model, eval_ds, collator, device, masked=False, batch_size=eb, num_batches=nb)
    elif task == "induction":
        # Pool-restricted copy accuracy (not argmax-over-vocab -- see induction_pool_accuracy).
        circ = circuit_eval.induction_pool_accuracy(
            model, eval_ds, collator, device, masked=True, batch_size=eb, num_batches=nb)
        cap = circuit_eval.induction_pool_accuracy(
            model, eval_ds, collator, device, masked=False, batch_size=eb, num_batches=nb)
    else:  # doc: single-token answer, teacher-forced argmax == accuracy
        circ = circuit_eval.circuit_token_accuracy(
            model, eval_ds, collator, device, batch_size=eb, num_batches=nb)
        cap = circuit_eval.ioi_accuracy(
            model, eval_ds, collator, device, batch_size=eb, num_batches=nb)
    return {"circuit_acc": round(circ, 4), "faith": faith, "full_cap": round(cap, 4)}


def heal(model, keep_idx, train_ds, collator, device, steps, lr, batch, delta,
         cap_tol: float = 0.0, dual_lr: float = 0.05,
         delta_min: float = 1e-3, delta_max: float = 1e3,
         acc_floor: float = 0.0, acc_every: int = 30, orig_acc: float = 1.0,
         cap_fn=None, alpha: float = 1.0,
         circ_floor: float = 0.0, circ_fn=None,
         early_stop_acc: float = 0.0, early_every: int = 40,
         abort_fn=None, freeze_weights: bool = False,
         heal_mode: str = "standard", mix_p: float = 0.5,
         weight_decay: float = 0.01) -> tuple[float, float]:
    """Short leashed heal at a fixed mask. LoRA is NOT reset (accumulates across rounds).

    freeze_weights=True (the C0 arm: full adaptive controller -- EAP-IG ranking, margin cut,
    backtracking, cap-gate -- but weights genuinely never move): there is nothing to train, so
    this returns immediately (0 seconds, delta unchanged). The circuit's accuracy at a new mask
    is then whatever the untouched weights give, deterministically -- no heal loop needed, and
    the capability probe trivially always passes (an untrained/frozen LoRA can't drift).

    Leash term = delta * KL(orig||full). THREE modes (precedence: acc_floor > cap_tol > fixed):
      fixed (default)  -> delta is the constant `delta`.
      cap_tol > 0      -> KL-TOLERANCE: dual ascent to hold KL(orig||full) at `cap_tol`.
                          [TESTED NEGATIVE: KL conflates productive reshaping with capability
                          loss, so delta runs away to delta_max and strangles the circuit.]
      acc_floor > 0    -> ACCURACY-GATED (the fix): every `acc_every` steps measure the FULL
                          model's actual accuracy via `cap_fn()`, and dual-ascend delta on the
                          *real* capability drop:
                              log_delta += dual_lr * ((orig_acc - cur_acc) - acc_floor)
                          delta only RISES if capability genuinely fell past the floor; if
                          accuracy is fine (the usual case) delta DECAYS, freeing the circuit to
                          reshape. Pairs with a low delta_max so it can never run away.
    early_stop_acc > 0 turns healing into an EARLY-STOP loop: `steps` becomes a CEILING, and
    every `early_every` steps we measure the circuit's accuracy (circ_fn); the moment it reaches
    `early_stop_acc` we stop and return, so we never burn the full budget once the circuit is
    already good enough. Saves compute on easy rounds; hard rounds still run to the ceiling.

    Returns (secs, final_delta)."""
    import math
    set_binary_mask(model, keep_idx)
    model.eval()  # deterministic gates -> mask exactly held
    if freeze_weights:
        return 0.0, delta
    lora_params, gate_params = split_parameters(model)
    for p in gate_params:
        p.requires_grad_(False)
    for p in lora_params:
        p.requires_grad_(True)
    # weight_decay pulls the LoRA params toward ZERO, and a zero LoRA delta IS the original model --
    # so this is a drift leash toward the unmodified weights, enforced structurally rather than by
    # training on any corpus (rehearsing on general text would contaminate the capability claim).
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    loader = cycle(DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collator))
    adaptive = acc_floor > 0.0 or cap_tol > 0.0 or circ_floor > 0.0
    log_delta = math.log(max(delta, delta_min))
    lo, hi = math.log(delta_min), math.log(delta_max)
    t0 = time.time()
    for step in range(steps):
        b = next(loader)
        clean_ids, clean_mask = b["clean_ids"].to(device), b["clean_mask"].to(device)
        corr_ids, corr_mask = b["corr_ids"].to(device), b["corr_mask"].to(device)
        corrupted = model.capture_contributions(corr_ids, corr_mask)
        if heal_mode != "taskce":   # taskce heals on task labels, no frozen-original reference
            with model.peft_model.disable_adapter(), torch.no_grad():
                ref = model(clean_ids, clean_mask, masked=False)
        if heal_mode == "taskce":
            # C1-taskCE ABLATION: heal the circuit on TASK cross-entropy (predict the answer token)
            # instead of KL(orig‖masked). No reference to the frozen original -- this is exactly the
            # "just fine-tune the adapter on the task" alternative. Expected to reach
            # task accuracy but DRIFT from the original (capability loss, KL_orig up), which is what
            # justifies C1's KL-to-original objective.
            import torch.nn.functional as F
            masked = model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
            labels = b["labels"].to(device)
            lp = masked[:, :-1, :].reshape(-1, masked.size(-1)).float()
            loss = F.cross_entropy(lp, labels[:, 1:].reshape(-1), ignore_index=-100)
            l_preserve = loss
        elif heal_mode == "mixed":
            # MIXED-CUT-EDGE heal: one objective --
            # reproduce ref whether the cut edges carry the CORRUPTED signal (prob mix_p) or the
            # CLEAN signal (prob 1-mix_p, == the full-model/preservation case). No alpha/delta
            # ratio: full_cap preservation falls out for free, and AdamW's per-step normalisation
            # makes the balance the (bounded) frequency mix_p instead of an unbounded multiplier.
            if random.random() < mix_p:
                out = model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
            else:
                out = model(clean_ids, clean_mask, masked=False)
            loss = kl_loss(out, ref, clean_mask)
            l_preserve = loss  # for the (unused-in-mixed) dual-ascent logging paths below
        else:
            masked = model(clean_ids, clean_mask, corrupted=corrupted, masked=True)
            loss = alpha * kl_loss(masked, ref, clean_mask)
            d = math.exp(log_delta) if adaptive else delta
            if d > 0.0:
                full = model(clean_ids, clean_mask, masked=False)
                l_preserve = kl_loss(full, ref, clean_mask)
                loss = loss + d * l_preserve
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if circ_floor > 0.0:  # Q2: gate δ on the CIRCUIT's accuracy (raise δ when circuit underperforms)
            if step % acc_every == 0 and circ_fn is not None:
                log_delta += dual_lr * (circ_floor - float(circ_fn()))
                log_delta = min(max(log_delta, lo), hi)
        elif acc_floor > 0.0:  # accuracy-gated dual ascent (react to REAL capability loss)
            if step % acc_every == 0 and cap_fn is not None:
                drop = orig_acc - float(cap_fn())   # how much the full model actually lost
                log_delta += dual_lr * (drop - acc_floor)
                log_delta = min(max(log_delta, lo), hi)
        elif cap_tol > 0.0:  # KL-tolerance (kept for the ablation record)
            log_delta += dual_lr * (float(l_preserve.detach()) - cap_tol)
            log_delta = min(max(log_delta, lo), hi)
        # RUNAWAY ABORT: catch an OOD-capability-destroying trajectory mid-heal (minutes, not
        # a full heal). The round-level cap-gate then rejects and re-rolls it.
        if abort_fn is not None and step > 0 and step % early_every == 0:
            if abort_fn():
                print(f"      [heal ABORT] capability drift past the abort threshold at step "
                      f"{step}/{steps} — runaway trajectory, stopping this heal", flush=True)
                break
        # EARLY-STOP: stop as soon as the circuit is already good enough (compute-saving).
        if early_stop_acc > 0.0 and circ_fn is not None and step > 0 and step % early_every == 0:
            if float(circ_fn()) >= early_stop_acc:
                print(f"      [heal early-stop] acc>={early_stop_acc:.3f} at step {step}/{steps} "
                      f"(saved {steps - step - 1} steps)", flush=True)
                break
    return time.time() - t0, (math.exp(log_delta) if adaptive else delta)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(Path(args.checkpoint) / "config.json") as f:
        cfg = json.load(f)
    model_path = args.model_path or cfg["model_path"]
    granularity = cfg.get("graph_granularity", "block")
    max_length = args.max_length or cfg.get("max_length", 64)
    task = args.task or cfg.get("task", "ioi")
    gt_generative = task == "gt" and cfg.get("gt_mode", "generative") == "generative"
    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    print(f"checkpoint={args.checkpoint} model={model_path} task={task} start={args.start} "
          f"schedule={schedule} rerank={args.rerank} heal_steps/round={args.heal_steps} "
          f"lr={args.heal_lr} delta={args.heal_delta} gt_generative={gt_generative}", flush=True)

    assert_disjoint_splits(args.train_data, args.eval_data, args.test_data)

    tok = load_tokenizer(model_path)
    collator = PairCollator(tok.pad_token_id)
    train_ds = build_dataset(task, args.train_data, tok, max_length, gt_generative)
    eval_ds = build_dataset(task, args.eval_data, tok, max_length, gt_generative)
    # SEALED test split: controller decisions use eval; the FINAL reported numbers use test.
    test_ds = build_dataset(task, args.test_data, tok, max_length, gt_generative)

    def meas():
        return measure(model, eval_ds, collator, device, args.eval_batch, args.eval_batches,
                       task, gt_generative, tok)

    # The sealed test split is read only twice per run (m0 at the start, the endpoint at the end),
    # so there is no reason to subsample it the way the per-round eval does. Use ALL of it: with a
    # 1000-example test split this takes the reported number's 95% CI from ~+-0.068 to ~+-0.030,
    # which is what makes "the circuit is within tol of the original" a statement rather than a
    # coin flip. (args.eval_batches is sized for the ~20 per-round evals, where speed matters.)
    n_test_batches = math.ceil(len(test_ds) / args.eval_batch)

    def meas_test():
        return measure(model, test_ds, collator, device, args.eval_batch, n_test_batches,
                       task, gt_generative, tok)

    def cap_only():
        """Cheap full-model accuracy on a few batches (for accuracy-gated δ during heal)."""
        if task == "gt" and gt_generative:
            return circuit_eval.greater_than_generative_accuracy(
                model, tok, eval_ds.examples, device, masked=False, max_examples=4 * args.eval_batch)
        if task == "gt":
            return circuit_eval.greater_than_accuracy(
                model, eval_ds, collator, device, masked=False, batch_size=args.eval_batch, num_batches=4)
        if task == "agreement":
            return circuit_eval.agreement_accuracy(
                model, eval_ds, collator, device, masked=False, batch_size=args.eval_batch, num_batches=4)
        if task == "ioi":
            return circuit_eval.ioi_forced_choice_accuracy(
                model, eval_ds, collator, device, masked=False, batch_size=args.eval_batch, num_batches=4)
        if task == "induction":
            return circuit_eval.induction_pool_accuracy(
                model, eval_ds, collator, device, masked=False, batch_size=args.eval_batch, num_batches=4)
        return circuit_eval.ioi_accuracy(  # doc: single-token argmax
            model, eval_ds, collator, device, batch_size=args.eval_batch, num_batches=4)

    def circ_only():
        """Cheap CIRCUIT accuracy (masked) on a few batches (for circuit-acc-gated δ, Q2)."""
        if task == "gt" and gt_generative:
            return circuit_eval.greater_than_generative_accuracy(
                model, tok, eval_ds.examples, device, masked=True, max_examples=4 * args.eval_batch)
        if task == "gt":
            return circuit_eval.greater_than_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=args.eval_batch, num_batches=4)
        if task == "agreement":
            return circuit_eval.agreement_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=args.eval_batch, num_batches=4)
        if task == "ioi":
            return circuit_eval.ioi_forced_choice_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=args.eval_batch, num_batches=4)
        if task == "induction":
            return circuit_eval.induction_pool_accuracy(
                model, eval_ds, collator, device, masked=True, batch_size=args.eval_batch, num_batches=4)
        return circuit_eval.circuit_token_accuracy(  # doc
            model, eval_ds, collator, device, batch_size=args.eval_batch, num_batches=4)

    model = load_model(model_path, args.checkpoint, granularity, device)
    model.eval()

    # General-capability probe prompts (unrelated text), loaded up front when any of the
    # probe-driven features need them (final probe, per-round probe, accept gate, heal abort,
    # gamma sweep).
    probe_prompts = None
    if (args.capability_probe or args.probe_every_round or args.cap_gate > 0
            or args.cap_abort > 0 or args.gamma_sweep or args.cap_gate_auto
            or args.diagnostic_sweep):
        # No silent fallback to eval: the probe gates accept/reject, so falling back to the
        # controller's own decision split makes the gate self-referential (and violates the
        # always-mixed-corpus rule). Fail loudly instead, and verify the corpus is not leaky.
        if not args.probe_data:
            raise ValueError(
                "a capability gate/probe/abort was requested but --probe-data was not set. "
                "Refusing to fall back to --eval-data: the probe drives accept/reject, so gating "
                "on the decision split is self-referential. Pass data/probe_mixed.jsonl "
                "(slurm/prune_heal_gradual.slurm sets PROBE_DATA to it by default).")
        probe_path = args.probe_data
        assert_probe_disjoint(probe_path, args.eval_data, args.test_data)
        with open(probe_path) as f:
            probe_prompts = [json.loads(l)["clean_prompt"] for l in f if l.strip()]

    def cap_ratio(nb):
        """Probe ppl ratio (healed/original) on unrelated text, nb batches. ~1.0 = healthy."""
        return capability_probe(model, probe_prompts, tok, device,
                                args.probe_max_length or max_length,
                                eb=args.eval_batch, nb=nb)["probe_ppl_ratio"]

    # Start the descent from the trained circuit, or from ALL edges ON (the full model).
    if args.start == "full":
        current_keep = set(range(len(model.edges)))
    else:
        m0 = model.gate.hard_mask().bool().cpu()
        current_keep = set(torch.nonzero(m0).squeeze(-1).tolist())
    set_binary_mask(model, current_keep)
    start_n = len(current_keep)
    base = meas()
    if args.measure_panel:
        base["panel"] = measure_panel(model, eval_ds, collator, device, args.panel_batches)
    # m0 on the SEALED TEST split too. `base` is eval-side (it has to be: the round accept/reject
    # floors derive from it, and those are eval decisions). But every "circuit vs original"
    # comparison downstream reads `final_test` (test-side) against `m0_baseline` (eval-side) --
    # two different 200-example datasets. At n=200, 1 sigma is ~0.034, and the splits are not
    # stratified (agreement/llama3b's original scores 0.74 on eval vs 0.655 on test), so that
    # mismatch can manufacture a false "the circuit beats the original model" alarm.
    # Record the like-for-like number so downstream readers can compare test against test.
    base_test = meas_test() if args.test_data else None
    if base_test is not None:
        print(f"start/test: {start_n} edges {base_test}", flush=True)
    print(f"start ({args.start}): {start_n} edges {base}", flush=True)

    # SELF-CALIBRATE the capability gate (no prior run/checkpoint needed): the STARTING circuit
    # (right here, before any cut) already carries a real, trained LoRA from the base training
    # stage and is healthy by construction -- calibrate the gate's noise floor on it directly.
    cap_gate_calib = None
    # Auto-calibration overwrites args.cap_gate below, so the value the caller ASKED for is
    # otherwise unrecoverable from the run. Keep it: an arm once ran at an effective ~1.01
    # while its launcher and the paper both said 1.05, and nothing in the artifact showed it.
    cap_gate_requested = args.cap_gate
    if args.cap_gate_auto:
        cap_gate_calib = calibrate_cap_gate(model, probe_prompts, tok, device, max_length,
                                            gate_n=args.cap_gate_batches * args.eval_batch,
                                            n_boot=args.cap_gate_boot, batch=args.eval_batch,
                                            sigma_mult=args.cap_gate_sigma_mult)
        args.cap_gate = cap_gate_calib["gate"]
        # DEFERRED calibration for a fresh/untrained checkpoint: the calibration
        # above is genuinely degenerate here (sigma=0, not just small -- see calibrate_cap_gate's
        # docstring), so there's no real signal to gate on yet. Rather than substitute a number
        # borrowed from a DIFFERENT checkpoint's own calibration (not defensible -- it would
        # quietly break the "self-calibrating, no prior run needed" property), leave the gate OFF
        # for the first --cap-gate-calib-rounds rounds (the smallest, least risky perturbations,
        # moving away from a known-safe starting point anyway) and recalibrate from THIS run's
        # own state after EACH of them, averaging across all of them for the final gate used from
        # then on -- one round alone might catch the noise level of only the tiniest possible
        # nudge, understating what it looks like once training has actually progressed a bit.
        # Fully self-contained either way: no external number, ever.
        if cap_gate_calib.get("degenerate"):
            print(f"[cap-gate-auto] degenerate at start (fresh/untrained checkpoint) -- gate OFF "
                  f"for the first {args.cap_gate_calib_rounds} round(s), will recalibrate from "
                  f"this run's own heals", flush=True)
            args.cap_gate = 0.0
    pending_recalib = bool(args.cap_gate_auto and cap_gate_calib and cap_gate_calib.get("degenerate"))
    recalib_sigmas: list = []

    # Optional single ranking up front (rerank=once).
    static_rank = None
    if args.rerank == "once":
        set_binary_mask(model, current_keep)
        static_rank = eap_ranking(model, train_ds, collator, device, args.rank_batches, seed=args.seed, ig_steps=args.ig_steps)

    rows = []
    report_extra = {}

    # Per-round edge SETS, streamed to a sidecar JSONL next to --out, so analyses that need the
    # actual circuits (seed containment, minimality, overlap) don't require re-running the cell.
    # Appending after every accepted round also survives a job hitting the slurm time limit.
    edges_log = (Path(args.out).with_suffix("").parent /
                 (Path(args.out).with_suffix("").name + "_edges.jsonl")) if args.out else None
    if edges_log is not None:
        edges_log.parent.mkdir(parents=True, exist_ok=True)
        edges_log.write_text("")   # fresh file per run; one accepted round per line

    def log_edges(phase, keep, row):
        if edges_log is None:
            return
        with open(edges_log, "a") as f:
            f.write(json.dumps({"round": len(rows), "phase": phase,
                                "n_edges": len(keep),
                                "circuit_acc": row.get("circuit_acc"),
                                "faith": row.get("faith"),
                                "edges": sorted(keep)}) + "\n")

    # B1 CONTROL under the adaptive controller: heal repeatedly at the FULL (starting) mask,
    # never cutting a single edge, for --adaptive-b1-rounds rounds. Tests whether repeated
    # healing ALONE (no edge removal) already explains any circuit concentration -- the
    # adaptive-controller analogue of the old fixed-schedule --no-prune control (which only
    # ever ran through the retired `for target in schedule:` loop further below).
    if args.adaptive and args.no_prune:
        n_rounds = args.adaptive_b1_rounds if args.adaptive_b1_rounds > 0 else 10
        print(f"[adaptive:B1-control] no-prune: {n_rounds} heal-only rounds at the full "
              f"{len(current_keep)}-edge mask", flush=True)
        best_keep, best_snap, best_acc = set(current_keep), snapshot_lora(model), base["circuit_acc"]
        for i in range(n_rounds):
            s, d = heal(model, current_keep, train_ds, collator, device,
                       args.heal_steps, args.heal_lr, args.heal_batch, args.heal_delta,
                       cap_tol=args.heal_cap_tol, dual_lr=args.heal_dual_lr,
                       delta_max=args.heal_delta_max, acc_floor=args.heal_acc_floor,
                       acc_every=args.heal_acc_every, orig_acc=base["full_cap"],
                       cap_fn=cap_only, alpha=args.heal_alpha,
                       circ_floor=args.heal_circ_floor, circ_fn=circ_only,
                       early_stop_acc=base["circuit_acc"], early_every=args.adaptive_early_every,
                       freeze_weights=args.freeze_weights, heal_mode=args.heal_mode, weight_decay=args.heal_weight_decay, mix_p=args.mix_p)
            m = meas()
            if args.cap_gate > 0 or args.probe_every_round:
                m["cap_ratio"] = cap_ratio(args.cap_gate_batches)
            realised = circuit_eval.circuit_size(model)
            row = {"target": start_n, "realised_edges": realised, **m, "heal_secs": round(s, 1),
                   "delta": round(d, 3), "retries": 0, "phase": "b1_control",
                   "knee": None, "beats_knee": None, "milestone": False, "cut_frac": 0.0}
            if args.measure_panel:
                row["panel"] = measure_panel(model, eval_ds, collator, device, args.panel_batches)
            rows.append(row)
            log_edges("b1_control", current_keep, row)
            print(f"  [B1-control] round {i+1}/{n_rounds} edges={realised} acc={m['circuit_acc']:.3f} "
                  f"faith={m['faith']:.2f}", flush=True)
            best_keep, best_snap, best_acc = set(current_keep), snapshot_lora(model), m["circuit_acc"]
        current_keep = best_keep
        report_extra = {"best_faithful_edges": len(best_keep), "best_faithful_acc": round(best_acc, 4),
                        "most_pruned_edges": len(best_keep), "most_pruned_acc": round(best_acc, 4),
                        "retry_stats": {"acc_floor_rejects": 0, "cap_gate_rejects": 0, "heal_aborts": 0,
                                       "backtrack_halvings": 0, "cliff_reheal_probes": 0}}
        print(f"[adaptive:B1-control] STOP after {n_rounds} rounds. final = {len(best_keep)}e@{best_acc:.3f} "
              f"(never cut, by design)", flush=True)
        schedule = []  # skip the fixed-schedule loop below

    # ADAPTIVE MODE: no hand-set schedule and no hand-set stop. Each round cuts a FRACTION of
    # the current edges, heals, and checks whether the circuit is still faithful (accuracy >=
    # baseline - tol). If a cut drops below the floor, re-heal a few times to try to recover;
    # if it still can't, we've hit the cliff -> STOP and return the last faithful circuit. Cuts
    # get gentler as we near the floor. This removes the last manual knob (the schedule).
    elif args.adaptive:
        ref = base["circuit_acc"]
        floor = ref - args.adaptive_tol
        full_floor = base["full_cap"] - args.adaptive_fullcap_tol   # task-capability gate (C1 only in practice)
        cut = args.adaptive_cut

        # Transparency counters: every rejected/re-rolled attempt gets counted here and reported
        # in the final JSON, so "how many tries did it take" is a disclosed number.
        stats = {"acc_floor_rejects": 0, "cap_gate_rejects": 0, "fullcap_floor_rejects": 0,
                 "heal_aborts": 0, "backtrack_halvings": 0, "cliff_reheal_probes": 0,
                 "mixed_switch_probes": 0, "mixed_switch_latched": 0, "lazy_skips": 0}

        def abort_fn_counted():
            if cap_ratio(4) > args.cap_abort:
                stats["heal_aborts"] += 1
                return True
            return False
        abort_fn = abort_fn_counted if args.cap_abort > 0 else None

        def _heal_once(new_keep, es, hmode):
            return heal(model, new_keep, train_ds, collator, device,
                        args.heal_steps, args.heal_lr, args.heal_batch, args.heal_delta,
                        cap_tol=args.heal_cap_tol, dual_lr=args.heal_dual_lr,
                        delta_max=args.heal_delta_max, acc_floor=args.heal_acc_floor,
                        acc_every=args.heal_acc_every, orig_acc=base["full_cap"],
                        cap_fn=cap_only, alpha=args.heal_alpha,
                        circ_floor=args.heal_circ_floor, circ_fn=circ_only,
                        early_stop_acc=es, early_every=args.adaptive_early_every,
                        abort_fn=abort_fn, freeze_weights=args.freeze_weights, heal_mode=hmode, mix_p=args.mix_p, weight_decay=args.heal_weight_decay)

        def _cut_and_heal(keep, tgt, restore_snap, heal_buf, hmode):
            """ONE probe: restore the last-GOOD weights, cut to the top-`tgt` important edges,
            heal once (early-stopping at floor+heal_buf). Returns (new_keep, meas, secs, delta).
            Always probing from the last-good weights means a failed probe never poisons the
            model and every probe is a clean, comparable attempt."""
            if restore_snap is not None:
                restore_lora(model, restore_snap)
            set_binary_mask(model, keep)
            imp = static_rank if static_rank is not None else \
                eap_ranking(model, train_ds, collator, device, args.rank_batches, seed=args.seed, ig_steps=args.ig_steps)
            if args.rank_mode == "random":
                # B2 control: ignore importance -- keep a RANDOM subset of the current edges.
                shuffled = list(keep)
                torch.manual_seed(args.seed + len(keep))
                perm = torch.randperm(len(shuffled)).tolist()
                ranked = [shuffled[i] for i in perm]
            elif args.rank_mode == "anti":
                # B2 control: keep the LEAST important edges (negate the ranking).
                ranked = sorted(keep, key=lambda e: float(imp[e]))
            else:  # eap (default)
                ranked = sorted(keep, key=lambda e: float(imp[e]), reverse=True)
            new_keep = set(ranked[:tgt])
            es = min(ref, floor + heal_buf) if args.adaptive_early_every > 0 else 0.0
            if args.lazy_heal and not args.freeze_weights:
                # LAZY HEAL: test the RAW cut first; heal only if it dropped the circuit
                # below the accuracy floor. Healing is the sole source of capability drift (frozen
                # weights never drift -- that IS C0), so healing a cut that already passes is pure
                # drift for zero benefit -- it just pushes cap_ratio toward the gate for nothing,
                # which is what stops C1 early on drift-sensitive tasks (docstring). Where healing is
                # genuinely needed (cut fails) this path is identical to the eager heal.
                set_binary_mask(model, new_keep)
                raw = meas()
                if raw["circuit_acc"] >= floor:
                    stats["lazy_skips"] = stats.get("lazy_skips", 0) + 1
                    s, d = 0.0, args.heal_delta
                else:
                    s, d = _heal_once(new_keep, es, hmode)
            else:
                s, d = _heal_once(new_keep, es, hmode)
            m = meas()
            if args.cap_gate > 0 or args.probe_every_round:
                m["cap_ratio"] = cap_ratio(args.cap_gate_batches)
            return new_keep, m, s, d

        saved_rounds: list[Path] = []

        def save_round_state(realised, m):
            """Persist an accepted (adapter, gates) state so the trajectory can be re-scored
            offline. Keeps a rolling window: the walk-back and the Pareto curve both only
            need the tail, and a full history is 10-70MB per round per run."""
            if not args.save_rounds:
                return
            root = Path(args.save_rounds)
            d = root / f"r{len(rows):03d}_e{realised}"
            try:
                d.mkdir(parents=True, exist_ok=True)
                model.peft_model.save_pretrained(str(d / "lora"))
                torch.save(model.gate.state_dict(), d / "gates.pt")
                (d / "meta.json").write_text(json.dumps(
                    {"edges": realised, "round": len(rows), "eval_acc": m.get("circuit_acc"),
                     "faith": m.get("faith"), "full_cap": m.get("full_cap")}, indent=1))
                saved_rounds.append(d)
                while len(saved_rounds) > max(1, args.save_rounds_keep):
                    shutil.rmtree(saved_rounds.pop(0), ignore_errors=True)
            except Exception as e:   # never let checkpointing kill a run
                print(f"      [save-rounds] WARN {type(e).__name__}: {e}", flush=True)

        def round_ok(m):
            """ACCEPT predicate: task-faithful AND (if gated) generally healthy. A capability
            failure is rejected + re-rolled exactly like an accuracy failure."""
            acc_for_test = m["circuit_acc"]
            if args.accept_lcb > 0:
                # One-sided normal-approximation lower bound. n is the number of eval
                # examples the gate actually scores, not the split size.
                n_eval = max(1, args.eval_batch * args.eval_batches)
                phat = min(max(acc_for_test, 0.0), 1.0)
                se = (phat * (1.0 - phat) / n_eval) ** 0.5
                acc_for_test = acc_for_test - args.accept_lcb * se
            if acc_for_test < floor:
                stats["acc_floor_rejects"] += 1
                return False
            if args.adaptive_fullcap_tol > 0 and m["full_cap"] < full_floor:
                stats["fullcap_floor_rejects"] += 1
                print(f"      [fullcap] REJECT: full-model task acc {m['full_cap']:.3f} < "
                      f"{full_floor:.3f} (base {base['full_cap']:.3f} - {args.adaptive_fullcap_tol}); "
                      f"circuit_acc {m['circuit_acc']:.3f} was fine", flush=True)
                return False
            if args.cap_gate > 0 and m.get("cap_ratio", 0.0) > args.cap_gate:
                stats["cap_gate_rejects"] += 1
                print(f"      [cap-gate] REJECT: probe ratio {m['cap_ratio']:.3f} > "
                      f"{args.cap_gate} (acc {m['circuit_acc']:.3f} was fine)", flush=True)
                return False
            return True

        def fullcap_only_fail(m):
            """True iff this round would be rejected SOLELY because full_cap fell below its
            floor -- circuit_acc and cap_ratio are both fine. This is the 'full_cap is the
            binding constraint' signature the conditional-mixed switch reacts to. Read-only:
            it does NOT touch the stats counters (round_ok already counted the real reject)."""
            return (args.adaptive_fullcap_tol > 0
                    and m["circuit_acc"] >= floor
                    and m["full_cap"] < full_floor
                    and not (args.cap_gate > 0 and m.get("cap_ratio", 0.0) > args.cap_gate))

        # best = smallest FAITHFUL circuit (roll back here for edit-eval); we also snapshot its
        # LoRA weights so a later divergent heal can't corrupt the model we report on.
        best_keep = set(current_keep)
        best_snap = snapshot_lora(model)
        best_acc = ref
        acc_now = ref                       # accuracy at the CURRENT (pre-cut) circuit
        # CONDITIONAL-MIXED SWITCH state. active_mode is the heal mode used every round; it flips
        # to "mixed" (and latches there) the first time the switch fires. Starts at --heal-mode,
        # so --heal-mode mixed means active_mode=="mixed" from the outset and the switch is inert.
        active_mode = args.heal_mode
        mixed_latched = (args.heal_mode in ("mixed", "taskce"))   # taskce is terminal: switch inert
        # AUTO-SIZED heal buffer: heal each round to floor + (expected drop of the NEXT cut), where
        # the expected drop is a running average of how much accuracy each cut actually costs. Cuts
        # barely hurt near the top (small buffer, cheap heals) and hurt more near the cliff (buffer
        # grows automatically to protect the endgame). Seeded by frac*tol until we have history.
        drop_ema = None
        buf_seed = args.adaptive_heal_buffer_frac * args.adaptive_tol
        def heal_buffer():
            if args.adaptive_heal_buffer > 0:   # absolute manual override
                return args.adaptive_heal_buffer
            b = buf_seed if drop_ema is None else drop_ema
            return min(args.adaptive_tol, max(0.005, b))   # clip to [0.5%, tol]
        max_cut = args.adaptive_max_cut
        gain = args.adaptive_prop_gain if args.adaptive_prop_gain > 0 else max_cut / args.adaptive_tol
        mode = "margin-prop" if args.adaptive_prop else "fixed+halve"
        print(f"[adaptive:{mode}] ref_acc={ref:.3f} floor={floor:.3f} "
              f"start_cut={cut} gain={gain:.1f}", flush=True)
        while len(current_keep) > args.adaptive_min:
            n = len(current_keep)
            if args.adaptive_prop:
                cut = min(max_cut, max(args.adaptive_min_cut, gain * (acc_now - floor)))
            # BACKTRACKING LINE-SEARCH on the cut size. Try the cut; if the healed circuit is
            # BELOW the floor we OVER-LEAPT the cliff, so HALVE the cut and probe a gentler jump
            # (from the last-good weights) — never charge over the edge. Only when even the
            # smallest cut (min_cut) still fails do we spend fresh confirm-retries; if those also
            # fail, it's the real cliff. Because healing hides the drop until the last moment,
            # this backtracking is what keeps the aggressive cut from leaping past the knee.
            accepted = None; secs = 0.0; probes = 0; fdelta = args.heal_delta
            while True:
                target = max(args.adaptive_min, int(round(n * (1 - cut))))
                if target >= n:
                    target = n - 1
                new_keep, m, s, fdelta = _cut_and_heal(current_keep, target, best_snap, heal_buffer(), active_mode)
                secs += s; probes += 1
                if round_ok(m):
                    accepted = (new_keep, m); break
                # CONDITIONAL-MIXED SWITCH: this round failed ONLY on full_cap (circuit + probe both
                # fine) => full_cap is the binding constraint. Re-heal the SAME cut with the mixed
                # heal from the SAME last-good weights (best_snap, undoing the std heal's capability
                # damage) BEFORE spending a backtrack. If it clears every gate, latch mixed on for
                # the rest of the run -- once full_cap binds at some sparsity it stays binding.
                if (args.adaptive_mixed_switch and not mixed_latched
                        and active_mode != "mixed" and fullcap_only_fail(m)):
                    stats["mixed_switch_probes"] += 1
                    nk_mix, m_mix, s_mix, _dm = _cut_and_heal(current_keep, target, best_snap,
                                                              heal_buffer(), "mixed")
                    secs += s_mix; probes += 1
                    if round_ok(m_mix):
                        active_mode = "mixed"; mixed_latched = True
                        stats["mixed_switch_latched"] += 1
                        print(f"      [mixed-switch] full_cap-bound at {target}e: std full_cap "
                              f"{m['full_cap']:.3f} -> mixed {m_mix['full_cap']:.3f} "
                              f"(circ {m_mix['circuit_acc']:.3f}); LATCHED mixed for rest of run",
                              flush=True)
                        accepted = (nk_mix, m_mix); m = m_mix; break
                    # mixed also failed the gates -> fall through to the normal backtrack below.
                if cut > args.adaptive_min_cut + 1e-9:
                    stats["backtrack_halvings"] += 1
                    print(f"      [backtrack] {target}e@{m['circuit_acc']:.3f} rejected -> "
                          f"halve cut {cut:.3f}->{max(args.adaptive_min_cut, cut*0.5):.3f}", flush=True)
                    cut = max(args.adaptive_min_cut, cut * 0.5)
                    continue
                # smallest cut still fails -> confirm the cliff with fresh retries (keep best).
                best_m, best_nk, best_ok = m, new_keep, False
                for _ in range(args.adaptive_reheal):
                    nk, m2, s2, _d = _cut_and_heal(current_keep, target, best_snap, heal_buffer(), active_mode)
                    secs += s2; probes += 1; stats["cliff_reheal_probes"] += 1
                    ok2 = round_ok(m2)
                    if ok2 or (not best_ok and m2["circuit_acc"] > best_m["circuit_acc"]):
                        best_m, best_nk, best_ok = m2, nk, ok2
                    if ok2:
                        break
                if best_ok:
                    accepted = (best_nk, best_m)
                else:
                    accepted = None; m = best_m
                break
            if accepted is None:
                restore_lora(model, best_snap); set_binary_mask(model, best_keep)   # clean state for tail/edit
                print(f"  [adaptive] CLIFF below {len(best_keep)}e (min-cut probe {m['circuit_acc']:.3f} "
                      f"< floor {floor:.3f} after {probes} probes)", flush=True)
                break
            current_keep, m = accepted
            realised = circuit_eval.circuit_size(model)
            if pending_recalib:
                # This round just genuinely moved the weights -- real signal now exists, entirely
                # from THIS run. Recalibrate from here and accumulate; once we've collected
                # --cap-gate-calib-rounds of them, average and turn the gate on. No external/
                # borrowed number at any point -- just possibly more than one of this run's own.
                recalib = calibrate_cap_gate(model, probe_prompts, tok, device, max_length,
                                             gate_n=args.cap_gate_batches * args.eval_batch,
                                             n_boot=args.cap_gate_boot, batch=args.eval_batch,
                                             sigma_mult=args.cap_gate_sigma_mult)
                if not recalib.get("degenerate"):
                    recalib_sigmas.append(recalib["sigma_log"])
                    print(f"[cap-gate-auto] calibration sample {len(recalib_sigmas)}/"
                          f"{args.cap_gate_calib_rounds}: sigma(log)={recalib['sigma_log']:.5f} "
                          f"(this round's gate would be {recalib['gate']:.4f})", flush=True)
                    if len(recalib_sigmas) >= args.cap_gate_calib_rounds:
                        avg_sigma = sum(recalib_sigmas) / len(recalib_sigmas)
                        args.cap_gate = round(math.exp(args.cap_gate_sigma_mult * avg_sigma), 4)
                        cap_gate_calib = {"gate": args.cap_gate, "sigma_log": round(avg_sigma, 5),
                                          "sigma_samples": recalib_sigmas, "n_boot": args.cap_gate_boot,
                                          "sigma_mult": args.cap_gate_sigma_mult}
                        pending_recalib = False
                        print(f"[cap-gate-auto] recalibrated from {len(recalib_sigmas)} of this "
                              f"run's own rounds (sigmas={[round(s, 5) for s in recalib_sigmas]}, "
                              f"avg={avg_sigma:.5f}) -> gate={args.cap_gate:.4f} (now active)",
                              flush=True)
                else:
                    print("[cap-gate-auto] still degenerate this round (heal_steps too short "
                          "to move weights measurably?) -- gate stays off, will retry next round",
                          flush=True)
            row = {"target": len(current_keep), "realised_edges": realised, **m, "heal_secs": round(secs, 1),
                   "delta": round(fdelta, 3), "retries": probes - 1, "phase": "search",
                   "knee": None, "beats_knee": None, "milestone": False, "cut_frac": round(cut, 3)}
            if args.measure_panel:
                row["panel"] = measure_panel(model, eval_ds, collator, device, args.panel_batches)
            rows.append(row)
            log_edges("search", current_keep, row)
            save_round_state(realised, m)
            print(f"  [adaptive] {realised:5d}e acc={m['circuit_acc']:.3f} faith={m['faith']:.2f} "
                  f"cut={cut:.3f} probes={probes} -> OK", flush=True)
            # bookkeeping: update the drop estimate + the best (last-good) checkpoint.
            obs_drop = max(0.0, acc_now - m["circuit_acc"])
            drop_ema = obs_drop if drop_ema is None else \
                args.adaptive_buffer_beta * drop_ema + (1 - args.adaptive_buffer_beta) * obs_drop
            best_keep = set(current_keep)
            best_snap = snapshot_lora(model)
            best_acc = m["circuit_acc"]
            acc_now = m["circuit_acc"]
            if not args.adaptive_prop:   # enforced gradual decay (accuracy-independent pacing)
                cut = max(args.adaptive_min_cut, cut * args.adaptive_cut_decay)

        # OVERSHOOT: keep pruning a few MORE rounds PAST the cliff to chart how accuracy falls
        # off — the knee-curve tail. Each tail round restarts from the best
        # faithful weights (so a single divergent heal doesn't poison the whole tail) and cuts
        # one more min-cut step deeper. We remember the most-pruned checkpoint too.
        pruned_keep, pruned_snap, pruned_acc = set(best_keep), best_snap, best_acc
        if args.adaptive_overshoot > 0 and len(best_keep) > args.adaptive_min:
            print(f"[adaptive] OVERSHOOT: {args.adaptive_overshoot} extra rounds past the knee", flush=True)
            tail_keep = set(best_keep)
            for o in range(args.adaptive_overshoot):
                restore_lora(model, best_snap)  # tail always measured from the healthy knee
                n = len(tail_keep)
                tgt = max(args.adaptive_min, int(round(n * (1 - args.adaptive_min_cut))))
                if tgt >= n:
                    tgt = n - 1
                if tgt < args.adaptive_min:
                    break
                tail_keep, m, secs, fdelta = _cut_and_heal(set(best_keep) if o == 0 else tail_keep,
                                                           tgt, best_snap, heal_buffer(), active_mode)
                realised = circuit_eval.circuit_size(model)
                row = {"target": tgt, "realised_edges": realised, **m, "heal_secs": round(secs, 1),
                       "delta": round(fdelta, 3), "retries": 0, "phase": "overshoot",
                       "knee": None, "beats_knee": None, "milestone": False,
                       "cut_frac": round(args.adaptive_min_cut, 3)}
                if args.measure_panel:
                    row["panel"] = measure_panel(model, eval_ds, collator, device, args.panel_batches)
                rows.append(row)
                log_edges("overshoot", tail_keep, row)
                pruned_keep, pruned_snap, pruned_acc = set(tail_keep), snapshot_lora(model), m["circuit_acc"]
                print(f"  [overshoot] {realised:5d}e acc={m['circuit_acc']:.3f} "
                      f"(knee was {len(best_keep)}e@{best_acc:.3f})", flush=True)

        # roll BACK to the best faithful circuit — this is the model we report / edit-eval on.
        restore_lora(model, best_snap)
        current_keep = set(best_keep)
        set_binary_mask(model, current_keep)
        report_extra = {"best_faithful_edges": len(best_keep), "best_faithful_acc": round(best_acc, 4),
                        "most_pruned_edges": len(pruned_keep), "most_pruned_acc": round(pruned_acc, 4),
                        "retry_stats": stats}
        print(f"[adaptive] STOP. best faithful = {len(best_keep)}e@{best_acc:.3f} | "
              f"most pruned (tail) = {len(pruned_keep)}e@{pruned_acc:.3f} | "
              f"retries: {stats}", flush=True)
        schedule = []  # skip the fixed-schedule loop below

    # DIAGNOSTIC SWEEP: NOT the real headline controller -- a separate, additive
    # mode that charts the FULL degradation curve past where the adaptive controller above would
    # normally stop. No retries/backtracking (that machinery exists to protect the real headline
    # number; this mode explicitly wants to see past the cliff it defines). Straight descent:
    # rank -> cut a fixed fraction -> heal -> record circuit_acc/full_cap/cap_ratio every round,
    # until cap_ratio hits the requested ceiling, an accuracy safety floor trips (real collapse,
    # not informative -- e.g. induction's 40e/0.000 crash from earlier today), or the edge floor
    # is reached. Milestone cap_ratio values get tagged post-hoc on the closest real round.
    elif args.diagnostic_sweep:
        milestones = [1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2]
        cut = args.adaptive_cut
        stop_reason = None
        print(f"[diagnostic-sweep] ceiling cap_ratio>={args.diagnostic_sweep_cap_ratio}, "
              f"acc floor={args.diagnostic_acc_floor}, starting from {len(current_keep)}e", flush=True)
        while True:
            n = len(current_keep)
            target = max(args.adaptive_min, int(round(n * (1 - cut))))
            if target >= n or n <= args.adaptive_min:
                stop_reason = f"edge floor ({args.adaptive_min}) reached"
                break
            set_binary_mask(model, current_keep)
            imp = eap_ranking(model, train_ds, collator, device, args.rank_batches, seed=args.seed, ig_steps=args.ig_steps)
            ranked = sorted(current_keep, key=lambda e: float(imp[e]), reverse=True)
            new_keep = set(ranked[:target])
            secs, fdelta = heal(model, new_keep, train_ds, collator, device,
                                args.heal_steps, args.heal_lr, args.heal_batch, args.heal_delta,
                                cap_fn=cap_only, circ_fn=circ_only,
                                early_stop_acc=base["circuit_acc"], early_every=args.adaptive_early_every,
                                freeze_weights=args.freeze_weights, heal_mode=args.heal_mode, weight_decay=args.heal_weight_decay, mix_p=args.mix_p)
            current_keep = new_keep
            realised = circuit_eval.circuit_size(model)
            m = meas()
            m["cap_ratio"] = cap_ratio(args.cap_gate_batches)
            milestone_tag = min(milestones, key=lambda x: abs(x - m["cap_ratio"])) \
                if abs(min(milestones, key=lambda x: abs(x - m["cap_ratio"])) - m["cap_ratio"]) < 0.0125 else None
            row = {"target": target, "realised_edges": realised, **m, "heal_secs": round(secs, 1),
                   "delta": round(fdelta, 3), "phase": "diagnostic_sweep",
                   "cut_frac": round(cut, 3), "milestone": milestone_tag}
            rows.append(row)
            log_edges("diagnostic_sweep", current_keep, row)
            print(f"  [diag-sweep] {realised:5d}e circuit_acc={m['circuit_acc']:.3f} "
                  f"full_cap={m['full_cap']:.3f} cap_ratio={m['cap_ratio']:.4f}"
                  f"{' <milestone ' + str(milestone_tag) + '>' if milestone_tag else ''}", flush=True)
            if m["cap_ratio"] >= args.diagnostic_sweep_cap_ratio:
                stop_reason = f"cap_ratio ceiling ({args.diagnostic_sweep_cap_ratio}) reached"
                break
            if m["circuit_acc"] < args.diagnostic_acc_floor or m["full_cap"] < args.diagnostic_acc_floor:
                stop_reason = f"accuracy safety floor ({args.diagnostic_acc_floor}) tripped"
                break
        print(f"[diagnostic-sweep] STOP: {stop_reason}", flush=True)
        report_extra = {"diagnostic_sweep_stop_reason": stop_reason,
                        "diagnostic_sweep_milestones": [r for r in rows if r.get("milestone")]}
        schedule = []  # skip the fixed-schedule loop below

    for target in schedule:
        if not args.no_prune and target >= len(current_keep):
            continue  # schedule entry not below current size; skip
        # rank the CURRENT kept edges, keep the top-`target`.
        set_binary_mask(model, current_keep)
        if args.no_prune:
            # B1 distillation control: NEVER cut — heal the same LoRA at the FULL mask for
            # the same number of rounds. Tests whether distillation alone concentrates (it
            # should not; only pruning does). realised edges stay = start_n.
            pass
        else:
            imp = static_rank if static_rank is not None else \
                eap_ranking(model, train_ds, collator, device, args.rank_batches, seed=args.seed, ig_steps=args.ig_steps)
            if args.rank_mode == "random":
                # B2 control: ignore importance — keep a RANDOM subset of the current edges.
                shuffled = list(current_keep)
                torch.manual_seed(args.seed + len(current_keep))
                perm = torch.randperm(len(shuffled)).tolist()
                ranked = [shuffled[i] for i in perm]
            elif args.rank_mode == "anti":
                # B2 control: keep the LEAST important edges (negate the ranking).
                ranked = sorted(current_keep, key=lambda e: float(imp[e]))
            else:  # eap (default)
                ranked = sorted(current_keep, key=lambda e: float(imp[e]), reverse=True)
            current_keep = set(ranked[:target])

        # DYNAMIC δ ramp: tighten the leash as the circuit shrinks (the optimum moves up toward the floor).
        round_delta = args.heal_delta
        if args.delta_ramp:
            lo, hi = (float(x) for x in args.delta_ramp.split(","))
            frac = max(0.0, min(1.0, 1.0 - target / max(start_n, 1)))  # 0 at full, 1 at the floor
            round_delta = lo + (hi - lo) * frac

        def _heal_once():
            return heal(model, current_keep, train_ds, collator, device,
                        args.heal_steps, args.heal_lr, args.heal_batch, round_delta,
                        cap_tol=args.heal_cap_tol, dual_lr=args.heal_dual_lr,
                        delta_max=args.heal_delta_max,
                        acc_floor=args.heal_acc_floor, acc_every=args.heal_acc_every,
                        orig_acc=base["full_cap"], cap_fn=cap_only, alpha=args.heal_alpha,
                        circ_floor=args.heal_circ_floor, circ_fn=circ_only,
                        freeze_weights=args.freeze_weights, heal_mode=args.heal_mode, weight_decay=args.heal_weight_decay, mix_p=args.mix_p)

        # heal at the new (smaller) mask; LoRA carries over.
        secs, final_delta = _heal_once()
        realised = circuit_eval.circuit_size(model)
        m = meas()
        # Q3 (adaptive prune-rate): if the circuit underperforms, spend EXTRA heal rounds at
        # this same edge count (slow the schedule) instead of cutting further. Keeps the loss
        # simple; the *schedule* adapts to where the model is struggling.
        retries = 0
        while args.retry_floor > 0.0 and m["circuit_acc"] < args.retry_floor and retries < args.retry_max:
            s2, final_delta = _heal_once()
            secs += s2
            retries += 1
            m = meas()
        if args.probe_every_round:
            m["cap_ratio"] = cap_ratio(args.cap_gate_batches)
        knee = JOINT_LOSS_KNEE.get(target) if task == "ioi" else None
        row = {"target": target, "realised_edges": realised, **m, "heal_secs": round(secs, 1),
               "delta": round(final_delta, 3), "retries": retries, "knee": knee,
               "beats_knee": (knee is not None and m["circuit_acc"] >= knee),
               "milestone": target in MILESTONES}
        if args.measure_panel:
            row["panel"] = measure_panel(model, eval_ds, collator, device, args.panel_batches)
        rows.append(row)
        log_edges("fixed_schedule", current_keep, row)
        tag = " *" if target in MILESTONES else "  "
        print(f"{tag} edges {realised:4d} (target {target:4d}) | acc {m['circuit_acc']:.3f} "
              f"faith {m['faith']:.2f} cap {m['full_cap']:.3f} δ={final_delta:.2f} ({secs:.0f}s) | "
              f"knee={knee} beats={row['beats_knee']}", flush=True)

    # FINAL faithfulness on the SEALED test split (never used by the controller).
    final_test = meas_test() if args.test_data else None
    endpoint_valid_on_test = None
    if final_test is not None:
        print(f"[final/test] circuit={circuit_eval.circuit_size(model)}e | test circuit-acc "
              f"{final_test['circuit_acc']} | test faith-KL {final_test['faith']} "
              f"(eval was best_faithful) on {Path(args.test_data).name}", flush=True)

        # ENDPOINT VALIDATION on the sealed split. The accept predicate `round_ok`
        # runs on eval, and near the cliff a below-floor cut gets up to --adaptive-reheal EXTRA
        # fresh heals, accepted if ANY passes (see the loop above). That is a max-of-N over a
        # noisy 200-example score: E[max of 4] sits ~1.03 sigma (~+0.035 acc) above a single draw.
        # The frozen arm (C0) cannot use this -- heal() early-returns when freeze_weights, and
        # the eval path has no shuffle, so its reheals are bit-identical duplicates. So C1 can accept an endpoint whose TRUE
        # accuracy is below the floor while C0 cannot, and the arms are then NOT at equal capability
        # -- which is exactly what the headline "C1 < C0 at the same capability" asserts.
        # Reading test accuracy does not undo the edge-count selection, but it does tell us whether
        # the endpoint we selected actually clears the bar out of sample. If this is False, the
        # endpoint was bought with luck and must not be reported as faithful.
        if base_test is not None and args.adaptive:
            test_floor = base_test["circuit_acc"] - args.adaptive_tol
            endpoint_valid_on_test = bool(final_test["circuit_acc"] >= test_floor)
            verdict = "PASS" if endpoint_valid_on_test else "FAIL (endpoint does not clear the floor out of sample)"
            print(f"[endpoint-validation] test acc {final_test['circuit_acc']:.3f} vs test floor "
                  f"{test_floor:.3f} (m0_test {base_test['circuit_acc']:.3f} - tol {args.adaptive_tol}) "
                  f"-> {verdict}", flush=True)

    edit = None
    if args.edit_eval:
        edit = edit_eval(model, test_ds, collator, device, current_keep,
                         args.eval_batch, args.eval_batches, args.seed,
                         task=task, gt_generative=gt_generative, tok=tok)
        print(f"[C2 edit] circuit={edit['circuit_edges']}e | target: full {edit['target_full']} "
              f"-> circuit-ablated {edit['target_circuit_ablated']} (random-ablated {edit['target_random_ablated']}) "
              f"| collateral CE: full {edit['collateral_ce_full']} -> circuit-ablated "
              f"{edit['collateral_ce_circuit_ablated']} (random {edit['collateral_ce_random_ablated']})", flush=True)

    cap_probe = None
    if args.capability_probe:
        # No silent fallback to eval: the probe gates accept/reject, so falling back to the
        # controller's own decision split makes the gate self-referential (and violates the
        # always-mixed-corpus rule). Fail loudly instead, and verify the corpus is not leaky.
        if not args.probe_data:
            raise ValueError(
                "a capability gate/probe/abort was requested but --probe-data was not set. "
                "Refusing to fall back to --eval-data: the probe drives accept/reject, so gating "
                "on the decision split is self-referential. Pass data/probe_mixed.jsonl "
                "(slurm/prune_heal_gradual.slurm sets PROBE_DATA to it by default).")
        probe_path = args.probe_data
        assert_probe_disjoint(probe_path, args.eval_data, args.test_data)
        with open(probe_path) as f:
            probe_prompts = [json.loads(l)["clean_prompt"] for l in f if l.strip()]
        cap_probe = capability_probe(model, probe_prompts, tok, device, max_length,
                                     eb=args.eval_batch, nb=args.eval_batches)
        print(f"[capability] unrelated-text perplexity: original {cap_probe['probe_ppl_orig']} "
              f"-> healed {cap_probe['probe_ppl_healed']} (ratio {cap_probe['probe_ppl_ratio']}, "
              f"Δce {cap_probe['probe_delta_ce']}) on {Path(probe_path).name}", flush=True)

    # WiSE-FT-style gamma sweep: scale the healed LoRA update by gamma and re-measure. Scaling
    # only the lora_B matrices scales the LoRA delta (B@A) linearly; gamma=1 is the healed model,
    # gamma=0 is exactly the original weights. If capability recovers at a gamma where the circuit
    # still works, the OOD damage lives in LoRA directions the circuit doesn't need.
    gamma_rows = None
    if args.gamma_sweep:
        gammas = [float(x) for x in args.gamma_sweep.split(",") if x.strip()]
        gp_prompts = probe_prompts
        healed_snap = snapshot_lora(model)
        lora_B = [p for n, p in model.named_parameters() if "lora_B" in n]
        gamma_rows = []
        for g in gammas:
            restore_lora(model, healed_snap)
            with torch.no_grad():
                for p in lora_B:
                    p.data.mul_(g)
            set_binary_mask(model, current_keep)
            m = meas()
            cp = capability_probe(model, gp_prompts, tok, device, max_length,
                                  eb=args.eval_batch, nb=args.eval_batches)
            gamma_rows.append({"gamma": g, **m,
                               "probe_ppl_healed": cp["probe_ppl_healed"],
                               "probe_ppl_ratio": cp["probe_ppl_ratio"]})
            print(f"[gamma {g:4.2f}] circuit_acc={m['circuit_acc']:.3f} faith={m['faith']:.2f} "
                  f"full_cap={m['full_cap']:.3f} | probe ppl ratio {cp['probe_ppl_ratio']}", flush=True)
        restore_lora(model, healed_snap)   # leave the model at gamma=1 (healed)
        set_binary_mask(model, current_keep)

    if args.save_healed:
        sd = Path(args.save_healed)
        sd.mkdir(parents=True, exist_ok=True)
        model.peft_model.save_pretrained(str(sd / "lora_healed"))
        torch.save(model.gate.state_dict(), sd / "gates_healed.pt")
        with open(sd / "kept_edges.json", "w") as f:
            json.dump(sorted(current_keep), f)
        print(f"[save] healed LoRA + gates -> {sd}", flush=True)

    report = {"checkpoint": args.checkpoint, "model": model_path, "task": task, "mode": "gradual",
              "gamma_sweep": gamma_rows,
              "edit_eval": edit, "capability_probe": cap_probe, "final_test": final_test,
              "start": args.start, "schedule": schedule, "rerank": args.rerank,
              "heal_steps_per_round": args.heal_steps, "heal_lr": args.heal_lr,
              "heal_alpha": args.heal_alpha,
              "heal_delta": args.heal_delta, "heal_cap_tol": args.heal_cap_tol,
              "heal_acc_floor": args.heal_acc_floor, "heal_delta_max": args.heal_delta_max,
              "heal_dual_lr": args.heal_dual_lr, "heal_circ_floor": args.heal_circ_floor,
              "cap_gate": args.cap_gate, "cap_abort": args.cap_abort, "cap_gate_calib": cap_gate_calib,
              "retry_floor": args.retry_floor, "retry_max": args.retry_max, "start_edges": start_n,
              # Provenance for the settings that define the SELECTION RULE. Anything that
              # changes which circuit is returned (e.g. a self-calibrated capability gate whose
              # effective value exists only at runtime) must be readable off the artifact.
              "cfg_accept_lcb": args.accept_lcb, "cfg_adaptive_min": args.adaptive_min,
              "cfg_adaptive": bool(args.adaptive), "cfg_freeze_weights": bool(args.freeze_weights),
              "cfg_cap_gate_auto": bool(getattr(args, "cap_gate_auto", False)),
              "cfg_cap_gate_requested": cap_gate_requested,
              "cfg_cap_gate_effective": args.cap_gate,
              "cfg_eval_data": str(args.eval_data), "cfg_eval_batches": args.eval_batches,
              "cfg_test_data": str(args.test_data) if args.test_data else None,
              "cfg_adaptive_tol": args.adaptive_tol, "cfg_heal_mode": args.heal_mode,
              "m0_baseline": base, "m0_baseline_test": base_test,
              "endpoint_valid_on_test": endpoint_valid_on_test, "rounds": rows, **report_extra}
    out = Path(args.out) if args.out else Path(args.checkpoint) / "prune_heal_gradual.json"
    out.parent.mkdir(parents=True, exist_ok=True)   # don't lose a finished run to a missing dir
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
