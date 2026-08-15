"""The ILM preserve-and-sparsify co-training loop (2026-06-23 pivot).

Per step (first-order, single backward):
    1. capture corrupted contributions (detached baseline for OFF edges)
    2. reference forward (LoRA disabled, no grad) -> the frozen ORIGINAL target
    3. masked clean forward -> sparse circuit (must match the original)
    4. full   clean forward -> the moving model (must stay equal to the original)
    L = alpha*KL(orig||masked) + delta*KL(orig||full) + lambda*L0
There is NO ground-truth task loss; capability is held fixed and only sparsity is
pushed. LoRA weights and gate dials update together; lambda is ramped up. In
``freeze_weights`` mode the LoRA is locked and only the gates train (the baseline).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..eval import circuit as circuit_eval
from ..eval.eap import concentration_panel
from .losses import fraction_open, kl_loss
from .setup import split_parameters


class Trainer:
    def __init__(self, cfg: TrainConfig, masked_model, train_ds, eval_ds, collator, device,
                 tokenizer=None):
        self.cfg = cfg
        self.model = masked_model
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.collator = collator
        self.device = device
        self.tokenizer = tokenizer

        lora_params, gate_params = split_parameters(masked_model)
        if cfg.freeze_weights:
            # Baseline mode: lock the model, train ONLY the gates. Capability cannot
            # change -> a pure reference knee (smallest faithful circuit as-is).
            for p in lora_params:
                p.requires_grad_(False)
            param_groups = [{"params": gate_params, "lr": cfg.mask_lr}]
        else:
            param_groups = [
                {"params": lora_params, "lr": cfg.model_lr},
                {"params": gate_params, "lr": cfg.mask_lr},
            ]
        self.optimizer = torch.optim.AdamW(param_groups)

        # Dual-variable (Lagrangian) state. The multipliers live in log-space and are
        # updated by hand each step (dual ascent) -- they are NOT autograd parameters
        # (minimising the loss wrt them would drive them to 0, erasing the pressure).
        self.log_lambda = math.log(max(cfg.loss.lambda_sparse, cfg.lambda_min))
        self.log_delta = math.log(max(cfg.loss.delta_preserve, cfg.delta_min))

        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.eval_log = self.out / "eval_log.jsonl"
        self.train_log = self.out / "train_log.jsonl"      # per-step loss components
        self.samples_log = self.out / "samples.jsonl"      # sample generations per eval
        for f in (self.eval_log, self.train_log, self.samples_log):
            f.unlink(missing_ok=True)

    def _lambda(self, step: int) -> float:
        """Ramp the L0 cost from 0 to its fixed value over the warmup, so the model
        eases into sparsity rather than being shocked into dropping edges at once."""
        warmup = max(int(self.cfg.sparsity_warmup_frac * self.cfg.num_steps), 1)
        frac = min(step / warmup, 1.0)
        return frac * self.cfg.loss.lambda_sparse

    def _target_frac(self, step: int) -> float:
        """Budget mode: ramp the edge budget from the initial open fraction down to
        the target over the warmup, so the model tightens gently (mirrors the fixed-
        lambda ramp) instead of being shocked into a tiny circuit at once."""
        warmup = max(int(self.cfg.sparsity_warmup_frac * self.cfg.num_steps), 1)
        frac = min(step / warmup, 1.0)
        start = self.cfg.init_open_prob
        return start + frac * (self.cfg.target_frac_open - start)

    def _update_duals(self, parts: dict, step: int) -> None:
        """Dual ascent on the Lagrange multipliers (no autograd). Each multiplier rises
        when its constraint is violated and falls when it is satisfied:
          budget:    log_lambda += lr * (frac_open - target_frac)      [too many edges -> raise price]
          tolerance: log_delta  += lr * (KL(orig||full) - capability_tol) [drifted -> tighten leash]
        Multiplicative (log-space) update keeps the multiplier positive and scale-free."""
        cfg = self.cfg
        if cfg.lambda_mode == "budget":
            target = self._target_frac(step)
            self.log_lambda += cfg.dual_lr_lambda * (parts["frac_open"] - target)
            self.log_lambda = min(max(self.log_lambda, math.log(cfg.lambda_min)),
                                  math.log(cfg.lambda_max))
            parts["target_frac"] = target
        if cfg.delta_mode == "tolerance" and not cfg.freeze_weights:
            self.log_delta += cfg.dual_lr_delta * (parts["preserve"] - cfg.capability_tol)
            self.log_delta = min(max(self.log_delta, math.log(cfg.delta_min)),
                                 math.log(cfg.delta_max))
            parts["capability_tol"] = cfg.capability_tol

    def _loader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size,
            shuffle=True, collate_fn=self.collator, drop_last=True,
        )

    def train(self) -> None:
        cfg, lc = self.cfg, self.cfg.loss
        self.model.train()
        step = 0
        t0 = time.time()
        while step < cfg.num_steps:
            for batch in self._loader():
                if step >= cfg.num_steps:
                    break
                # Effective penalty weights: fixed (ramped) or the current dual value.
                lam = math.exp(self.log_lambda) if cfg.lambda_mode == "budget" else self._lambda(step)
                dlt = math.exp(self.log_delta) if cfg.delta_mode == "tolerance" else lc.delta_preserve
                loss, parts = self._step(batch, lam, dlt)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]], cfg.grad_clip
                )
                self.optimizer.step()
                # Dual ascent AFTER the primal step, using this step's constraint values.
                self._update_duals(parts, step)

                if step % 10 == 0:
                    on = self.model.gate.num_open()
                    tgt = f" tgt {parts['target_frac']:.3f}" if "target_frac" in parts else ""
                    print(f"step {step:5d} | loss {loss.item():.4f} | "
                          f"circuit {parts['circuit']:.3f} preserve {parts['preserve']:.3f} "
                          f"frac_open {parts['frac_open']:.3f}{tgt} sparse {parts['sparse']:.3f} | "
                          f"edges_on {on}/{self.model.graph.num_edges} | "
                          f"lambda {parts['lambda']:.4f} delta {parts['delta']:.3f} | "
                          f"{time.time()-t0:.0f}s", flush=True)
                    with open(self.train_log, "a") as f:
                        f.write(json.dumps({"step": step, "loss": loss.item(),
                                            "edges_on": on, **parts}) + "\n")

                if step > 0 and step % cfg.eval_every == 0:
                    self.evaluate(step)
                    self.model.train()
                step += 1

        self.evaluate(step)
        self.save()

    def _step(self, batch, lambda_sparse: float, delta_preserve: float):
        cfg, lc = self.cfg, self.cfg.loss
        clean_ids = batch["clean_ids"].to(self.device)
        clean_mask = batch["clean_mask"].to(self.device)
        corr_ids = batch["corr_ids"].to(self.device)
        corr_mask = batch["corr_mask"].to(self.device)

        corrupted = self.model.capture_contributions(corr_ids, corr_mask)

        # The frozen ORIGINAL model (LoRA disabled) — the target both the sparse
        # circuit and the (possibly moving) full model must match. No grad, no labels.
        with self.model.peft_model.disable_adapter(), torch.no_grad():
            logits_ref = self.model(clean_ids, clean_mask, masked=False)

        logits_masked = self.model(clean_ids, clean_mask, corrupted=corrupted, masked=True)

        expected_l0 = self.model.gate.expected_l0()
        frac_open = fraction_open(expected_l0, self.model.graph.num_edges)

        # Circuit-loss mask: normally every real (non-pad) position. For "agreement", the
        # answer slot is one of ~49 near-interchangeable verbs drawn arbitrarily per example
        # -- nothing in the prompt disambiguates which one, so even the frozen ORIGINAL
        # model can't predict it there. Distilling THAT into the (approximate) circuit is
        # pure noise on top of noise. Exclude it from l_circuit only -- NOT from l_preserve:
        # l_preserve compares the full model to its own original self at that position, which
        # is a real, low-noise signal (same architecture, same weights modulo LoRA) regardless
        # of whether the position's content is predictable in an absolute sense. A first
        # attempt excluded the position from BOTH terms and made things WORSE (full-model acc
        # dropped to 0.14, below even the un-fixed run's 0.25) -- removing l_preserve's
        # coverage there removed the only thing anchoring that position, so nothing stopped
        # it drifting while sparsity pressure kept pulling on the shared LoRA weights.
        circuit_mask = clean_mask
        if cfg.task == "agreement":
            labels = batch["labels"].to(self.device)
            answer_mask = torch.zeros_like(clean_mask)
            answer_mask[:, :-1] = (labels[:, 1:] != -100).to(answer_mask.dtype)
            circuit_mask = clean_mask * (1 - answer_mask)

        # (a) sparse circuit must reproduce the ORIGINAL; (b) the full model must stay
        # equal to the original (capability leash; skipped when weights are frozen);
        # (c) use as few edges as possible. NO ground-truth task loss.
        l_circuit = kl_loss(logits_masked, logits_ref, circuit_mask)
        if cfg.freeze_weights:
            l_preserve = torch.zeros((), device=logits_masked.device)
        else:
            logits_full = self.model(clean_ids, clean_mask, masked=False)
            l_preserve = kl_loss(logits_full, logits_ref, clean_mask)
        l_sparse = lambda_sparse * expected_l0

        loss = lc.alpha_circuit * l_circuit + delta_preserve * l_preserve + l_sparse

        parts = {
            "circuit": l_circuit.item(), "preserve": float(l_preserve),
            "frac_open": frac_open.item(), "sparse": l_sparse.item(),
            "lambda": lambda_sparse, "delta": delta_preserve,
        }
        return loss, parts

    def evaluate(self, step: int) -> dict:
        cfg = self.cfg
        self.model.eval()
        # Circuit accuracy (gate-dependent) is the knee y-axis; generative accuracy
        # of the full model is the capability anchor; faithfulness says how well the
        # trimmed circuit reproduces the full model.
        if cfg.task == "gt" and getattr(self.eval_ds, "generative", False):
            # Greater-than knee y-axis (Llama/Qwen): generate the year THROUGH the circuit.
            circuit_acc = circuit_eval.greater_than_generative_accuracy(
                self.model, self.tokenizer, self.eval_ds.examples, self.device, masked=True,
                max_examples=cfg.eval_batches * cfg.batch_size,
            )
        elif cfg.task == "gt":
            # Greater-than knee y-axis: argmax-valid through the MASKED circuit (GPT-2 single-token).
            circuit_acc = circuit_eval.greater_than_accuracy(
                self.model, self.eval_ds, self.collator, self.device, masked=True,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
        elif cfg.task == "agreement":
            # Agreement knee y-axis: argmax-valid (same-number verb set) through the MASKED circuit.
            circuit_acc = circuit_eval.agreement_accuracy(
                self.model, self.eval_ds, self.collator, self.device, masked=True,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
        else:
            circuit_acc = circuit_eval.circuit_token_accuracy(
                self.model, self.eval_ds, self.collator, self.device,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
        faith = circuit_eval.circuit_faithfulness(
            self.model, self.eval_ds, self.collator, self.device, batch_size=cfg.batch_size,
        )
        # Capability anchor (full model, masks OFF) — must stay flat as edges drop.
        # GSM8K: free-generate + match the final number. IOI: top-token name match.
        if cfg.task in ("ioi", "doc", "induction"):
            # Single fixed correct token -> exact-top-token capability anchor.
            acc = circuit_eval.ioi_accuracy(
                self.model, self.eval_ds, self.collator, self.device,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
            gen = {"accuracy": acc, "unparseable_rate": 0.0, "samples": []}
        elif cfg.task == "agreement":
            # Agreement capability anchor: argmax-valid (same-number verb set), full model.
            acc = circuit_eval.agreement_accuracy(
                self.model, self.eval_ds, self.collator, self.device, masked=False,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
            gen = {"accuracy": acc, "unparseable_rate": 0.0, "samples": []}
        elif cfg.task == "gt" and getattr(self.eval_ds, "generative", False):
            # Greater-than capability anchor (Llama/Qwen): generate the year, full model.
            acc = circuit_eval.greater_than_generative_accuracy(
                self.model, self.tokenizer, self.eval_ds.examples, self.device, masked=False,
                max_examples=cfg.eval_batches * cfg.batch_size,
            )
            gen = {"accuracy": acc, "unparseable_rate": 0.0, "samples": []}
        elif cfg.task == "gt":
            # Greater-than capability anchor: argmax-valid through the FULL model.
            acc = circuit_eval.greater_than_accuracy(
                self.model, self.eval_ds, self.collator, self.device, masked=False,
                batch_size=cfg.batch_size, num_batches=cfg.eval_batches,
            )
            gen = {"accuracy": acc, "unparseable_rate": 0.0, "samples": []}
        else:
            gen = circuit_eval.generative_accuracy(
                self.model.peft_model, self.tokenizer, self.eval_ds.examples, self.device,
                max_new_tokens=cfg.eval_max_new_tokens, batch_size=cfg.batch_size,
                max_examples=cfg.eval_batches * cfg.batch_size, return_details=True,
            )
        n_edges = circuit_eval.circuit_size(self.model)

        # EAP panel on one eval batch (needs grad -> not under no_grad).
        panel = self._eap_panel()

        record = {"step": step, "accuracy": circuit_acc, "gen_accuracy": gen["accuracy"],
                  "unparseable_rate": gen["unparseable_rate"], "faithfulness": faith,
                  "num_edges": n_edges, "total_edges": self.model.graph.num_edges, **panel}
        with open(self.eval_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        # Dump a few generations so we can SEE what the model produces.
        with open(self.samples_log, "a") as f:
            for s in gen["samples"]:
                f.write(json.dumps({"step": step, **s}) + "\n")
        print(f"[eval] step {step} | circuit_acc {circuit_acc:.3f} gen_acc {gen['accuracy']:.3f} "
              f"unparseable {gen['unparseable_rate']:.2f} faith_kl {faith:.3f} | "
              f"edges {n_edges}/{record['total_edges']} | "
              f"entropy {panel['normalized_entropy']:.3f} PR {panel['participation_ratio']:.1f} "
              f"kurt {panel['kurtosis']:.2f}", flush=True)
        return record

    def _eap_panel(self) -> dict:
        loader = DataLoader(self.eval_ds, batch_size=self.cfg.batch_size, collate_fn=self.collator)
        batch = next(iter(loader))
        clean_ids = batch["clean_ids"].to(self.device)
        clean_mask = batch["clean_mask"].to(self.device)
        corr_ids = batch["corr_ids"].to(self.device)
        corr_mask = batch["corr_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        corrupted = self.model.capture_contributions(corr_ids, corr_mask)
        metric_fn = circuit_eval.make_logit_diff_metric(labels)
        scores = self.model.edge_attributions(clean_ids, clean_mask, corrupted, metric_fn)
        self.optimizer.zero_grad(set_to_none=True)
        return concentration_panel(scores)

    def save(self) -> None:
        self.model.peft_model.save_pretrained(str(self.out / "lora"))
        torch.save(self.model.gate.state_dict(), self.out / "gates.pt")
        with open(self.out / "config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        print(f"[save] wrote LoRA + gates to {self.out}", flush=True)
