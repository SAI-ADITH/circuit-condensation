#!/usr/bin/env python3
"""The Zhang same-axis contrast: ordinary task fine-tuning, then measure concentration.

Our method (C1) reshapes a model so a *tiny* circuit suffices. Zhang-2026's claim is
the opposite for plain RL: optimizing the task makes the circuit MORE diffuse. To put
our headline sentence on solid ground we must run that comparison on our OWN setup:
take the same model+task, fine-tune it the ordinary way (no gates, no sparsity — just
"get the task right"), and read our EAP concentration panel before vs after.

Two modes of "ordinary fine-tuning":
  --mode sft : supervised fine-tune. Cross-entropy on the gold answer tokens. The cheap
               proxy for "task-optimizing fine-tuning" (Step E0, the kill-test).
  --mode rl  : REINFORCE. Sample a generation, reward = answer correct (1/0), push up the
               log-prob of rewarded samples. The actual thing Zhang studies (Step E1).

Crucially the EAP panel is measured with **all gates open = the full model's circuit**,
so the only thing that changed between before/after is the WEIGHTS — exactly Zhang's axis
(does task fine-tuning diffuse the full model's circuit?). Gates never train here; this is
a normal LoRA fine-tune, the disentangled wrapper is reused only to run the panel.

PASS for E0: SFT does NOT concentrate (entropy/PR flat-or-up). If even gentle SFT already
concentrates, the "task fine-tuning diffuses" premise is false on our setup -> stop before
building RL. PASS for E1: RL diffuses (entropy up / PR up) while our C1 concentrates (down).

    python scripts/task_finetune.py --mode sft --task ioi --graph-granularity head \
        --model-path .../models/gpt2 --data .../ioi_train.jsonl --eval-data .../ioi_eval.jsonl \
        --output-dir .../checkpoints/ft_ioi_sft --num-steps 1000 --max-length 64
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.config import TrainConfig
from ilm.data.dataset import PairCollator, PairDataset, load_pairs
from ilm.data.ioi import IOIDataset, load_samples
from ilm.data.greater_than import GreaterThanDataset, load_samples as load_gt_samples
from ilm.data.docstring import DocstringDataset, load_samples as load_doc_samples
from ilm.data.induction import InductionDataset, load_samples as load_ind_samples
from ilm.data.agreement import AgreementDataset, load_samples as load_agr_samples
from ilm.eval import circuit as circuit_eval
from ilm.eval.eap import concentration_panel
from ilm.training.setup import build_masked_model, load_tokenizer

GATE_OPEN_LOGALPHA = 20.0  # sigmoid(20)~1 -> every edge ON (panel reads the FULL model)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sft", "rl", "grpo"], required=True,
                   help="sft=cross-entropy; rl=plain REINFORCE (unstable, deprecated); "
                        "grpo=Group Relative Policy Optimization (Zhang's algorithm: group-normalized "
                        "advantages + KL leash to frozen reference — the STABLE RL).")
    p.add_argument("--task", choices=["ioi", "gsm8k", "gt", "doc", "induction", "agreement"], required=True)
    p.add_argument("--gt-mode", default="generative", choices=["generative", "token"],
                   help="greater-than eval mode (generative number-match vs single-token argmax).")
    p.add_argument("--graph-granularity", default="block", choices=["block", "head"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (matches model_lr).")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--panel-batches", type=int, default=8, help="batches averaged for the EAP panel.")
    p.add_argument("--eval-batches", type=int, default=25, help="batches for capability accuracy.")
    p.add_argument("--max-new-tokens", type=int, default=256, help="RL/gen generation budget.")
    p.add_argument("--group-size", type=int, default=8, help="GRPO: completions sampled per prompt.")
    p.add_argument("--beta", type=float, default=0.04, help="GRPO: KL-penalty weight to the frozen reference.")
    p.add_argument("--temperature", type=float, default=1.0, help="GRPO/RL sampling temperature.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_datasets(task, data, eval_data, tokenizer, max_length, gt_generative=True):
    if task == "ioi":
        return (IOIDataset(load_samples(data), tokenizer, max_length),
                IOIDataset(load_samples(eval_data), tokenizer, max_length))
    if task == "gt":
        return (GreaterThanDataset(load_gt_samples(data), tokenizer, max_length, generative=gt_generative),
                GreaterThanDataset(load_gt_samples(eval_data), tokenizer, max_length, generative=gt_generative))
    if task == "doc":
        return (DocstringDataset(load_doc_samples(data), tokenizer, max_length),
                DocstringDataset(load_doc_samples(eval_data), tokenizer, max_length))
    if task == "induction":
        return (InductionDataset(load_ind_samples(data), tokenizer, max_length),
                InductionDataset(load_ind_samples(eval_data), tokenizer, max_length))
    if task == "agreement":
        return (AgreementDataset(load_agr_samples(data), tokenizer, max_length),
                AgreementDataset(load_agr_samples(eval_data), tokenizer, max_length))
    return (PairDataset(load_pairs(data), tokenizer, max_length),
            PairDataset(load_pairs(eval_data), tokenizer, max_length))


@torch.no_grad()
def force_gates_open(model) -> None:
    """Pin every gate ON so the panel measures the FULL model's circuit (weights-only axis)."""
    model.gate.log_alpha.data.fill_(GATE_OPEN_LOGALPHA)
    for p in model.gate.parameters():
        p.requires_grad_(False)


def measure_panel(model, eval_ds, collator, device, batch_size, panel_batches) -> dict:
    """Average the concentration panel over batches, gates open (full model). Uses EAP-IG
    (integrated gradients, see edge_masked_model.edge_attributions_ig) instead of plain
    single-point EAP -- matches scripts/prune_heal_gradual.py's measure_panel() exactly, so
    the 'diffuse' (this file) and 'concentrate' (prune_heal_gradual.py) axes stay directly
    comparable. Confirmed empirically on IOI/GPT-2: EAP-IG ranks edges at least as well as
    plain EAP everywhere, with the biggest gain right at the accuracy cliff."""
    model.eval()
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)
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
    return {k: acc[k] / max(n, 1) for k in keys}


def measure_capability(model, task, eval_ds, collator, tokenizer, device, batch_size, eval_batches,
                       gt_generative=True):
    """Full-model (gates open) task accuracy, dispatched per task exactly like prune_heal_gradual.measure()."""
    model.eval()
    # Per-task forced-choice dispatch, mirroring prune_heal_gradual.measure() EXACTLY so the
    # 'diffuse' arm's capability is on the same ruler as the condensation grid's. The old
    # blanket ioi_accuracy (argmax-over-vocab) read agreement ~0.00 -- see eval-method-overhaul
    # memory -- which invalidated every agreement diffusion cell.
    if task == "agreement":
        return circuit_eval.agreement_accuracy(model, eval_ds, collator, device,
                                               masked=False, batch_size=batch_size, num_batches=eval_batches)
    if task == "ioi":
        return circuit_eval.ioi_forced_choice_accuracy(model, eval_ds, collator, device,
                                                       masked=False, batch_size=batch_size, num_batches=eval_batches)
    if task == "induction":
        return circuit_eval.induction_pool_accuracy(model, eval_ds, collator, device,
                                                    masked=False, batch_size=batch_size, num_batches=eval_batches)
    if task == "doc":  # single-token answer: teacher-forced argmax == accuracy
        return circuit_eval.ioi_accuracy(model, eval_ds, collator, device,
                                         batch_size=batch_size, num_batches=eval_batches)
    if task == "gt" and gt_generative:
        return circuit_eval.greater_than_generative_accuracy(
            model, tokenizer, eval_ds.examples, device, masked=False,
            max_examples=eval_batches * batch_size)
    if task == "gt":
        return circuit_eval.greater_than_accuracy(
            model, eval_ds, collator, device, masked=False,
            batch_size=batch_size, num_batches=eval_batches)
    return circuit_eval.generative_accuracy(
        model.peft_model, tokenizer, eval_ds.examples, device,
        batch_size=batch_size, max_examples=eval_batches * batch_size)


def sft_loss(model, batch, device):
    """Cross-entropy on the supervised answer tokens through the FULL model (masked=False)."""
    ids = batch["clean_ids"].to(device)
    mask = batch["clean_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(ids, mask, masked=False)
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def rl_step(model, tokenizer, prompts, golds, task, device, max_new_tokens, baseline):
    """REINFORCE: sample a generation, reward=correct(1/0), loss=-(reward-baseline)*sum logprob.

    Greedy would give zero learning signal; we sample. The baseline (running mean reward)
    is the standard variance reduction. Reward is the SAME correctness used for accuracy.
    ``prompts``/``golds`` are plain lists pulled from the dataset's ``examples``.
    """
    pad = tokenizer.pad_token_id
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.peft_model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95,
                temperature=1.0, pad_token_id=pad)
        gen = out[:, plen:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        rewards = torch.tensor([float(_reward(task, t, g)) for t, g in zip(texts, golds)],
                               device=device)
        # log-prob of the sampled continuation under the CURRENT policy (with grad).
        full_mask = (out != pad).long()
        logits = model.peft_model(input_ids=out, attention_mask=full_mask).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = out[:, 1:]
        tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        gen_pos = torch.zeros_like(tgt, dtype=torch.bool)
        gen_pos[:, plen - 1:] = True
        gen_pos &= (tgt != pad)
        seq_logp = (tok_logp * gen_pos).sum(dim=1)
        adv = rewards - baseline
        loss = -(adv * seq_logp).mean()
    finally:
        tokenizer.padding_side = side
    return loss, rewards.mean().item()


def _reward(task, text, ex):
    """Reward = the SAME correctness criterion the task's accuracy metric uses (a mismatched
    reward silently starves RL of signal — the doc/gt bug this fixes). ``ex`` is the dataset
    example dict; a bare gold string is accepted as a fallback."""
    gold = ex.get("clean_answer") if isinstance(ex, dict) else ex
    if task in ("ioi", "doc", "induction", "agreement"):
        # single-token word answer: correct iff the generation starts with it
        return float(text.strip().startswith(str(gold).strip()))
    if task == "gt":
        # greater-than: ANY end year with YY > start_yy is correct (not just the gold token)
        m = re.search(r"\d{2,4}", text.strip())
        if m is None:
            return 0.0
        y = int(m.group())
        yy, cent = (y % 100, y // 100) if y >= 100 else (y, None)
        if isinstance(ex, dict) and "start_yy" in ex:
            if cent is not None and cent != ex.get("century"):
                return 0.0
            return float(yy > ex["start_yy"])
        return float(str(y) == str(gold))
    # gsm8k / numeric-extraction tasks
    pred = circuit_eval._extract_pred(text)
    correct = pred is not None and pred == int(gold)
    fmt = 0.1 if "####" in text else 0.0          # small bonus to encourage a clean, terminating answer
    return float(correct) + fmt


def grpo_step(model, tokenizer, prompts, golds, task, device, max_new_tokens,
              group_size, beta, temperature):
    """One GRPO step (Group Relative Policy Optimization — Zhang/DeepSeek's algorithm).

    For each prompt sample ``group_size`` completions; reward each; the advantage is the
    reward Z-SCORED WITHIN ITS GROUP (no value network needed). The loss pushes up the
    log-prob of above-average completions and is leashed by a per-token KL penalty to the
    FROZEN reference (LoRA disabled) — the k3 unbiased estimator, always >= 0. That KL leash
    is exactly what plain REINFORCE lacked (it diverged); GRPO keeps the policy near its
    original self while still chasing reward.
    """
    pad = tokenizer.pad_token_id
    P = len(prompts)
    rep_prompts = [p for p in prompts for _ in range(group_size)]
    rep_golds = [g for g in golds for _ in range(group_size)]
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(rep_prompts, return_tensors="pt", padding=True,
                        add_special_tokens=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.peft_model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95,
                temperature=temperature, pad_token_id=pad)
        texts = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
        rewards = torch.tensor([_reward(task, t, g) for t, g in zip(texts, rep_golds)],
                               device=device, dtype=torch.float32)
        # Group-relative advantage: z-score within each group of `group_size`.
        R = rewards.view(P, group_size)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)

        full_mask = (out != pad).long()
        tgt = out[:, 1:]
        gen_pos = torch.zeros_like(tgt, dtype=torch.bool)
        gen_pos[:, plen - 1:] = True
        gen_pos &= (tgt != pad)

        # policy log-probs (with grad)
        logits = model.peft_model(input_ids=out, attention_mask=full_mask).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # reference log-probs (LoRA disabled, no grad)
        with torch.no_grad(), model.peft_model.disable_adapter():
            rlogits = model.peft_model(input_ids=out, attention_mask=full_mask).logits
            rlogp = F.log_softmax(rlogits[:, :-1, :].float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        log_ratio = rlogp - logp                       # log(pi_ref / pi)
        kl = torch.exp(log_ratio) - log_ratio - 1.0    # k3 estimator, >= 0
        per_tok = -(adv.unsqueeze(1) * logp) + beta * kl
        loss = (per_tok * gen_pos).sum() / gen_pos.sum().clamp(min=1)
    finally:
        tokenizer.padding_side = side
    # report raw correctness mean (strip the +0.1 format bonus) for an honest reward number
    raw_correct = (rewards >= 1.0).float().mean().item()
    return loss, raw_correct, kl[gen_pos].mean().item()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | mode: {args.mode} | task: {args.task} | model: {args.model_path}",
          flush=True)

    cfg = TrainConfig()
    cfg.model_path = args.model_path
    cfg.graph_granularity = args.graph_granularity
    cfg.init_open_prob = 0.95
    tokenizer = load_tokenizer(args.model_path)
    model = build_masked_model(cfg, device)
    force_gates_open(model)
    print(f"Circuit graph: {model.graph.num_layers} layers, {model.graph.num_edges} edges "
          f"(gates pinned OPEN = full model on the panel)", flush=True)

    gt_generative = (args.gt_mode == "generative")
    train_ds, eval_ds = build_datasets(args.task, args.data, args.eval_data,
                                       tokenizer, args.max_length, gt_generative=gt_generative)
    collator = PairCollator(tokenizer.pad_token_id)
    print(f"train: {len(train_ds)} | eval: {len(eval_ds)}", flush=True)

    # LoRA params only (gates frozen) — this is an ordinary fine-tune.
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "gate" not in n]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "ft_log.jsonl"
    log.unlink(missing_ok=True)

    # --- BEFORE: panel + capability on the untouched model ---
    panel_before = measure_panel(model, eval_ds, collator, device, args.batch_size, args.panel_batches)
    cap_before = measure_capability(model, args.task, eval_ds, collator, tokenizer, device,
                                    args.batch_size, args.eval_batches, gt_generative=gt_generative)
    print(f"[before] cap_acc {cap_before:.3f} | entropy {panel_before['normalized_entropy']:.3f} "
          f"PR {panel_before['participation_ratio']:.1f} kurt {panel_before['kurtosis']:.2f}",
          flush=True)

    # --- TRAIN ---
    model.train()
    baseline, step, t0 = 0.0, 0, time.time()
    if args.mode == "grpo":
        # GRPO (Zhang/DeepSeek): `--batch-size` prompts per step, each sampled `--group-size`
        # times; group-normalized advantage + KL leash to the frozen reference. Lower lr.
        items = [(e["prompt"], e) for e in train_ds.examples]   # pass the full example (reward may need start_yy)
        g = torch.Generator().manual_seed(args.seed)
        while step < args.num_steps:
            idx = torch.randint(len(items), (args.batch_size,), generator=g).tolist()
            prompts = [items[i][0] for i in idx]
            golds = [items[i][1] for i in idx]
            loss, reward, kl = grpo_step(model, tokenizer, prompts, golds, args.task, device,
                                         args.max_new_tokens, args.group_size, args.beta,
                                         args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
            optimizer.step()
            if step % 20 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | reward(correct) {reward:.3f} "
                      f"| KL {kl:.4f} | {time.time()-t0:.0f}s", flush=True)
                with open(log, "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss.item(),
                                        "reward": reward, "kl": kl}) + "\n")
            step += 1
    elif args.mode == "rl":
        # RL reads raw prompts+gold answers from the dataset (the collated batch has no
        # prompts); sample a fresh random minibatch of indices each step.
        rl_items = [(e["prompt"], e) for e in train_ds.examples]   # full example (reward may need start_yy)
        g = torch.Generator().manual_seed(args.seed)
        while step < args.num_steps:
            idx = torch.randint(len(rl_items), (args.batch_size,), generator=g).tolist()
            prompts = [rl_items[i][0] for i in idx]
            golds = [rl_items[i][1] for i in idx]
            loss, reward = rl_step(model, tokenizer, prompts, golds, args.task, device,
                                   args.max_new_tokens, baseline)
            baseline = 0.95 * baseline + 0.05 * reward  # EMA reward baseline
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
            optimizer.step()
            if step % 50 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | reward {reward:.3f} "
                      f"base {baseline:.3f} | {time.time()-t0:.0f}s", flush=True)
                with open(log, "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss.item(), "reward": reward}) + "\n")
            step += 1
    else:
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collator, drop_last=True)
        while step < args.num_steps:
            for batch in loader:
                if step >= args.num_steps:
                    break
                loss = sft_loss(model, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
                optimizer.step()
                if step % 50 == 0:
                    print(f"step {step:5d} | loss {loss.item():.4f} | {time.time()-t0:.0f}s",
                          flush=True)
                    with open(log, "a") as f:
                        f.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
                step += 1

    # --- AFTER: panel + capability on the fine-tuned model ---
    panel_after = measure_panel(model, eval_ds, collator, device, args.batch_size, args.panel_batches)
    cap_after = measure_capability(model, args.task, eval_ds, collator, tokenizer, device,
                                   args.batch_size, args.eval_batches, gt_generative=gt_generative)
    print(f"[after]  cap_acc {cap_after:.3f} | entropy {panel_after['normalized_entropy']:.3f} "
          f"PR {panel_after['participation_ratio']:.1f} kurt {panel_after['kurtosis']:.2f}",
          flush=True)

    # --- VERDICT ---
    d_ent = panel_after["normalized_entropy"] - panel_before["normalized_entropy"]
    d_pr = panel_after["participation_ratio"] - panel_before["participation_ratio"]
    d_kurt = panel_after["kurtosis"] - panel_before["kurtosis"]
    diffuse = d_ent > 0 and d_pr > 0  # entropy & PR up = more diffuse (Zhang's direction)
    report = {"mode": args.mode, "task": args.task, "granularity": args.graph_granularity,
              "model_path": args.model_path, "num_steps": args.num_steps, "lr": args.lr,
              "capability_before": cap_before, "capability_after": cap_after,
              "panel_before": panel_before, "panel_after": panel_after,
              "delta_entropy": d_ent, "delta_PR": d_pr, "delta_kurtosis": d_kurt,
              "direction": "DIFFUSE (Zhang)" if diffuse else "concentrate/mixed"}
    with open(out / "contrast_report.json", "w") as f:
        json.dump(report, f, indent=2)
    # Save weights so the panel/eval can be re-run later.
    model.peft_model.save_pretrained(str(out / "lora"))
    torch.save(model.gate.state_dict(), out / "gates.pt")
    cfg.task = args.task
    cfg.data_path, cfg.eval_data_path = args.data, args.eval_data
    cfg.max_length, cfg.output_dir = args.max_length, str(out)
    with open(out / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print("\n==== CONTRAST ====")
    print(f"  capability: {cap_before:.3f} -> {cap_after:.3f}")
    print(f"  entropy:    {panel_before['normalized_entropy']:.3f} -> "
          f"{panel_after['normalized_entropy']:.3f}  (Δ {d_ent:+.3f}, up=diffuse)")
    print(f"  PR:         {panel_before['participation_ratio']:.1f} -> "
          f"{panel_after['participation_ratio']:.1f}  (Δ {d_pr:+.1f}, up=diffuse)")
    print(f"  kurtosis:   {panel_before['kurtosis']:.2f} -> "
          f"{panel_after['kurtosis']:.2f}  (Δ {d_kurt:+.2f}, down=diffuse)")
    print(f"  VERDICT: {report['direction']}")
    print(f"Wrote {out/'contrast_report.json'}", flush=True)


if __name__ == "__main__":
    main()
