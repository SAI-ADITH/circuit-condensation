# Circuit Condensation

**Post-training that makes a model's mechanism small enough to audit.**

Mechanistic interpretability tries to find the small subgraph of a network — the *circuit* — responsible for a behavior. The usual approach takes a frozen model and searches harder for that circuit. The premise here is the opposite: **don't find a circuit, make one.** A short post-training pass reshapes an already-capable model so a target behavior's causal circuit collapses into a subgraph small enough to verify exhaustively.

On a 96-run grid (4 behaviors × 8 models × 3 seeds), the condensed circuit is a **median 6.8× smaller** than what frozen-model discovery recovers on the same model at a matched selection rule, and **93/96** condensed endpoints clear a sealed test-set bar read exactly once.

---

## The method

Attach LoRA adapters and a learnable on/off gate to every edge of the model's computation graph, then run a controller that repeatedly:

1. **Rank** the surviving edges by attribution (EAP-IG — integrated gradients over the corrupted→clean path).
2. **Cut** a margin-proportional fraction of the lowest-ranked edges. An edge that is "off" reads its activation from a corrupted counterfactual input rather than being zeroed, which keeps the model in-distribution.
3. **Heal** the LoRA to restore the behavior through what remains, under an accept/reject **capability gate** — a bootstrap-calibrated perplexity ratio on a held-out, off-target corpus. A round that damages general capability is rejected and rolled back.

The loop stops at the smallest circuit that still passes on held-out validation data. Concentration is **measured, not optimized**: the training signal is task accuracy plus the capability probe, never the concentration metric itself.

Full controller: [`scripts/prune_heal_gradual.py`](scripts/prune_heal_gradual.py). Graph and gates: [`ilm/masking/`](ilm/masking/).

---

## Results

Grid: **4 behaviors** (indirect object identification, subject–verb agreement, induction, docstring completion) × **8 models** (GPT-2 124M, Llama-3.2 1B/3B, Gemma-3 1B/4B, Qwen3 0.6B/1.7B/4B) × **3 seeds**, at attention-head granularity.

**Selection rule, applied identically to every arm:** the smallest circuit whose accuracy on a 2,000-example validation split is within ε=0.05 of that arm's *own* full-graph accuracy. The 1,000-example test split is read once, afterward.

| Comparison | Median size ratio | Cells where condensed is smaller | Sign test |
|---|---|---|---|
| vs. **EAP-IG** (frozen discovery) | **6.8×** | 32/32 | p = 2.6e-06 |
| vs. **C0** (same controller, weights frozen) | **6.7×** | 32/32 | p = 2.6e-06 |
| vs. **Random ranking** | 69.3× | 32/32 | p = 4.7e-10 |

The **C0** row is the load-bearing one: identical gates, identical pruning loop, identical healing schedule, with only the weights frozen. It isolates the gain to the model changing rather than to a better search.

**What the small circuits buy — exhaustive verification.** At a few dozen edges you can enumerate every subset and every pair. Across 62 circuits we ran 169k pairwise edge ablations; the frozen circuits would need **35–13,300× more** ablations to be checked the same way. This is not "we did it better" — it is a check that is infeasible on the frozen side at any budget.

**What it costs.** A circuit is only defined relative to the counterfactual corruption used to build it. Scored against corruption families never trained on, the condensed circuit holds 0.64–0.66 where frozen circuits get 0.45–0.48 — but that is well below its 0.96 on the family it was trained with. Concentration is distribution-specific, and the repo reports that rather than hiding it.

---

## Layout

```
ilm/                    the library
  masking/              edge graph, hard-concrete gates, disentangled residual stream
  training/             co-training loop and losses
  discovery/            frozen-model baselines (EAP, EAP-IG, ACDC, random)
  eval/                 accuracy, faithfulness, concentration panel
  data/                 per-behavior clean/corrupted pair construction
scripts/                pipeline entry points (see below)
slurm/                  cluster launchers for each arm
results/
  condensation/         per-cell endpoints for the condensed (C1) and frozen-weight (C0) arms
  discovery/            frozen-baseline curves on the shared validation split
  verification/         exhaustive subset and all-pairs interaction results
  valselection_headline.json    the numbers in the table above
```

## Running it

```bash
pip install -r requirements.txt

# 1. build clean/corrupted pairs for a behavior
python scripts/prepare_ioi.py --model gpt2

# 2. attach fresh LoRA + gates (no pre-training; gates start open)
python scripts/build_fresh_checkpoint.py --task ioi --model gpt2 --granularity head

# 3. condense: rank -> cut -> heal, under the capability gate
python scripts/prune_heal_gradual.py --task ioi --model gpt2 --adaptive --cap-gate 1.05

# 4. the frozen-discovery comparison, on the same selection rule
python scripts/run_discovery_valselect.py --task ioi --model gpt2

# 5. rebuild the headline table
python scripts/analyze_valselection.py --c1-arm c1lcb --c0-arm c0lcb
```

Every arm records the configuration it actually ran with, and [`scripts/check_arm_config.py`](scripts/check_arm_config.py) asserts a finished arm matches its declared settings. That tool exists because a silent hyperparameter drift once invalidated a 760 A100-hour run — a setting that changes the result has to be readable off the artifact and checked mechanically, not trusted from the launcher.

## License

Apache 2.0.
