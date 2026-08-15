# Circuit Condensation

**Post-training that makes a model's mechanism small enough to audit.**

Mechanistic interpretability usually starts with a frozen model and asks: *what is the smallest subgraph — or circuit — responsible for a particular behavior?* Most methods then try to search harder for that circuit.

This project takes the opposite approach: **instead of finding a small circuit, make one.**

We run a short post-training procedure that reshapes an already-capable model so that the causal computation for a target behavior becomes concentrated in a much smaller subgraph — small enough that we can actually verify it exhaustively.

Across a 96-run grid (4 behaviors × 8 models × 3 seeds), the resulting condensed circuits are a **median 6.8× smaller** than circuits recovered by frozen-model discovery on the same model using the same selection rule. **93/96** condensed endpoints also clear a sealed test-set threshold that is evaluated exactly once.

---

## The method

We attach LoRA adapters and a learnable on/off gate to every edge in the model's computation graph. A controller then repeatedly does three things:

1. **Rank.** Score the surviving edges using attribution with EAP-IG — integrated gradients along the corrupted→clean path.

2. **Cut.** Remove a margin-proportional fraction of the lowest-ranked edges. Importantly, turning an edge "off" does not zero its activation. Instead, that edge reads its activation from a corrupted counterfactual input, which helps keep the model in-distribution.

3. **Heal.** Update the LoRA weights so the model can recover the target behavior using the remaining edges. Every healing round is subject to an accept/reject **capability gate**, implemented as a bootstrap-calibrated perplexity ratio on a held-out, off-target corpus. If a round damages general capability, it is rejected and rolled back.

The process stops at the smallest circuit that still passes on held-out validation data.

One important detail is that concentration is **measured, not optimized**. The training objective only uses task accuracy and the capability probe. The concentration metric itself is never part of the training signal.

Full controller: [`scripts/prune_heal_gradual.py`](scripts/prune_heal_gradual.py). Graph and gating implementation: [`ilm/masking/`](ilm/masking/).

---

## Results

The main grid covers **4 behaviors** — indirect object identification, subject–verb agreement, induction, and docstring completion — across **8 models**: GPT-2 124M, Llama-3.2 1B/3B, Gemma-3 1B/4B, and Qwen3 0.6B/1.7B/4B. Each setting is run with **3 seeds**, using attention-head granularity.

Every arm uses the exact same selection rule:

> Choose the smallest circuit whose accuracy on a 2,000-example validation split is within ε=0.05 of that arm's *own* full-graph accuracy.

Only after that selection is complete do we evaluate once on the separate 1,000-example test split.

| Comparison                                   | Median size ratio | Cells where condensed is smaller |   Sign test |
| -------------------------------------------- | ----------------: | -------------------------------: | ----------: |
| vs. **EAP-IG** (frozen discovery)            |          **6.8×** |                            32/32 | p = 2.6e-06 |
| vs. **C0** (same controller, weights frozen) |          **6.7×** |                            32/32 | p = 2.6e-06 |
| vs. **Random ranking**                       |             69.3× |                            32/32 | p = 4.7e-10 |

The **C0** comparison is the most important one. It uses the same gates, the same pruning loop, and the same healing schedule, but keeps the model weights frozen. The only meaningful difference is whether the model itself is allowed to change. That lets us isolate the improvement from post-training rather than attributing it to a better search procedure.

### What do the smaller circuits actually buy us?

The main benefit is **exhaustive verification**.

Once a circuit contains only a few dozen edges, it becomes feasible to enumerate every subset and every pair of edges. Across 62 circuits, we ran **169k pairwise edge ablations**.

Running the same exhaustive check on the corresponding frozen-model circuits would require **35–13,300× more ablations**.

So the point is not simply that the condensed circuits score better on some interpretability metric. They make a form of verification computationally possible that is effectively infeasible for the frozen circuits at the same budget.

### What does it cost?

The resulting circuit is still defined relative to the counterfactual corruption used to construct it.

When we evaluate against corruption families that were never used during training, the condensed circuits retain scores of **0.64–0.66**, compared with **0.45–0.48** for frozen circuits. But this is still substantially below the **0.96** performance they achieve on the corruption family they were trained with.

So the concentration is not universal. It is distribution-specific, and the repository reports that limitation directly rather than hiding it.

---

## Layout

```text
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

Every arm records the configuration it actually ran with, and [`scripts/check_arm_config.py`](scripts/check_arm_config.py) checks that a completed arm matches its declared settings.

That check exists for a practical reason: a silent hyperparameter drift once invalidated a **760 A100-hour run**. Since then, any setting that can change the result has to be visible in the artifact and mechanically verified rather than trusted from the launcher.

## License

Apache 2.0.
