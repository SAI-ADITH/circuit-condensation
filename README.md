# Circuit Condensation

**Post-training that concentrates a behavior's causal circuit.**

Paper: [arXiv:2608.27254](https://arxiv.org/abs/2608.27254) · Interactive demo: [sai-adith.github.io/circuit-condensation](https://sai-adith.github.io/circuit-condensation/)

Circuit discovery usually searches a frozen model for the smallest subgraph that carries a behavior. Circuit Condensation instead post-trains the model so that a much smaller subgraph suffices. The circuits it returns are small enough to verify exhaustively.

## Results

Four behaviors (indirect object identification, subject–verb agreement, induction, docstring completion), eight models (GPT-2 small, Llama-3.2-1B/3B-Instruct, Gemma-3-1B/4B-it, Qwen3-0.6B/1.7B/4B), three seeds each: 32 task–model cells, 96 runs, attention-head granularity.

Every arm uses the same selection rule: the smallest circuit whose validation accuracy is within 0.05 of that arm's own full graph. A separate 1,000-example test split is read once, after selection.

| Frozen arm vs. condensed circuit (C1) | Geometric-mean size ratio | Cells where C1 is smaller |
| --- | ---: | ---: |
| Random ranking | 103× | 32/32 |
| EAP | 18.4× | 31/32 |
| EAP-GP | 10.6× | 30/32 |
| **EAP-IG** (strongest frozen baseline) | **8.1×** | **30/32** |
| **C0** (same controller, weights frozen) | **7.4×** | **29/32** |

The C0 row isolates the effect of post-training: it runs the identical search without letting the weights move. 91 of 96 condensed endpoints also meet the tolerance on the sealed test split.

Smaller circuits make exhaustive checks feasible. For the 19 circuits small enough to enumerate, every subset is tested (5,952 masked circuits): 11 are exactly minimal. All-pairs ablations (169,476 in total) show that no circuit's edges act independently; the matched frozen circuits would need a median 55× more pair ablations.

## Method

The model is treated as a graph of residual-stream writers and readers. A closed edge passes the activation from a corrupted counterfactual prompt rather than zero (interchange ablation). Starting from the full graph and a fresh LoRA adapter, each round:

1. **Ranks** the active edges with EAP-IG.
2. **Prunes** the lowest-scoring 30% of active edges. A rejected proposal restores the last accepted state and halves the cut, down to 5%.
3. **Heals** the LoRA adapter (base weights frozen) by distilling the original model's output distribution through the reduced graph.
4. **Accepts** the round only if task accuracy stays within 0.05 of the full graph on held-out validation data *and* perplexity on a held-out mixed-domain probe stays within 1.05× of the original model.

The procedure stops when no smaller proposal can be accepted. C1 is the smallest accepted circuit; C0 is the same controller with the weights frozen.

## Installation

```bash
git clone https://github.com/SAI-ADITH/circuit-condensation
cd circuit-condensation
pip install -e .
```

Model checkpoints are read from `models/<name>/` (Hugging Face format). All paths are relative to the repository root; set `ILM_ROOT` to run from elsewhere.

## Reproducing the paper

Model names used throughout: `gpt2 llama llama3b gemma1b gemma4b qwen3-0.6b qwen3-1.7b qwen3-4b`, with each checkpoint in `models/<model>/`. Tasks: `ioi agreement induction doc`. The `slurm/` launchers assume one A100 per job; the grid launchers loop over the 4 tasks × 8 models × 3 seeds by array index.

**1. Data.** The generators default to small splits, so pass the paper's sizes explicitly:

```bash
python scripts/prepare_ioi.py       --model-path models/<model> --out-dir data/ioi/<model>       --n-train 2000 --n-test 1000
python scripts/prepare_agreement.py --model-path models/<model> --out-dir data/agreement/<model> --n-train 2000 --n-test 1000
python scripts/prepare_docstring.py --model-path models/<model> --out-dir data/docstring/<model> --n-train 2000 --n-test 1000
python scripts/prepare_induction.py --model      models/<model> --out-dir data/induction/<model> --n-train 2000 --n-test 1000 --repeats 4
python scripts/prepare_bigval.py    # 2,000-example validation split, eval_big.jsonl, for every task and model
```

**2. Fresh checkpoint** (untrained LoRA plus edge gates, all open), once per task and model. `<prefix>` is `ioi`, `agr`, `ind` or `doc`:

```bash
sbatch --export=ALL,OUT=checkpoints/<prefix>_<model>_head_fresh,MODEL=models/<model>,TASK=<task>,GRAN=head \
       slurm/build_fresh_checkpoint.slurm
```

**3–7. The grid and the analyses.**

| Step | Command | Writes |
| --- | --- | --- |
| 3. Condensation, C1 | `sbatch slurm/regrid_gate.slurm` | `results/headgrid/c1gate/` |
| 4. Frozen-weight arm, C0 | `sbatch slurm/regrid_lcb_c0.slurm` | `results/headgrid/c0lcb/` |
| 5. Frozen discovery baselines | `sbatch slurm/discovery_valselect.slurm` | EAP, EAP-IG, EAP-GP and random curves |
| 6. Headline table | `python scripts/analyze_valselection.py` | size ratios, cell wins, sign tests, cluster-bootstrap CIs |
| 7. Verification | `sbatch slurm/payoff_c1gate.slurm` and `sbatch slurm/payoff_verify.slurm` | single-edge necessity, exhaustive subset minimality, all-pairs interactions |

`analyze_valselection.py` defaults to the arms the paper reports (`c1gate`, `c0lcb`); override with `--c1-arm` / `--c0-arm`. Before step 7, `scripts/materialize_headsave.py` converts the controller's saved rounds into the checkpoint layout the verification scripts read. `slurm/payoff_v_cells.txt` lists the verification jobs, and `slurm/payoff_v_skip.txt` the pairwise jobs skipped for cost.

## Repository layout

```
ilm/
  data/         clean/corrupted prompt pairs for each task
  masking/      head-level edge graph and gated residual stream
  discovery/    frozen baselines: EAP, EAP-IG, EAP-GP, ACDC, random
  eval/         restricted-choice accuracy and attribution measures
  training/     LoRA setup and distillation loss
scripts/        data preparation, the condensation controller (prune_heal_gradual.py),
                frozen discovery, headline analysis, verification
slurm/          launchers for the experiment grid
```

## Citation

```bibtex
@article{senthilkumar2026circuit,
  title   = {Circuit Condensation: Post-Training that Concentrates a Behavior's Causal Circuit},
  author  = {Senthil Kumar, Sai Adith},
  journal = {arXiv preprint arXiv:2608.27254},
  year    = {2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
