#!/bin/bash
#SBATCH --job-name=b1base
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/b1base_%A_%a.out
#SBATCH --array=0-31%8
#
# "Does post-training WITHOUT the pruning loop already give a small circuit?" (R1 M3, R2 B1/B2)
#
# Two variants per cell, because the objective decides whether the question is even meaningful:
#
#   kl  heal on KL(original || masked) with no cut. NOTE: with the full mask the masked model IS
#       the original and a fresh LoRA is the identity, so the loss starts at 0 with ~0 gradient --
#       this arm is expected to be a no-op, and it is included as an explicit negative control
#       so that "healing alone concentrates nothing" is not confused with "healing never ran".
#       The old B1 runs used only this objective (their logged faith is 0.0 -> 0.0 throughout).
#
#   ce  heal on TASK cross-entropy with no cut (--heal-mode taskce). This is ordinary task
#       fine-tuning: the adapter has a real objective, genuinely changes the model, and improves
#       the task. This is the arm that actually tests whether ordinary post-training concentrates
#       a circuit, and it is the baseline both reviewers asked for.
#
# Distinct from results/headgrid/taskce/, which is task-CE healing WITH pruning (an ablation of
# the healing objective, not a no-prune baseline).
#
# Stage 2 for each variant: EAP-IG discovery over the full graph of that healed model, budgets
# scored on the 2000-example validation split and the sealed 1000-example test split.
#
# Seed 11 first; extend to 11/22/33 only for cells that land near the decision boundary, since
# seed variance on this grid is large.

set -euo pipefail
cd ${ILM_ROOT:-$PWD}

TASKS=(ioi agreement induction doc)
MODELS=(gpt2 llama llama3b gemma1b gemma4b qwen3-0.6b qwen3-1.7b qwen3-4b)
declare -A PREFIX=( [ioi]=ioi [agreement]=agr [induction]=ind [doc]=doc )
declare -A TDIR=( [ioi]=ioi [agreement]=agreement [induction]=induction [doc]=docstring )

I=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( I % 8 ))]}
TASK=${TASKS[$(( I / 8 ))]}
SEED=11
CKPT="checkpoints/${PREFIX[$TASK]}_${MODEL}_head_fresh"

for VARIANT in kl ce; do
  HEALED="checkpoints/b1base_${VARIANT}/seed${SEED}/${PREFIX[$TASK]}_${MODEL}"
  KNEE="results/headgrid/b1base_${VARIANT}/seed${SEED}/${TASK}_${MODEL}_knee.json"
  mkdir -p "$(dirname "$KNEE")" "$HEALED"

  MODE_ARG=""
  [ "$VARIANT" = "ce" ] && MODE_ARG="taskce"

  echo "=== B1[$VARIANT] stage 1 (heal only, no prune): $TASK/$MODEL seed $SEED ==="
  if [ ! -f "$HEALED/gates_healed.pt" ]; then
    # NO_PRUNE is required: --adaptive-b1-rounds is ignored without it, and the run would be an
    # ordinary condensation wearing a B1 label.
    env CKPT="$CKPT" TASK="$TASK" MODEL_NAME="$MODEL" SEED="$SEED" \
        ADAPTIVE=1 NO_PRUNE=1 ADAPT_B1_ROUNDS=10 CAP_GATE=1.10 CAP_GATE_AUTO=1 \
        ${MODE_ARG:+HEAL_MODE=$MODE_ARG} \
        EVAL_DATA="data/${TDIR[$TASK]}/${MODEL}/eval_big.jsonl" EVAL_BATCHES=250 \
        SAVE_HEALED="$HEALED" OUT="$KNEE" \
        bash slurm/prune_heal_gradual.slurm
  else
    echo "stage 1 already done, reusing $HEALED"
  fi

  echo "=== B1[$VARIANT] stage 2 (EAP-IG discovery on the healed model) ==="
  export PYTHONPATH=${ILM_ROOT:-$PWD}
  python3 scripts/run_discovery_b1.py --task "$TASK" --model "$MODEL" \
          --b1-ckpt "$HEALED" --seed "$SEED" --tag "$VARIANT"
done
