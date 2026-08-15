#!/bin/bash
#SBATCH --job-name=xcorrupt
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/xcorrupt_%A_%a.out
#SBATCH --array=0-7%4
#
# R1's fatal flaw (F1): every number in the paper is measured under the interchange
# convention the method trains against. This scores the SAME retained circuits under
# corruption families they never saw (different template, resampled counterfactual,
# off-task text) with the clean prompt and answer held fixed.
#
# Reads head-level C1 checkpoints from checkpoints/headsave/c1lazy/seed<S>/.

set -euo pipefail
cd ${ILM_ROOT:-$PWD}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
source ~/.bashrc
conda activate venv

CELLS=(
  "ioi gpt2"        "ioi llama"       "ioi gemma1b"     "ioi qwen3-1.7b"
  "agreement llama" "induction gemma1b" "doc llama"     "induction qwen3-0.6b"
)
read -r TASK MODEL <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
SEED=${SEED:-11}

export GRAN=head
export COND_CKPT_ROOT="checkpoints/headsave/c1lazy/seed${SEED}"
export PYTHONPATH=${ILM_ROOT:-$PWD}

echo "=== cross-corruption: $TASK/$MODEL seed $SEED ==="
python3 scripts/run_cross_corruption.py --task "$TASK" --model "$MODEL" --seed "$SEED"
