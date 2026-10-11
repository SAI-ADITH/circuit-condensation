#!/bin/bash
#SBATCH --job-name=payoff
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --array=0-14
#
# Verification and legibility analyses on the c1gate arm (the paper's C1).
#
# The scripts are arm-parameterised via C1_ARM. checkpoints/headsave/c1gate/ is materialised
# from the grid's saved rounds by scripts/materialize_headsave.py. Nothing here retrains
# anything; every task reads a saved endpoint and ablates it. Output filenames carry the arm.
set -euo pipefail
PROJECT=${ILM_ROOT:-$PWD}
cd "$PROJECT"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv

export C1_ARM=c1gate
export GRAN=head
export FRESH_SUFFIX=_head_fresh

I=$SLURM_ARRAY_TASK_ID
SEEDS=(11 22 33)
TASKS=(ioi agreement induction doc)

# 0-2    necessity, one per seed          (E4: is the whole circuit load-bearing)
# 3-14   minimality, 4 tasks x 3 seeds    (E4: exhaustive subset enumeration)
if [ "$I" -lt 3 ]; then
  SEED=${SEEDS[$I]}
  export COND_CKPT_ROOT="checkpoints/headsave/${C1_ARM}/seed${SEED}"
  export OUT_TAG="_${C1_ARM}_s${SEED}"
  echo "=== necessity: seed $SEED -> results/transfer/necessity_all${OUT_TAG}.json ==="
  python3 scripts/run_necessity_all.py
else
  J=$(( I - 3 ))
  SEED=${SEEDS[$(( J / 4 ))]}
  TASK=${TASKS[$(( J % 4 ))]}
  export COND_CKPT_ROOT="checkpoints/headsave/${C1_ARM}/seed${SEED}"
  export OUT_TAG="_${C1_ARM}_s${SEED}"
  echo "=== minimality: $TASK seed $SEED -> results/transfer/minimality_all_${TASK}${OUT_TAG}.json ==="
  python3 scripts/run_minimality_all.py --task "$TASK"
fi
