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
# E4 (verification) and E5 (legibility) re-run on the c1gate arm.
#
# The paper's payoff numbers were all computed on c1lazy -- an arm two generations old --
# because these scripts had that name typed into them. They are now arm-parameterised via
# C1_ARM, and checkpoints/headsave/c1gate/ was materialised from the regrid's saved rounds
# by scripts/materialize_headsave.py (96/96 cells). Nothing here retrains anything; every
# task reads a saved endpoint and ablates it.
#
# Output filenames carry the arm, so these cannot overwrite the c1lazy results the current
# draft still cites -- the two can be diffed before anything is swapped into the paper.
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
