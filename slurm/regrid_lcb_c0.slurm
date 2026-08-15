#!/bin/bash
#SBATCH --job-name=regridc0
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/regridc0_%A_%a.out
#SBATCH --array=0-95%8
#
# C0 re-run under the identical selection fix as regrid_lcb.slurm (2026-08-06).
#
# Required, not optional: C0 is the ablation that isolates reshaping from search, so it must
# be selected by the same rule as C1 or the comparison stops being apples-to-apples. It is
# also the arm that needs it most -- under the old point-estimate rule only 43 of 96 C0
# endpoints re-cleared the tolerance on the sealed split, the worst of any arm, which is the
# expected signature of an adaptive controller over-fitting a 200-example split.
#
# Identical to the C1 arm in every respect except FREEZE_WEIGHTS=1: same fresh checkpoint,
# same ranking, same cut schedule, same capability gate, same 2000-example validation split,
# same one-sided 95% accept bound, same edge floor, same seeds.
#
# Writes results/headgrid/c0lcb/seed<S>/<task>_<model>_knee.json

set -euo pipefail
cd ${ILM_ROOT:-$PWD}

TASKS=(ioi agreement induction doc)
MODELS=(gpt2 llama llama3b gemma1b gemma4b qwen3-0.6b qwen3-1.7b qwen3-4b)
SEEDS=(11 22 33)
declare -A PREFIX=( [ioi]=ioi [agreement]=agr [induction]=ind [doc]=doc )
declare -A TDIR=( [ioi]=ioi [agreement]=agreement [induction]=induction [doc]=docstring )

I=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( I % 3 ))]}
REST=$(( I / 3 ))
MODEL=${MODELS[$(( REST % 8 ))]}
TASK=${TASKS[$(( REST / 8 ))]}

CKPT="checkpoints/${PREFIX[$TASK]}_${MODEL}_head_fresh"
OUT="results/headgrid/c0lcb/seed${SEED}/${TASK}_${MODEL}_knee.json"
mkdir -p "$(dirname "$OUT")"

echo "=== regrid-C0 task $I: $TASK/$MODEL seed $SEED -> $OUT ==="
export CKPT TASK MODEL_NAME="$MODEL" SEED
export ADAPTIVE=1 CAP_GATE=1.10 CAP_GATE_AUTO=1 MEASURE_PANEL=1
export ADAPTIVE_MIN=1 ACCEPT_LCB=1.645
export EVAL_DATA="data/${TDIR[$TASK]}/${MODEL}/eval_big.jsonl" EVAL_BATCHES=250
export FREEZE_WEIGHTS=1
export OUT

bash slurm/prune_heal_gradual.slurm
