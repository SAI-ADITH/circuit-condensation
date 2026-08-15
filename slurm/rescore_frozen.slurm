#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-16:00:00
#SBATCH --output=outputs/rescore_%A_%a.out
#SBATCH --array=0-31%8
#
# Put the frozen arms on the same validation split as the controllers, so ONE selection rule
# can be applied to all four arms. Without this, C1/C0 are held to a confidence bound on 2000
# examples while EAP/EAP-IG/Random are held to a raw point estimate on 200 -- two different
# rules, and the strict one is ours.
#
# Only validation is re-measured. The budget grid is unchanged and the rankings are reproduced
# with the same seeding, so the circuit at each budget is the same object and its stored
# sealed-test accuracy is still valid; the re-scoring can only change WHICH budget is selected.
# Test is not re-read. ACDC is skipped (dropped from the paper).

set -euo pipefail
cd ${ILM_ROOT:-$PWD}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH=${ILM_ROOT:-$PWD}
source ~/.bashrc
conda activate venv

TASKS=(ioi agreement induction doc)
MODELS=(gpt2 llama llama3b gemma1b gemma4b qwen3-0.6b qwen3-1.7b qwen3-4b)
I=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( I % 8 ))]}
TASK=${TASKS[$(( I / 8 ))]}

echo "=== rescore frozen on 2000-example val: $TASK/$MODEL ==="
python3 scripts/rescore_frozen_bigval.py --task "$TASK" --model "$MODEL"
