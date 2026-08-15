#!/bin/bash
#SBATCH --job-name=disc_valsel
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00
#SBATCH --array=0-31%12
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err
# Review remediation Job A (plan 2026-08-04-review-remediation.md): frozen discovery curves
# scored on BOTH the validation split (for budget selection) and the sealed test split (for the
# once-only report), one array task per task/model cell. Fixes fatal E2 (test-set selection) and
# independently verifies the 2026-07-30 test-curve regeneration (fatal E1).
set -e
PROJECT=${ILM_ROOT:-$PWD}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv
cd "$PROJECT"
TASKS=(ioi agreement induction doc)
MODELS=(gpt2 llama gemma1b qwen3-1.7b llama3b gemma4b qwen3-0.6b qwen3-4b)
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 8))]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 8))]}
python scripts/run_discovery_valselect.py --cells ${TASK}/${MODEL}
