#!/bin/bash
#SBATCH --job-name=ilm_ftcontrast
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=0-06:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

# The Zhang same-axis contrast (plan 2026-06-26, steps E0/E1): fine-tune the model the
# ORDINARY way (SFT or RL, no gates/sparsity) and read the EAP concentration panel
# before vs after. Gates are pinned OPEN so the panel measures the FULL model's circuit
# (weights-only axis) -- exactly Zhang's question. PASS E0: SFT does NOT concentrate.
# PASS E1: RL diffuses (entropy/PR up) while our C1 concentrates (down).
#
# E0 kill-test (cheap, IOI/GPT-2, SFT):
#   sbatch slurm/task_finetune.slurm
# E1 RL (after E0 passes):  MODE=rl sbatch slurm/task_finetune.slurm
# GSM8K/Qwen:  TASK=gsm8k MODEL=.../Qwen2.5-1.5B-Instruct DATA=.../gsm8k_train.jsonl \
#              EVAL_DATA=.../gsm8k_eval.jsonl GRAN=block MAX_LENGTH=1024 sbatch ...

set -euo pipefail

PROJECT=${ILM_ROOT:-$PWD}
mkdir -p "$PROJECT/outputs" "$PROJECT/checkpoints"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

source ~/.bashrc
conda activate venv
cd "$PROJECT"
if [ -f "$PROJECT/.env" ]; then set -a; source "$PROJECT/.env"; set +a; fi

# Defaults = E0 kill-test: SFT on IOI (GPT-2, head granularity).
MODE="${MODE:-sft}"
TASK="${TASK:-ioi}"
MODEL="${MODEL:-$PROJECT/models/gpt2}"
# Data layout (2026-07-03): data/<task_dir>/<model>/{train,eval,test}.jsonl. Set MODEL_NAME
# (gpt2|llama|qwen) to auto-derive DATA/EVAL_DATA; set them explicitly to override (e.g. for
# legacy/pre-migration data or GSM8K, which stays a flat data/gsm8k_*.jsonl pair).
case "${TASK:-ioi}" in
  doc) TASK_DIR=docstring ;;
  *)   TASK_DIR="${TASK:-ioi}" ;;
esac
MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA="${DATA:-$PROJECT/data/$TASK_DIR/$MODEL_NAME/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$PROJECT/data/$TASK_DIR/$MODEL_NAME/eval.jsonl}"
GRAN="${GRAN:-head}"
MAX_LENGTH="${MAX_LENGTH:-64}"
NUM_STEPS="${NUM_STEPS:-1000}"
LR="${LR:-1e-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"
BETA="${BETA:-0.04}"
RUN="${RUN:-ft_${TASK}_${MODE}}"

echo "contrast: mode=$MODE task=$TASK gran=$GRAN model=$MODEL batch=$BATCH_SIZE group=$GROUP_SIZE -> $PROJECT/checkpoints/$RUN on $(hostname)"
nvidia-smi || true

python scripts/task_finetune.py \
    --mode "$MODE" \
    --task "$TASK" \
    --graph-granularity "$GRAN" \
    --model-path "$MODEL" \
    --data "$DATA" \
    --eval-data "$EVAL_DATA" \
    --output-dir "$PROJECT/checkpoints/$RUN" \
    --num-steps "$NUM_STEPS" \
    --lr "$LR" \
    --max-length "$MAX_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --beta "$BETA"

echo "Done $RUN."
