#!/bin/bash
#SBATCH --job-name=build_fresh
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=0-00:15:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

# Build a fresh (never-trained) C1 starting checkpoint -- no Stage 1 at all (see
# scripts/build_fresh_checkpoint.py's docstring for why). Feed the
# result straight into slurm/prune_heal_gradual.slurm (--adaptive) to get the real C1
# knee curve and smallest-faithful circuit. Required env: OUT MODEL TASK. Optional:
# GRAN (default block; the paper uses head) MAX_LENGTH (default 64).
#   sbatch --export=ALL,OUT=checkpoints/ioi_fresh,MODEL=models/gpt2,TASK=ioi,GRAN=head,MAX_LENGTH=64 \
#     slurm/build_fresh_checkpoint.slurm
set -euo pipefail
PROJECT=${ILM_ROOT:-$PWD}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv
cd "$PROJECT"

: "${OUT:?set OUT (output checkpoint dir)}"; : "${MODEL:?set MODEL}"; : "${TASK:?set TASK}"
GRAN="${GRAN:-block}"
MAX_LENGTH="${MAX_LENGTH:-64}"

echo "build_fresh_checkpoint: model=$MODEL task=$TASK granularity=$GRAN max_length=$MAX_LENGTH -> $OUT on $(hostname)"
python scripts/build_fresh_checkpoint.py --out-dir "$OUT" --model-path "$MODEL" --task "$TASK" \
  --graph-granularity "$GRAN" --max-length "$MAX_LENGTH"
echo "Done."
