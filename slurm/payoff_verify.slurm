#!/bin/bash
#SBATCH --job-name=payverify
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00
#SBATCH --output=outputs/%x-%A_%a.out
#
# The two checks that make the "why smaller circuits matter" argument CATEGORICAL rather than
# incremental, re-run on the c1gate arm:
#
#   subset   -- enumerate ALL 2^k edge subsets. This is Wang et al. 2023's own completeness
#               test, which that paper calls computationally intractable and replaces with three
#               sampling heuristics. Condensation puts k at 3-14, so the exact test runs; the
#               matched frozen circuit would need 2^416.
#   pairwise -- every (i,j) knockout against the sum of its parts. O(k^2): minutes here, and
#               35-13,300x more forwards on the matched frozen circuit.
#
# The cell list is derived from the finished grid (slurm/payoff_v_cells.txt) rather than
# hardcoded, so it can never drift from the endpoints it is meant to verify.
set -euo pipefail
PROJECT=${ILM_ROOT:-$PWD}
cd "$PROJECT"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv

export C1_ARM=c1gate GRAN=head FRESH_SUFFIX=_head_fresh

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/payoff_v_cells.txt)
read -r KIND TASK MODEL SEED <<< "$LINE"
[ -n "${KIND:-}" ] || { echo "no cell at index $SLURM_ARRAY_TASK_ID"; exit 1; }
export COND_CKPT_ROOT="checkpoints/headsave/${C1_ARM}/seed${SEED}"

# Cost cutoff (plans/2026-08-20-pairwise-cost-cutoff.md). Exhaustive pairwise is O(k^2), and a
# handful of cells project into the hundreds of GPU-hours -- those are the cells where
# condensation left a LARGE circuit, so they are reported by closed-form cost in
# results/pairwise_cost_table.json rather than executed. The skip list is keyed by
# task/model/seed, never by array index: this file is dispatched with `sed -n "$((ID+1))p"`, so
# renumbering the cell list would silently repoint any in-flight array task at a different cell.
if [ -f slurm/payoff_v_skip.txt ] && grep -qxF "$KIND $TASK $MODEL $SEED" slurm/payoff_v_skip.txt; then
  echo "SKIP over cost cutoff: $KIND $TASK $MODEL $SEED (see results/pairwise_cost_table.json)"
  exit 0
fi


echo "=== $KIND : $TASK/$MODEL seed $SEED ==="
case "$KIND" in
  subset)   python3 scripts/run_subset_completeness.py   --task "$TASK" --model "$MODEL" --seed "$SEED" ;;
  pairwise) python3 scripts/run_pairwise_completeness.py --task "$TASK" --model "$MODEL" --seed "$SEED" ;;
  *) echo "unknown kind $KIND"; exit 1 ;;
esac
