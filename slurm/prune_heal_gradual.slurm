#!/bin/bash
#SBATCH --job-name=ph_grad
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=0-06:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

# Gradual prune-and-heal on GPT-2 IOI. Env knobs (all optional):
#   CKPT SCHEDULE RERANK STEPS HEAL_LR HEAL_DELTA OUT
set -euo pipefail
PROJECT=${ILM_ROOT:-$PWD}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv
cd "$PROJECT"

CKPT="${CKPT:-checkpoints/ioi_head_c1_d20/lam_0.002}"
SCHEDULE="${SCHEDULE:-142,100,70,50,35,24,16,11,8}"
RERANK="${RERANK:-each}"
STEPS="${STEPS:-500}"
HEAL_LR="${HEAL_LR:-1e-4}"
# Default 0 (leash OFF): the validated production recipe (2026-07-02 reviewer-hardening
# runs, see capability-gate-confirmed memory) heals with NO delta leash at all and relies
# entirely on the capability gate (reject/retry) below. 10 was the OLD delta-leash value
# from before that pivot (see delta-leash-finding / ood-damage-is-stochastic) -- it's kept
# as an explicit override option for callers that intentionally sweep delta as their own
# variable (several historical phg_delta*.json runs did exactly that), but it must not be
# the silent default the grid launches on.
HEAL_DELTA="${HEAL_DELTA:-0}"
HEAL_ALPHA="${HEAL_ALPHA:-1.0}"
HEAL_CAP_TOL="${HEAL_CAP_TOL:-0.0}"
HEAL_DUAL_LR="${HEAL_DUAL_LR:-0.05}"
HEAL_ACC_FLOOR="${HEAL_ACC_FLOOR:-0.0}"
HEAL_ACC_EVERY="${HEAL_ACC_EVERY:-30}"
HEAL_DELTA_MAX="${HEAL_DELTA_MAX:-1000}"
HEAL_CIRC_FLOOR="${HEAL_CIRC_FLOOR:-0.0}"
RETRY_FLOOR="${RETRY_FLOOR:-0.0}"
RETRY_MAX="${RETRY_MAX:-0}"
RAMP_ARG=""
if [ -n "${DELTA_RAMP:-}" ]; then RAMP_ARG="--delta-ramp $DELTA_RAMP"; fi
PANEL_ARG=""
if [ -n "${MEASURE_PANEL:-}" ]; then PANEL_ARG="--measure-panel"; fi
NOPRUNE_ARG=""
if [ -n "${NO_PRUNE:-}" ]; then NOPRUNE_ARG="--no-prune"; fi
RANKMODE_ARG=""
if [ -n "${RANK_MODE:-}" ]; then RANKMODE_ARG="--rank-mode $RANK_MODE"; fi
EDIT_ARG=""
if [ -n "${EDIT_EVAL:-}" ]; then EDIT_ARG="--edit-eval"; fi
PROBE_ARG=""
if [ -n "${CAP_PROBE:-}" ]; then PROBE_ARG="--capability-probe"; fi
# Mixed-domain probe corpus (2026-07-02): default to the 320-prompt, 5-domain corpus
# (code/history/math/prose/templated) for every capability-probe feature, never the task's own
# eval set -- a single-domain probe reads "healthy" while other domains silently break (the old
# ioi_test-only probe once showed 1.02x while math sat at 328x on the same checkpoint). Override
# with PROBE_DATA if a run genuinely needs a different corpus.
PROBE_DATA="${PROBE_DATA:-data/probe_mixed.jsonl}"
PROBE_ARG="$PROBE_ARG --probe-data $PROBE_DATA"
GAMMA_ARG=""
if [ -n "${GAMMA_SWEEP:-}" ]; then GAMMA_ARG="--gamma-sweep $GAMMA_SWEEP"; fi
if [ -n "${SAVE_HEALED:-}" ]; then GAMMA_ARG="$GAMMA_ARG --save-healed $SAVE_HEALED"; fi
if [ -n "${PROBE_ROUND:-}" ]; then GAMMA_ARG="$GAMMA_ARG --probe-every-round"; fi
# Capability gate (2026-07-02): defaults ON for every ADAPTIVE run (Unknown A/B confirmed the
# un-gated controller can pass every accuracy check and still land 20x+ capability-damaged --
# the damage is a late jump, invisible to task accuracy, so the gate + mid-heal abort is now
# standard, not an experimental opt-in). Threshold: SELF-CALIBRATED by default (--cap-gate-auto
# -- bootstraps the noise floor on THIS model/task's own starting checkpoint, no prior run
# needed, works out of the box for a brand-new model/task in the grid). Set CAP_GATE to a
# number (e.g. 1.08, our IOI/docstring-Qwen-1.5B calibration) to pin a fixed value instead.
if [ -n "${ADAPTIVE:-}" ]; then
  CAP_ABORT="${CAP_ABORT:-2.0}"
  if [ -z "${CAP_GATE:-}" ]; then CAP_GATE_AUTO=1; fi
fi
if [ -n "${CAP_GATE:-}" ]; then GAMMA_ARG="$GAMMA_ARG --cap-gate $CAP_GATE --cap-gate-batches ${CAP_GATE_BATCHES:-50} --probe-max-length ${PROBE_MAXLEN:-256}"; fi
if [ -n "${CAP_GATE_AUTO:-}" ]; then GAMMA_ARG="$GAMMA_ARG --cap-gate-auto --cap-gate-batches ${CAP_GATE_BATCHES:-8} --cap-gate-boot ${CAP_GATE_BOOT:-30} --cap-gate-calib-rounds ${CAP_GATE_CALIB_ROUNDS:-1} --cap-gate-sigma-mult ${CAP_GATE_SIGMA_MULT:-3.0}"; fi
if [ -n "${CAP_ABORT:-}" ]; then GAMMA_ARG="$GAMMA_ARG --cap-abort $CAP_ABORT"; fi
ADAPT_ARG=""
if [ -n "${ADAPTIVE:-}" ]; then ADAPT_ARG="--adaptive --adaptive-tol ${ADAPT_TOL:-0.05} --adaptive-fullcap-tol ${ADAPT_FULLCAP_TOL:-0.05} --adaptive-cut ${ADAPT_CUT:-0.30} --adaptive-overshoot ${ADAPT_OVERSHOOT:-4} --adaptive-early-every ${ADAPT_EARLY:-40} --adaptive-reheal ${ADAPT_REHEAL:-3} --adaptive-heal-buffer ${ADAPT_BUFFER:-0} --adaptive-buffer-beta ${ADAPT_BETA:-0.6}"; fi
if [ -n "${ADAPT_PROP:-}" ]; then ADAPT_ARG="$ADAPT_ARG --adaptive-prop --adaptive-max-cut ${ADAPT_MAXCUT:-0.30} --adaptive-prop-gain ${ADAPT_GAIN:-0}"; fi
if [ -n "${ADAPTIVE:-}" ]; then ADAPT_ARG="$ADAPT_ARG --adaptive-cut-decay ${ADAPT_DECAY:-1.0} --seed ${SEED:-0}"; fi
# Conditional-mixed switch (2026-07-04): std heal by default, re-heal+latch mixed only when a round
# fails SOLELY on full_cap. Set ADAPT_MIXED_SWITCH=1 to enable (needs default --heal-mode standard).
if [ -n "${ADAPT_MIXED_SWITCH:-}" ]; then ADAPT_ARG="$ADAPT_ARG --adaptive-mixed-switch"; fi
# B1 control (2026-07-03): heal-only rounds at the full mask under --adaptive --no-prune.
if [ -n "${ADAPT_B1_ROUNDS:-}" ]; then ADAPT_ARG="$ADAPT_ARG --adaptive-b1-rounds $ADAPT_B1_ROUNDS"; fi
# ADAPTIVE_MIN: the controller's edge floor (default 4 in the script). Exposed so the
# floor-bound Gemma cells can be re-run without it to check the endpoint is not censored.
if [ -n "${ADAPTIVE_MIN:-}" ]; then ADAPT_ARG="$ADAPT_ARG --adaptive-min $ADAPTIVE_MIN"; fi
# ACCEPT_LCB: require the lower end of a one-sided confidence interval to clear the
# accuracy floor instead of the point estimate (fixes the winner's curse that made a
# third of endpoints miss the same tolerance on the sealed split). Needs a large
# EVAL_BATCHES to be usable -- see scripts/prepare_bigval.py.
if [ -n "${ACCEPT_LCB:-}" ]; then ADAPT_ARG="$ADAPT_ARG --accept-lcb $ACCEPT_LCB"; fi
# SAVE_ROUNDS: keep the last few accepted states so the trajectory can be re-scored
# offline (Pareto curves) without re-running the cell.
if [ -n "${SAVE_ROUNDS:-}" ]; then ADAPT_ARG="$ADAPT_ARG --save-rounds $SAVE_ROUNDS --save-rounds-keep ${SAVE_ROUNDS_KEEP:-6}"; fi
# C0.5 arm (2026-07-03): full adaptive controller, LoRA weights never move (heal() is a no-op).
FREEZE_ARG=""
if [ -n "${FREEZE_WEIGHTS:-}" ]; then FREEZE_ARG="--freeze-weights"; fi
# DIAGNOSTIC sweep (2026-07-03): NOT the real headline controller -- charts the full degradation
# curve (circuit_acc/full_cap/cap_ratio) past where --adaptive would normally stop. Mutually
# exclusive with ADAPTIVE -- set this instead, not alongside it.
DIAG_ARG=""
# W3 control (2026-07-11): fixed-schedule runs also need --seed (random rank-mode masks)
if [ -z "${ADAPTIVE:-}" ] && [ -n "${SEED:-}" ]; then DIAG_ARG="--seed $SEED"; fi
if [ -n "${DIAGNOSTIC_SWEEP:-}" ]; then DIAG_ARG="--diagnostic-sweep --diagnostic-sweep-cap-ratio ${DIAG_CAP_RATIO:-1.2} --diagnostic-acc-floor ${DIAG_ACC_FLOOR:-0.4}"; fi
START="${START:-trained}"
# Data layout (2026-07-03): data/<task_dir>/<model>/{train,eval,test}.jsonl. Set TASK (the
# --task flag value: ioi|gt|doc|induction|agreement) + MODEL_NAME (gpt2|llama|qwen) and the
# three paths below auto-derive -- no need to type full paths by hand for the standard
# layout. Set TRAIN_DATA/EVAL_DATA/TEST_DATA explicitly to override (e.g. to point at
# legacy/pre-migration data for reproducing an old checkpoint exactly).
case "${TASK:-ioi}" in
  doc) TASK_DIR=docstring ;;
  *)   TASK_DIR="${TASK:-ioi}" ;;
esac
MODEL_NAME="${MODEL_NAME:-gpt2}"
TRAIN_DATA="${TRAIN_DATA:-data/$TASK_DIR/$MODEL_NAME/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/$TASK_DIR/$MODEL_NAME/eval.jsonl}"
TEST_DATA="${TEST_DATA:-data/$TASK_DIR/$MODEL_NAME/test.jsonl}"
# MANDATORY sealed test split (2026-07-02): the controller's accept/reject decisions must
# never be graded on the same data as the final reported number. Greater-than is GPT-2-only
# (see CLAUDE.md) -- data/gt/gpt2/ is the only valid MODEL_NAME for TASK=gt.
if [ -z "${TEST_DATA:-}" ] || [ ! -f "$TEST_DATA" ]; then
  echo "ERROR: TEST_DATA ('$TEST_DATA') is required and must exist (sealed split, disjoint from TRAIN_DATA/EVAL_DATA)." >&2
  echo "  set TASK + MODEL_NAME to auto-derive from data/<task_dir>/<model>/test.jsonl, or set TEST_DATA explicitly." >&2
  exit 1
fi
TEST_ARG="--test-data $TEST_DATA"
EVAL_BATCHES="${EVAL_BATCHES:-25}"
RANK_BATCHES="${RANK_BATCHES:-8}"
LAZY_HEAL_ARG=""; if [ -n "${LAZY_HEAL:-}" ]; then LAZY_HEAL_ARG="--lazy-heal"; fi
HEAL_WD="${HEAL_WD:-0.01}"      # LoRA drift leash (decay toward the original model)
IG_STEPS="${IG_STEPS:-5}"       # EAP-IG integration points (ranking quality)
TASK_ARG=""
if [ -n "${TASK:-}" ]; then TASK_ARG="--task $TASK"; fi
OUT_ARG=""
if [ -n "${OUT:-}" ]; then OUT_ARG="--out $OUT"; fi
echo "ph-gradual: ckpt=$CKPT task=${TASK:-cfg} start=$START schedule=$SCHEDULE rerank=$RERANK steps/rd=$STEPS lr=$HEAL_LR delta=$HEAL_DELTA on $(hostname)"

python scripts/prune_heal_gradual.py \
  --checkpoint "$CKPT" $TASK_ARG --start "$START" \
  --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" $TEST_ARG \
  --schedule "$SCHEDULE" --rerank "$RERANK" \
  --heal-steps "$STEPS" --heal-lr "$HEAL_LR" --heal-delta "$HEAL_DELTA" --heal-alpha "$HEAL_ALPHA" \
  --heal-mode "${HEAL_MODE:-standard}" --mix-p "${MIX_P:-0.5}" \
  --heal-cap-tol "$HEAL_CAP_TOL" --heal-dual-lr "$HEAL_DUAL_LR" \
  --heal-acc-floor "$HEAL_ACC_FLOOR" --heal-acc-every "$HEAL_ACC_EVERY" \
  --heal-delta-max "$HEAL_DELTA_MAX" --heal-circ-floor "$HEAL_CIRC_FLOOR" \
  --heal-weight-decay "$HEAL_WD" --ig-steps "$IG_STEPS" $LAZY_HEAL_ARG --retry-floor "$RETRY_FLOOR" --retry-max "$RETRY_MAX" $RAMP_ARG $PANEL_ARG $NOPRUNE_ARG $RANKMODE_ARG $EDIT_ARG $PROBE_ARG $GAMMA_ARG $ADAPT_ARG $FREEZE_ARG $DIAG_ARG \
  --eval-batches "$EVAL_BATCHES" --rank-batches "$RANK_BATCHES" \
  $OUT_ARG
echo "Done."
