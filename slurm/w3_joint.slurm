#!/bin/bash
#SBATCH --job-name=w3joint
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=0-10:00:00
#SBATCH --array=0-63%6
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err
#
# W3: the joint gates+LoRA baseline (R1 M4 / R2 B6) -- plans/2026-08-14-w3-joint-gates-lora.md.
# Identical to the failed learned-gate arm (app:lgate) EXCEPT the weights are unfrozen: no
# --freeze-weights, so scripts/train.py puts the LoRA params in the optimizer and activates the
# delta*KL(orig||full) capability leash. One flag of difference = clean attribution.
#
# 64 tasks = 4 cells x 2 deltas x 8 lambdas. Endpoints are scored separately by
# scripts/score_w3_endpoints.py on the common ruler (eval_big val + sealed test, binarized gates).

set -euo pipefail
PROJECT=${ILM_ROOT:-$PWD}
cd "$PROJECT"
mkdir -p outputs checkpoints/w3joint
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
source ~/.bashrc
conda activate venv

# ---- index -> (cell, delta, lambda) --------------------------------------------------------
CELLS=(ioi:gpt2 agreement:llama induction:gemma1b doc:llama)   # the four app:lgate cells
DELTAS=(1.0 10.0)
LAMBDAS=(0.002 0.005 0.02 0.05 0.1 0.3 0.8 2.0)                # same ladder as app:lgate

I=$SLURM_ARRAY_TASK_ID
CELL=${CELLS[$(( I / 16 ))]}
DELTA=${DELTAS[$(( (I % 16) / 8 ))]}
LAMBDA=${LAMBDAS[$(( I % 8 ))]}
TASK=${CELL%%:*}
MODEL_NAME=${CELL##*:}

case "$MODEL_NAME" in
  gpt2)    MODEL=models/gpt2 ;;
  llama)   MODEL=models/Llama-3.2-1B-Instruct ;;
  gemma1b) MODEL=models/gemma-3-1b-it ;;
  *) echo "unknown model $MODEL_NAME"; exit 1 ;;
esac
case "$TASK" in
  doc) TASK_DIR=docstring; MAXLEN=96 ;;
  *)   TASK_DIR=$TASK;     MAXLEN=64 ;;
esac

NUM_STEPS=3000
CKPT="checkpoints/w3joint/${TASK}_${MODEL_NAME}/d${DELTA}_lam${LAMBDA}"
echo "W3 joint: $TASK/$MODEL_NAME delta=$DELTA lambda=$LAMBDA -> $CKPT"

# ---- guard 1: never retrain a finished run -------------------------------------------------
# The trainer unlinks its own logs at startup, so WITHOUT this guard an array resubmission
# would destroy the very run it duplicates (the rescore arrays made exactly this mistake).
if python3 - "$CKPT" "$NUM_STEPS" <<'EOF'
import json, sys
from pathlib import Path
ck, steps = Path(sys.argv[1]), int(sys.argv[2])
try:
    last = [json.loads(l) for l in (ck / "eval_log.jsonl").read_text().splitlines() if l.strip()][-1]
    done = (ck / "gates.pt").exists() and (ck / "config.json").exists() and last["step"] >= steps
except Exception:
    done = False
sys.exit(0 if done else 1)
EOF
then
    echo "already complete, skipping training (delete $CKPT to force a re-run)"
else
    python3 scripts/train.py \
        --task "$TASK" \
        --model-path "$PROJECT/$MODEL" \
        --data "data/$TASK_DIR/$MODEL_NAME/train.jsonl" \
        --eval-data "data/$TASK_DIR/$MODEL_NAME/eval.jsonl" \
        --max-length "$MAXLEN" \
        --output-dir "$CKPT" \
        --num-steps "$NUM_STEPS" \
        --lambda-sparse "$LAMBDA" \
        --delta-preserve "$DELTA" \
        --graph-granularity head \
        --eval-every 300 \
        --eval-batches 25 \
        --seed 0
        # NO --freeze-weights: that one absent flag IS the experiment.
fi

# ---- guard 2: the setting that defines the arm is read off the artifact, not assumed -------
# The c1lcb arm burned 760 A100-h because a launcher said one thing and the run did another.
python3 - "$CKPT" "$LAMBDA" "$DELTA" "$NUM_STEPS" <<'EOF'
import json, sys
from pathlib import Path
ck = Path(sys.argv[1]); lam, dlt, steps = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cfg = json.loads((ck / "config.json").read_text())
checks = {
    "freeze_weights": (cfg["freeze_weights"], False),
    "graph_granularity": (cfg["graph_granularity"], "head"),
    "lambda_sparse": (cfg["loss"]["lambda_sparse"], lam),
    "delta_preserve": (cfg["loss"]["delta_preserve"], dlt),
    "num_steps": (cfg["num_steps"], steps),
    "lambda_mode": (cfg["lambda_mode"], "fixed"),
    "delta_mode": (cfg["delta_mode"], "fixed"),
    "seed": (cfg["seed"], 0),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
if bad:
    print(f"CONFIG ASSERTION FAILED for {ck}:")
    for k, (got, want) in bad.items():
        print(f"  {k}: ran with {got!r}, intended {want!r}")
    sys.exit(1)
print(f"config verified: {ck} (freeze=False, head, lam={lam}, delta={dlt})")
EOF

echo "Done: $TASK/$MODEL_NAME delta=$DELTA lambda=$LAMBDA"
