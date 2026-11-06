#!/usr/bin/env bash
#SBATCH --job-name=relabel-luna
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# API calls only (gpt-6-luna); a MIG slice bills 1/7 of an H200 and CPU-only jobs queue for days.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/relabel-luna-%j.out
#
# Re-pick the preferred candidate at every state of a collection dataset with gpt-6-luna (same pinned
# ARM prompt and request as the Luna selector evals), for prepare_arm_joint_data.py --relabel.
#   DATASET=/gpfs/scrubbed/piotrt/owrl/c2-collect/round2-b0.3s1/training.jsonl OUT=.../luna-labels.jsonl sbatch tillicum/relabel_luna.sbatch
set -euo pipefail
REPO=/gpfs/home/piotrt/OpenWebRL; PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
DATASET=${DATASET:?}; OUT=${OUT:?}
TASKS=${TASKS:-/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/tasks.jsonl}
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
set -a; source $REPO/.env; set +a
cd $REPO
PORT=$((27000 + SLURM_JOB_ID % 1000))
setsid $PY scripts/serve_sol_selector.py --model gpt-6-luna ${PROBS:+--probabilities} --reasoning-effort ${LUNA_EFFORT:-high} --max-output-tokens 8192 \
  --max-cost-usd ${LUNA_MAX_USD:-40} --max-requests 30000 --concurrency 16 --port $PORT --output $(dirname $OUT)/luna-api \
  > $(dirname $OUT)/luna-server.log 2>&1 &
SPID=$!; trap 'kill -- -$SPID 2>/dev/null || true; rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do kill -0 $SPID || { tail -20 $(dirname $OUT)/luna-server.log; exit 1; }; sleep 3; done
$PY scripts/relabel_with_selector.py --dataset $DATASET --tasks $TASKS --port $PORT --out $OUT --parallel 16
cat $(dirname $OUT)/luna-api/usage.json; echo
