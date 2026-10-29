#!/usr/bin/env bash
#SBATCH --job-name=arm-scale-score
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --exclude=g013,g021
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/arm-scale-score-%j.out
#
# Teacher agreement on the task-disjoint test set for one or more merged ARMs, one sglang server
# at a time on this GPU:  sbatch tillicum/arm_scaling_score.sbatch states-1 draws-1 ...
# The label "released" scores Piotr's released SelectionARM (trained on ~98% of all draws,
# including the test tasks, so it is an upper reference, not a held-out score).
set -euo pipefail
ROOT=/gpfs/scrubbed/piotrt/owrl/arm-scaling
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
trap 'kill -- -$SPID 2>/dev/null || true; rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd /gpfs/home/piotrt/OpenWebRL
for P in "$@"; do
  MODEL=$ROOT/runs/$P/merged
  [ "$P" = released ] && MODEL=/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89
  OUT=$ROOT/scores/$P.jsonl; mkdir -p $ROOT/scores
  [ -f ${OUT%.jsonl}.summary.json ] && { echo "[score] $P done"; continue; }
  setsid $ZRT/venv/bin/python -m sglang.launch_server --model-path $MODEL --host 127.0.0.1 --port 30000 \
    --dtype bfloat16 --tp 1 --mem-fraction-static 0.8 --context-length 16384 --served-model-name arm \
    > $ROOT/scores/$P.server.log 2>&1 &
  SPID=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:30000/health_generate >/dev/null; do
    kill -0 $SPID 2>/dev/null || { echo "FATAL server for $P died"; tail -20 $ROOT/scores/$P.server.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL server for $P not healthy"; exit 1; }
    sleep 5
  done
  $ZRT/arm-reproduction/venv/bin/python scripts/score_arm_sft.py --data $ROOT/data/sft-test --port 30000 --out $OUT
  kill -- -$SPID; wait $SPID 2>/dev/null || true
done
