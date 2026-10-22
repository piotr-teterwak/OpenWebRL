#!/usr/bin/env bash
#SBATCH --job-name=owrl-probe
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/probe-%j.out
#
# Offline round-2 probe (scripts/probe_round2.py) on one H200: the selection ARM
# stays up while each actor is served in turn. No browsers, no judge calls.
#   ACTORS="base 1a joint-sft" sbatch tillicum/probe_round2.sbatch
# Labels other than base resolve to merged models from the eval
# (/gpfs/scrubbed/piotrt/owrl/eval/merged/<label>); label=/path/to/model names any other.
# SKIP_REPLAY=1 skips the replay check (validated at 99.5% agreement in job 374159).
set -euo pipefail
ACTORS="${ACTORS:-base 1a joint-sft}"
STATES="${STATES:-400}"
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python
SERVER_PY=$ZRT/venv/bin/python
C2=$ZRT/arm-reproduction/runs/c2-full-282782-20260908T075414Z
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
ARM_MODEL=/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89
OUT=/gpfs/scrubbed/piotrt/owrl/probe/$SLURM_JOB_ID

export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
cd $REPO; mkdir -p $OUT
APORT=$((18200 + SLURM_JOB_ID % 500)); SPORT=$((APORT + 1))
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || kill $p 2>/dev/null || true; done; rm -rf "$OPENWEBRL_JOB_TMP"; }
trap cleanup EXIT
wait_ok() { local url=$1 pid=$2 t0=$(date +%s)
  until curl -sf $url >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "FATAL server for $url died"; tail -20 $OUT/*.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL $url not healthy after 15 min"; exit 1; }
    sleep 5
  done; }

setsid $ARM_PY scripts/serve_arm.py --mode selection --model $ARM_MODEL --base $BASE --port $SPORT > $OUT/arm.log 2>&1 &
PIDS+=($!); wait_ok http://127.0.0.1:$SPORT/health $!
if [ "${SKIP_REPLAY:-0}" != 1 ]; then
  echo "[probe] ARM up; replaying stored base candidates"
  $ARM_PY scripts/probe_round2.py --c2 $C2 --mode replay --label replay --selector-port $SPORT \
    --states $STATES --out $OUT/replay.jsonl | grep SUMMARY
fi

for entry in $ACTORS; do
  label=${entry%%=*}
  if [[ $entry == *=* ]]; then model=${entry#*=}; elif [ $label = base ]; then model=$BASE; else model=/gpfs/scrubbed/piotrt/owrl/eval/merged/$label; fi
  [ -f $model/config.json ] || { echo "FATAL no model for $label at $model"; exit 1; }
  setsid $SERVER_PY -m sglang.launch_server --model-path $model --host 127.0.0.1 --port $APORT --dtype bfloat16 \
    --tp 1 --mem-fraction-static 0.4 --context-length 32768 --max-running-requests 48 --chunked-prefill-size 4096 \
    > $OUT/actor-$label.log 2>&1 &
  apid=$!; PIDS+=($apid); wait_ok http://127.0.0.1:$APORT/health_generate $apid
  $ARM_PY scripts/probe_round2.py --c2 $C2 --mode sample --label $label --actor-port $APORT --selector-port $SPORT \
    --states $STATES --out $OUT/sample-$label.jsonl | grep SUMMARY
  kill -- -$apid; wait $apid 2>/dev/null || true
done
echo "[probe] DONE -> $OUT"
