#!/usr/bin/env bash
#SBATCH --job-name=redo-sftadm
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/redo-sftadm-%j.out
#
# Rerun invalid episodes of Zixian's SFT (sft1, head-k5-20261101) and SelectionARM (piotr5, controls-sameday-20261103)
# runs, matching the baseline reruns (tillicum/redo_baselines.py): same actor, seeds, T 1.0 / top-p 0.9, 1,024 tokens,
# 30 steps; step 600 s / task 3600 s timeouts. One benchmark per job, actor (+ SelectionARM 81b452d8) on one H200.
#   KIND=sft|adm BENCH=webvoyager PASS=zx-sft1-run1-redo2 INDICES=3,17,... sbatch tillicum/eval_redo_sftadm.sbatch
set -euo pipefail
KIND=${KIND:?sft or adm}; BENCH=${BENCH:?}; PASS=${PASS:?}; INDICES=${INDICES:?}
case "$BENCH" in
  online-mind2web) TASK_FILE=openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
ARM=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89
LABEL=$([ $KIND = adm ] && echo piotr5 || echo sft1)
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS/$LABEL; mkdir -p $OUT
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton; export OMP_NUM_THREADS=4
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="$OPENWEBRL_JOB_TMP/port_locks"; mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR"
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'"); export OPENAI_API_KEY
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
export JUDGE_API_BASE=${JUDGE_API_BASE:-https://api.openai.com/v1}
PORT0=$(python3 - <<'PYEOF'
import socket, random
def free(port):
    s = socket.socket(); s.settimeout(0.2)
    try:
        s.bind(('127.0.0.1', port)); return True
    except OSError:
        return False
    finally:
        s.close()
blocks = list(range(24000, 60000, 300)); random.shuffle(blocks)
for b in blocks:
    if all(free(b + k) for k in (10, 11)) and all(free(b + k) for k in range(100, 300)):
        print(b); break
PYEOF
)
[ -n "$PORT0" ] || { echo "FATAL no free port block"; exit 1; }
cd $REPO
MEM=$([ $KIND = adm ] && echo 0.45 || echo 0.8)
setsid $SERVER_PY -m sglang.launch_server --model-path $BASE --host 127.0.0.1 --port $((PORT0 + 10)) --dtype bfloat16 --tp 1 \
  --mem-fraction-static $MEM --context-length 32768 --max-running-requests 48 --chunked-prefill-size 4096 --disable-cuda-graph \
  > $OUT/actor-server-$BENCH.log 2>&1 & ACTOR=$!
PIDS="$ACTOR"
if [ $KIND = adm ]; then
  setsid $ARM_PY scripts/serve_arm.py --mode selection --model $ARM --base $BASE --port $((PORT0 + 11)) --prompt-format canonical \
    > $OUT/arm-server-$BENCH.log 2>&1 & PIDS="$PIDS $!"
fi
trap 'for p in $PIDS; do kill -- -$p 2>/dev/null; done; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && { [ $KIND = sft ] || curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; }; do
  sleep 10; for p in $PIDS; do kill -0 $p || { echo "FATAL a server died"; tail -20 $OUT/*-server-$BENCH.log; exit 1; }; done
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL servers not healthy"; exit 1; }; done
echo "[redo] $KIND $BENCH servers healthy after $(( $(date +%s) - t0 )) s"
MODE=$([ $KIND = adm ] && echo selection || echo baseline)
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode $MODE --actor $BASE \
  --actor-port $((PORT0 + 10)) --selector-endpoint http://127.0.0.1:$((PORT0 + 11)) --output $OUT/$BENCH \
  --parallel ${PARALLEL:-8} --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature ${TEMP:-1.0} --top-p 0.9 \
  --max-new-tokens 1024 --max-steps 30 --task-timeout ${TASK_TIMEOUT:-3600} --step-timeout ${STEP_TIMEOUT:-600} --env-file .env \
  --browser-port-start $((PORT0 + 100)) --browser-port-end $((PORT0 + 299)) --task-indices $INDICES > $OUT/eval-$BENCH.log 2>&1 || true
s=$OUT/$BENCH/summary.json; [ -f $s ] && echo "[redo] $KIND $BENCH: $(tr -d '\n ' < $s)" || { echo "[redo] $KIND $BENCH: incomplete"; tail -20 $OUT/eval-$BENCH.log; }
