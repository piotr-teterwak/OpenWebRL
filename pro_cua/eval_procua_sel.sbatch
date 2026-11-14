#!/usr/bin/env bash
#SBATCH --job-name=procua-sel
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/procua-sel-%j.out
# PRO-CUA's frozen-VLM process reward model (Qwen3-VL-4B-Instruct + their PRM_PROMPT, pro_cua/serve_procua_selector.py) as a
# best-of-5 selector over our SFT actor, one benchmark per job. One GPU: SGLang actor (mem 0.45) + SGLang Qwen3-VL-4B-Instruct
# (mem 0.30) + the selector server. arm_eval flags as tillicum/eval_concurrent.sbatch / best_of_q/eval_q.sbatch.
#   BENCH=deepshop PASS=procua1 sbatch --exclude=g011,g013,g015,g021 pro_cua/eval_procua_sel.sbatch   (SMOKE=1: tasks 0-1)
set -euo pipefail
BENCH=${BENCH:-deepshop}; PASS=${PASS:?set PASS}; SMOKE=${SMOKE:-0}
VLM=${VLM:-/gpfs/scrubbed/weihanx/models/Qwen3-VL-4B-Instruct}
case "$BENCH" in
  online-mind2web) TASK_FILE=openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS/base+procuasel; mkdir -p $OUT
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
export OMP_NUM_THREADS=4
# Browser-port lock files default to the shared /tmp/slime_browser_local_process_ports; on nodes where another user
# owns it every port is silently skipped ("No free local_process env_server port", jobs 414605/414800). Use a private dir.
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="$OPENWEBRL_JOB_TMP/port_locks"; mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR"
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
# arm_eval loads .env with override=False; our .env key is stale since the Oct 31 rotation, so export the live one first
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'"); export OPENAI_API_KEY
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
export JUDGE_API_BASE=${JUDGE_API_BASE:-https://api.openai.com/v1}
# A 300-port block whose actor/selector ports and the whole browser range (+100..+299) are free (a single free port is
# not enough on a shared node: job 414605 lost all 595 WebVoyager tasks to "No free local_process env_server port").
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
    if all(free(b + k) for k in (10, 11, 12)) and all(free(b + k) for k in range(100, 300)):
        print(b); break
PYEOF
)
[ -n "$PORT0" ] || { echo "FATAL no free port block"; exit 1; }
cd $REPO
setsid $SERVER_PY -m sglang.launch_server --model-path $BASE --host 127.0.0.1 --port $((PORT0 + 10)) --dtype bfloat16 --tp 1 \
  --mem-fraction-static 0.45 --context-length 32768 --max-running-requests 24 --chunked-prefill-size 4096 --disable-cuda-graph \
  > $OUT/actor-server-$BENCH.log 2>&1 & ACTOR=$!
# Start the judge VLM only after the actor has taken its memory pool: launched together, the second server can find
# too little free memory for its 0.30 share (job 416091: 'Not enough memory').
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && break; kill -0 $ACTOR || break; sleep 10; done
setsid $SERVER_PY -m sglang.launch_server --model-path $VLM --served-model-name Qwen3-VL-4B-Instruct --host 127.0.0.1 --port $((PORT0 + 12)) \
  --dtype bfloat16 --tp 1 --mem-fraction-static 0.30 --context-length 16384 --max-running-requests 40 --disable-cuda-graph \
  > $OUT/vlm-server-$BENCH.log 2>&1 & VS=$!
setsid $ARM_PY pro_cua/serve_procua_selector.py --vlm-url http://127.0.0.1:$((PORT0 + 12))/v1 --vlm-name Qwen3-VL-4B-Instruct \
  --port $((PORT0 + 11)) --log $OUT/$BENCH-selector-calls.jsonl > $OUT/q-server-$BENCH.log 2>&1 & QS=$!
trap 'kill -- -$ACTOR -$VS -$QS 2>/dev/null || true; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 12))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; do
  sleep 10; kill -0 $ACTOR && kill -0 $VS && kill -0 $QS || { echo "FATAL a server died"; tail -20 $OUT/*-server-$BENCH.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL servers not healthy"; exit 1; }; done
echo "[procua-sel] servers healthy after $(( $(date +%s) - t0 )) s: $(curl -s http://127.0.0.1:$((PORT0 + 11))/health)"
TASKS=(); [ "$SMOKE" = 1 ] && TASKS=(--task-indices 0,1)
[ -n "${INDICES:-}" ] && TASKS=(--task-indices $INDICES)  # explicit task-file indices (reruns)
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode scalar --actor $BASE \
  --actor-port $((PORT0 + 10)) --selector-endpoint http://127.0.0.1:$((PORT0 + 11)) --output $OUT/$BENCH \
  --parallel ${PARALLEL:-12} --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature ${TEMP:-0.7} --top-p 0.9 \
  --max-new-tokens 1024 --max-steps 30 --task-timeout ${TASK_TIMEOUT:-1800} --step-timeout ${STEP_TIMEOUT:-180} --env-file .env \
  --browser-port-start $((PORT0 + 100)) --browser-port-end $((PORT0 + 299)) "${TASKS[@]}" > $OUT/eval-$BENCH.log 2>&1 || true
s=$OUT/$BENCH/summary.json; [ -f $s ] && echo "[procua-sel] $BENCH: $(tr -d '\n ' < $s)" || { echo "[procua-sel] $BENCH: incomplete"; tail -20 $OUT/eval-$BENCH.log; }
