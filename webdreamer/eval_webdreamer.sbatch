#!/usr/bin/env bash
#SBATCH --job-name=webdreamer
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/webdreamer-%j.out
# WebDreamer baseline (webdreamer/serve_webdreamer_selector.py) as a best-of-5 selector over our SFT actor, one benchmark per
# job: Dreamer-7B world model predicts each candidate's effect, Qwen3-VL-4B-Instruct scores it with WebDreamer's value
# prompt. One GPU: SGLang actor (mem 0.45) + SGLang Qwen3-VL-4B-Instruct (0.25) + SGLang Dreamer-7B (0.6 of what is free when it starts, ~55 GB: SGLang fractions apply to the memory free at its own
# start, so 0.22 left no KV cache, job 422963; bfloat16, not the card's float16: float16 needs a flashinfer JIT build that fails
# to link -lcudart here, job 422971; and fa3 forced: SGLang picks flashinfer for Qwen2-VL, whose JIT build fails the same way
# in bf16, job 422975, while every server that works here runs fa3) + selector.
#   BENCH=deepshop PASS=webdreamer1 TEMP=1.0 sbatch --exclude=g011,g013,g015,g021 webdreamer/eval_webdreamer.sbatch
#   (SMOKE=N: tasks 0..N-1)
set -euo pipefail
BENCH=${BENCH:-deepshop}; PASS=${PASS:?set PASS}; SMOKE=${SMOKE:-0}
VLM=${VLM:-/gpfs/scrubbed/weihanx/models/Qwen3-VL-4B-Instruct}
DREAMER=${DREAMER:-/gpfs/scrubbed/piotrt/owrl/models/Dreamer-7B}; NSAMP=${NSAMP:-10}; SIMS=${SIMS:-1}
case "$BENCH" in
  online-mind2web) TASK_FILE=openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS/base+webdreamer${RANGE:+-r$RANGE}; mkdir -p $OUT
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
    if all(free(b + k) for k in (10, 11, 12, 13)) and all(free(b + k) for k in range(100, 300)):
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
  --dtype bfloat16 --tp 1 --mem-fraction-static 0.25 --context-length 16384 --max-running-requests 64 --disable-cuda-graph \
  > $OUT/vlm-server-$BENCH.log 2>&1 & VS=$!
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$((PORT0 + 12))/health_generate >/dev/null && break; kill -0 $VS || break; sleep 10; done
setsid $SERVER_PY -m sglang.launch_server --model-path $DREAMER --served-model-name Dreamer-7B --host 127.0.0.1 --port $((PORT0 + 13)) \
  --dtype bfloat16 --tp 1 --attention-backend fa3 --mem-fraction-static ${DREAMER_MEM:-0.6} --context-length 16384 --max-running-requests 32 --disable-cuda-graph \
  > $OUT/dreamer-server-$BENCH.log 2>&1 & DS=$!
setsid $ARM_PY webdreamer/serve_webdreamer_selector.py --dreamer-url http://127.0.0.1:$((PORT0 + 13))/v1 --dreamer-name Dreamer-7B \
  --value-url http://127.0.0.1:$((PORT0 + 12))/v1 --value-name Qwen3-VL-4B-Instruct --n $NSAMP --sims $SIMS --value-concurrency ${VALUE_CONC:-4} \
  --port $((PORT0 + 11)) --log $OUT/$BENCH-selector-calls.jsonl > $OUT/q-server-$BENCH.log 2>&1 & QS=$!
trap 'kill -- -$ACTOR -$VS -$DS -$QS 2>/dev/null || true; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 12))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 13))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; do
  sleep 10; kill -0 $ACTOR && kill -0 $VS && kill -0 $DS && kill -0 $QS || { echo "FATAL a server died"; for f in $OUT/*-server-$BENCH.log; do tail -5 $f; done; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL servers not healthy"; exit 1; }; done
echo "[webdreamer] servers healthy after $(( $(date +%s) - t0 )) s: $(curl -s http://127.0.0.1:$((PORT0 + 11))/health)"
TASKS=(); [ "$SMOKE" != 0 ] && TASKS=(--task-indices $(seq -s, 0 $((SMOKE - 1))))
[ -n "${INDICES:-}" ] && TASKS=(--task-indices $INDICES)  # explicit comma-separated task-file indices (timeout reruns)
[ -n "${RANGE:-}" ] && TASKS=(--task-indices $(seq -s, ${RANGE%-*} ${RANGE#*-}))  # RANGE=a-b: one shard of a benchmark
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode scalar --actor $BASE \
  --actor-port $((PORT0 + 10)) --selector-endpoint http://127.0.0.1:$((PORT0 + 11)) --output $OUT/$BENCH \
  --parallel 12 --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature ${TEMP:-1.0} --top-p 0.9 \
  --max-new-tokens 1024 --max-steps 30 --task-timeout ${TASK_TIMEOUT:-1800} --step-timeout ${STEP_TIMEOUT:-180} --env-file .env \
  --browser-port-start $((PORT0 + 100)) --browser-port-end $((PORT0 + 299)) "${TASKS[@]}" > $OUT/eval-$BENCH.log 2>&1 &
EVAL=$!
# Watchdog: if a model server dies mid-run, every later task fails as "unavailable" (job 423033: the value server died
# 10 min in and 145 of 150 DeepShop tasks were lost). Stop the eval as soon as any server is gone.
while kill -0 $EVAL 2>/dev/null; do
  sleep 30
  for pid in $ACTOR $VS $DS $QS; do kill -0 $pid 2>/dev/null || { echo "[webdreamer] FATAL server $pid died mid-run at $(date +%T); stopping eval"; for f in $OUT/*-server-$BENCH.log; do tail -3 $f; done; kill -TERM $EVAL 2>/dev/null; sleep 60; kill -KILL $EVAL 2>/dev/null; break 2; }; done
done
wait $EVAL 2>/dev/null || true
s=$OUT/$BENCH/summary.json; [ -f $s ] && echo "[webdreamer] $BENCH: $(tr -d '\n ' < $s)" || { echo "[webdreamer] $BENCH: incomplete"; tail -20 $OUT/eval-$BENCH.log; }
