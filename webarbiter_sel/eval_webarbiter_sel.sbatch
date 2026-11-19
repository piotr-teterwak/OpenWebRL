#!/usr/bin/env bash
#SBATCH --job-name=webarbiter-sel
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/webarbiter-sel-%j.out
# WebArbiter-7B (webarbiter_sel/serve_webarbiter_selector.py: released pairwise prompt + knockout tournament over the
# accessibility tree of the current viewport) as a best-of-5 selector over our SFT actor, one benchmark per job.
# arm_eval runs with --selector-page-tree, so the browser envs add WebArena's tree to each observation.
# One GPU: SGLang actor (mem 0.45) + SGLang WebArbiter-7B (0.6 of what is free when it starts: SGLang fractions apply to
# the memory free at its own start, webdreamer/eval_webdreamer.sbatch; fa3, as every server that works here) + selector.
# The selector also calls gpt-4o-mini once per step for the released state-difference summary.
#   BENCH=deepshop PASS=webarbiter1 sbatch --exclude=g001,g008,g011,g013,g015,g021 webarbiter_sel/eval_webarbiter_sel.sbatch
#   (SMOKE=N: tasks 0..N-1; INDICES=a,b,c; RANGE=a-b: one shard)
# Several benchmarks in one job (smoke tests): BENCH="online-mind2web webvoyager deepshop" INDICES="2,5;0,1;1,2" runs
# one arm_eval per benchmark concurrently on the same servers, each with its own slice of the browser port block.
set -euo pipefail
BENCH=${BENCH:-deepshop}; PASS=${PASS:?set PASS}; SMOKE=${SMOKE:-0}
ARBITER=${ARBITER:-/gpfs/scrubbed/piotrt/owrl/models/WebArbiter-7B}
read -r -a BENCHES <<< "$BENCH"
IFS=';' read -r -a BENCH_INDICES <<< "${INDICES:-}"
for b in "${BENCHES[@]}"; do
  case "$b" in online-mind2web|webvoyager|deepshop) ;; *) echo "bad BENCH=$b"; exit 1 ;; esac
done
task_file() { case "$1" in
  online-mind2web) echo openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) echo openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) echo openwebrl/data/eval/deepshop.jsonl ;; esac; }
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS/base+webarbitersel${RANGE:+-r$RANGE}; mkdir -p $OUT
TAG=${BENCH// /+}
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
export OMP_NUM_THREADS=4
# Browser-port lock files default to the shared /tmp/slime_browser_local_process_ports; on nodes where another user
# owns it every port is silently skipped ("No free local_process env_server port", jobs 414605/414800). Use a private dir.
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="$OPENWEBRL_JOB_TMP/port_locks"; mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR"
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
# arm_eval loads .env with override=False; our .env key is stale since the Oct 31 rotation, so export the live one first
# (also read by the selector for the gpt-4o-mini state-difference call)
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
  > $OUT/actor-server-$TAG.log 2>&1 & ACTOR=$!
# Start the judge only after the actor has taken its memory pool (job 416091: 'Not enough memory' when launched together).
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && break; kill -0 $ACTOR || break; sleep 10; done
setsid $SERVER_PY -m sglang.launch_server --model-path $ARBITER --served-model-name WebArbiter --host 127.0.0.1 --port $((PORT0 + 12)) \
  --dtype bfloat16 --tp 1 --attention-backend fa3 --mem-fraction-static ${ARBITER_MEM:-0.6} --context-length 32768 \
  --max-running-requests 32 --random-seed 42 --disable-cuda-graph > $OUT/arbiter-server-$TAG.log 2>&1 & AS=$!
setsid $ARM_PY webarbiter_sel/serve_webarbiter_selector.py --arbiter-url http://127.0.0.1:$((PORT0 + 12))/v1 --arbiter-name WebArbiter \
  --port $((PORT0 + 11)) --log $OUT/$TAG-selector-calls.jsonl > $OUT/q-server-$TAG.log 2>&1 & QS=$!
trap 'kill -- -$ACTOR -$AS -$QS 2>/dev/null || true; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 12))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; do
  sleep 10; kill -0 $ACTOR && kill -0 $AS && kill -0 $QS || { echo "FATAL a server died"; for f in $OUT/*-server-$TAG.log; do tail -5 $f; done; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL servers not healthy"; exit 1; }; done
echo "[webarbiter-sel] servers healthy after $(( $(date +%s) - t0 )) s: $(curl -s http://127.0.0.1:$((PORT0 + 11))/health)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
NB=${#BENCHES[@]}; SLICE=$(( 200 / NB )); EVALS=()
for k in "${!BENCHES[@]}"; do
  b=${BENCHES[$k]}; TASKS=()
  [ "$SMOKE" != 0 ] && TASKS=(--task-indices $(seq -s, 0 $((SMOKE - 1))))
  [ -n "${BENCH_INDICES[$k]:-}" ] && TASKS=(--task-indices ${BENCH_INDICES[$k]})  # explicit comma-separated task-file indices
  [ -n "${RANGE:-}" ] && TASKS=(--task-indices $(seq -s, ${RANGE%-*} ${RANGE#*-}))  # RANGE=a-b: one shard of a benchmark
  PS=$(( PORT0 + 100 + k * SLICE )); PE=$(( PS + SLICE - 1 ))
  timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode scalar --selector-page-tree --actor $BASE \
    --actor-port $((PORT0 + 10)) --selector-endpoint http://127.0.0.1:$((PORT0 + 11)) --output $OUT/$b \
    --parallel ${PARALLEL:-12} --protocol ${b//-/_} --task-file $(task_file $b) --seed 42 --temperature ${TEMP:-1.0} --top-p 0.9 \
    --max-new-tokens 1024 --max-steps 30 --task-timeout ${TASK_TIMEOUT:-1800} --step-timeout ${STEP_TIMEOUT:-180} --env-file .env \
    --browser-port-start $PS --browser-port-end $PE "${TASKS[@]}" > $OUT/eval-$b.log 2>&1 &
  EVALS+=($!)
done
# Watchdog: if a model server dies mid-run, every later task fails as "unavailable" (job 423033). Stop the evals then.
any_eval() { for e in "${EVALS[@]}"; do kill -0 $e 2>/dev/null && return 0; done; return 1; }
while any_eval; do
  sleep 30
  for pid in $ACTOR $AS $QS; do kill -0 $pid 2>/dev/null || { echo "[webarbiter-sel] FATAL server $pid died mid-run at $(date +%T); stopping eval"; for f in $OUT/*-server-$TAG.log; do tail -3 $f; done; kill -TERM "${EVALS[@]}" 2>/dev/null; sleep 60; kill -KILL "${EVALS[@]}" 2>/dev/null; break 2; }; done
done
for e in "${EVALS[@]}"; do wait $e 2>/dev/null || true; done
for b in "${BENCHES[@]}"; do
  s=$OUT/$b/summary.json; [ -f $s ] && echo "[webarbiter-sel] $b: $(tr -d '\n ' < $s)" || { echo "[webarbiter-sel] $b: incomplete"; tail -20 $OUT/eval-$b.log; }
done
