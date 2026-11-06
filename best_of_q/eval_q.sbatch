#!/usr/bin/env bash
#SBATCH --job-name=boq-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-eval-%j.out
# Best-of-Q as a best-of-5 selector over the SFT actor, one benchmark per job, one GPU shared by the SGLang actor
# (mem fraction 0.45) and the Q-selector server (frozen 4B embedder + MLP). arm_eval flags are those of
# tillicum/eval_concurrent.sbatch (the base / base+ARM runs this pairs with): seed 42, T 0.7, top-p 0.9, 30 steps.
#   BENCH=deepshop PASS=boq1 sbatch --exclude=g011,g013,g015,g021 best_of_q/eval_q.sbatch     (SMOKE=1: tasks 0-1)
set -euo pipefail
BENCH=${BENCH:-deepshop}; PASS=${PASS:?set PASS}; SMOKE=${SMOKE:-0}
Q=${Q:-/gpfs/scrubbed/piotrt/owrl/best_of_q/q-s0-best/q_function_best.pt}
case "$BENCH" in
  online-mind2web) TASK_FILE=openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS/base+boq; mkdir -p $OUT
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
    if all(free(b + k) for k in (10, 11)) and all(free(b + k) for k in range(100, 300)):
        print(b); break
PYEOF
)
[ -n "$PORT0" ] || { echo "FATAL no free port block"; exit 1; }
cd $REPO
setsid $SERVER_PY -m sglang.launch_server --model-path $BASE --host 127.0.0.1 --port $((PORT0 + 10)) --dtype bfloat16 --tp 1 \
  --mem-fraction-static 0.45 --context-length 32768 --max-running-requests 24 --chunked-prefill-size 4096 --disable-cuda-graph \
  > $OUT/actor-server-$BENCH.log 2>&1 & ACTOR=$!
setsid $ARM_PY best_of_q/serve_q_selector.py --q $Q --port $((PORT0 + 11)) --log $OUT/$BENCH-selector-calls.jsonl > $OUT/q-server-$BENCH.log 2>&1 & QS=$!
trap 'kill -- -$ACTOR -$QS 2>/dev/null; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; do
  sleep 10; kill -0 $ACTOR && kill -0 $QS || { echo "FATAL a server died"; tail -20 $OUT/*-server-$BENCH.log $OUT/q-server-$BENCH.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL servers not healthy"; exit 1; }; done
echo "[boq] servers healthy after $(( $(date +%s) - t0 )) s: $(curl -s http://127.0.0.1:$((PORT0 + 11))/health)"
TASKS=(); [ "$SMOKE" = 1 ] && TASKS=(--task-indices 0,1)
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode scalar --actor $BASE \
  --actor-port $((PORT0 + 10)) --selector-endpoint http://127.0.0.1:$((PORT0 + 11)) --output $OUT/$BENCH \
  --parallel 12 --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature ${TEMP:-0.7} --top-p 0.9 \
  --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file .env \
  --browser-port-start $((PORT0 + 100)) --browser-port-end $((PORT0 + 299)) "${TASKS[@]}" > $OUT/eval-$BENCH.log 2>&1 || true
s=$OUT/$BENCH/summary.json; [ -f $s ] && echo "[boq] $BENCH: $(tr -d '\n ' < $s)" || { echo "[boq] $BENCH: incomplete"; tail -20 $OUT/eval-$BENCH.log; }
