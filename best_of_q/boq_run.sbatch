#!/usr/bin/env bash
#SBATCH --job-name=boq-run
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-run-%j.out
# One paper-faithful Best-of-Q run on WebVoyager (arXiv 2601.22701): frozen OpenWebRL-4B-SFT actor (SGLang, mem 0.45),
# N = 3 candidates (greedy + distinct samples, best_of_q/boq_selector.py), 30 steps, our WebVoyager judge.
#   POLICY=epsilon SEED=1 RUN=seed-s1 sbatch boq_run.sbatch          # exploration run (eps 0.5); COLLECT=1 by default
#   POLICY=q Q=.../q_function_best.pt SEED=11 RUN=cyc1-s11 sbatch ...  # Best-of-Q (argmax Q) collection or eval
#   POLICY=random | base | greedy                                      # paper baselines; base = our N=1 at T 0.7
#   BOQ_N=5 BOQ_DRAWS=5: argmax over 5 distinct candidates instead of the paper's N = 3 (inference-time N ablation)
# COLLECT=1: after the run, extract transitions and embed them with the frozen Qwen2.5-VL-3B (BOQ_EMBEDDER) into
# RUN_DIR/boq/. SMOKE=1: tasks 0,1 only.
set -euo pipefail
POLICY=${POLICY:?}; SEED=${SEED:?}; RUN=${RUN:?}; COLLECT=${COLLECT:-1}; SMOKE=${SMOKE:-0}; Q=${Q:-}
ROOT=${BOQ_ROOT:-/gpfs/scrubbed/piotrt/owrl/best_of_q_v2}; OUT=$ROOT/runs/$RUN; mkdir -p $OUT
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
export BOQ_EMBEDDER=${BOQ_EMBEDDER:-/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct}
# TASK_FILE / PROTOCOL: what the run collects on or is scored on (default WebVoyager, the paper's setting); the
# training-pool variant collects on best_of_q_pool/tasks/pool_seed<k>.jsonl and evaluates on the three benchmarks.
TASK_FILE=${TASK_FILE:-openwebrl/data/eval/webvoyager_fara.jsonl}; PROTOCOL=${PROTOCOL:-webvoyager}
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
export OMP_NUM_THREADS=4
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="$OPENWEBRL_JOB_TMP/port_locks"; mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR"
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'"); export OPENAI_API_KEY
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
export JUDGE_API_BASE=${JUDGE_API_BASE:-https://api.openai.com/v1}
[ "$POLICY" = q ] && { [ -f "$Q" ] || { echo "FATAL POLICY=q needs Q=<q_function_best.pt>"; exit 1; }; }
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
echo "[boq] run $RUN policy $POLICY seed $SEED q ${Q:-none} collect $COLLECT node $(hostname -s) job $SLURM_JOB_ID"
setsid $SERVER_PY -m sglang.launch_server --model-path $BASE --host 127.0.0.1 --port $((PORT0 + 10)) --dtype bfloat16 --tp 1 \
  --mem-fraction-static 0.45 --context-length 32768 --max-running-requests 32 --chunked-prefill-size 4096 --disable-cuda-graph \
  > $OUT/actor-server.log 2>&1 & ACTOR=$!
QS=""
if [ "$POLICY" = q ]; then
  setsid $ARM_PY best_of_q/serve_q_selector.py --q $Q --port $((PORT0 + 11)) --log $OUT/selector-calls.jsonl > $OUT/q-server.log 2>&1 & QS=$!
fi
# set -e also applies inside the EXIT trap: a kill of an already-stopped server group must not fail the job (415760).
trap 'kill -- -$ACTOR ${QS:+-$QS} 2>/dev/null || true; sleep 5; rm -rf "${OPENWEBRL_JOB_TMP:-/nonexistent}"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$((PORT0 + 10))/health_generate >/dev/null && { [ -z "$QS" ] || curl -sf http://127.0.0.1:$((PORT0 + 11))/health >/dev/null; }; do
  sleep 10; kill -0 $ACTOR && { [ -z "$QS" ] || kill -0 $QS; } || { echo "FATAL a server died"; tail -20 $OUT/*server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 1200 ] || { echo "FATAL servers not healthy after 20 min"; exit 1; }; done
echo "[boq] servers healthy after $(( $(date +%s) - t0 )) s ${QS:+$(curl -s http://127.0.0.1:$((PORT0 + 11))/health)}"
TASKS=(); [ "$SMOKE" = 1 ] && TASKS=(--task-indices 0,1)
TEMP=${TEMP:-0.7}; [ "$POLICY" = greedy ] && TEMP=0.0  # actor sampling for base/greedy N=1; Best-of-Q sets its own per draw
EVAL=(--actor $BASE --actor-port $((PORT0 + 10)) --output $OUT --parallel 12 --protocol $PROTOCOL --task-file $TASK_FILE
      --seed $SEED --temperature $TEMP --top-p 0.9 --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file .env
      --browser-port-start $((PORT0 + 100)) --browser-port-end $((PORT0 + 299)) "${TASKS[@]}")
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); RESERVE=$([ "$COLLECT" = 1 ] && echo 2700 || echo 300)
REMAIN=$(( END - $(date +%s) - RESERVE ))
case "$POLICY" in
  epsilon|random|q) CMD=($ARM_PY best_of_q/boq_run.py --policy $POLICY ${QS:+--q-endpoint http://127.0.0.1:$((PORT0 + 11))} --n ${BOQ_N:-3} --draws ${BOQ_DRAWS:-3} -- "${EVAL[@]}") ;;
  base) CMD=($ARM_PY -m openwebrl.arm_eval --mode baseline "${EVAL[@]}") ;;
  greedy) CMD=($ARM_PY -m openwebrl.arm_eval --mode baseline "${EVAL[@]}") ;;
  *) echo "FATAL bad POLICY=$POLICY"; exit 1 ;;
esac
# Resuming a stopped run: the browser port block is drawn per job, and arm_eval refuses a run directory whose saved
# manifest differs (jobs 418059/418060), so carry the new range into the saved manifest; nothing else may differ.
if [ -f $OUT/manifest.json ]; then python3 -c "import json,sys; p=sys.argv[1]; m=json.load(open(p)); m['browser_port_range']=[int(sys.argv[2]),int(sys.argv[3])]; json.dump(m,open(p,'w'),indent=2); open(p,'a').write('\n')" $OUT/manifest.json $((PORT0 + 100)) $((PORT0 + 299)); fi
timeout --signal=TERM --kill-after=120 $REMAIN "${CMD[@]}" > $OUT/eval.log 2>&1 || true
s=$OUT/summary.json; [ -f $s ] && echo "[boq] $RUN: $(tr -d '\n ' < $s)" || { echo "[boq] $RUN: incomplete"; tail -20 $OUT/eval.log; }
if [ "$COLLECT" = 1 ] && [ "$POLICY" != base ] && [ "$POLICY" != greedy ]; then
  kill -- -$ACTOR ${QS:+-$QS} 2>/dev/null || true; sleep 10
  $ARM_PY best_of_q/extract_boq.py $OUT
  $ARM_PY best_of_q/embed.py states --rows $OUT/boq/rows.jsonl --out $OUT/boq/emb --batch 8
  $ARM_PY best_of_q/embed.py texts --rows $OUT/boq/rows.jsonl --out $OUT/boq/emb
  echo "[boq] $RUN collected: $(cat $OUT/boq/extract.json | tr -d '\n ')"
fi
