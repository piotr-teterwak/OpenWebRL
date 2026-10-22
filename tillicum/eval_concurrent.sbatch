#!/usr/bin/env bash
#SBATCH --job-name=owrl-eval5
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:5
# One H200 per model, all five concurrent. Zixian's 2-model x 12-browser eval
# (285854) used a mean of 8.1 busy cores and 65G MaxRSS; five models scale that
# to ~20 cores / ~165G. 40 CPUs is the node's per-GPU share (64 per 8 GPUs).
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
# 285854 did 200 tasks per model in 58 min at 12 parallel; 300 is ~1.5h, plus
# merging and server start-up. A timed-out pass resumes on resubmission.
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/eval5-%j.out
#
# Concurrent Online-Mind2Web eval of the base actor and the four retrained
# policies (1A, random control, joint SFT, joint DPO), same protocol as
# Zixian's run_arm_full300_checkpoint_eval.py: n=1 actor (arm_eval --mode
# baseline), T=0.7 top-p 0.9, 1024 new tokens, 30 steps, 12 browsers per
# model, o4-mini judge. All 300 tasks; overall-with-invalid-as-failure is the
# headline number.
#
#   PASS=1 sbatch tillicum/eval_concurrent.sbatch
#   PASS=2 sbatch tillicum/eval_concurrent.sbatch   # independent repeat
#   MODELS=base,arm-lr1e-5,arm-lr5e-5,arm-lr1e-4,arm-lr2e-4 PASS=grid1 sbatch tillicum/eval_concurrent.sbatch
#   MODELS=base,1a,random,joint-sft PASS=1 sbatch --gpus=h200:4 --cpus-per-task=32 --mem=256G tillicum/eval_concurrent.sbatch
#   SMOKE=1 PASS=smoke sbatch --qos=debug --gpus=h200:1 --cpus-per-task=8 --mem=120G --time=00:40:00 \
#     tillicum/eval_concurrent.sbatch   # base model, tasks 0-1: server, browsers, judge, key
# BENCH=webvoyager (595 tasks) or BENCH=deepshop (150, all Amazon) switches task file and
# judge (gpt-4o, 30 screenshots); default online-mind2web. Results go to $OUT/<label>/$BENCH.
set -euo pipefail
PASS="${PASS:?set PASS=1 or PASS=2}"
SMOKE="${SMOKE:-0}"
BENCH="${BENCH:-online-mind2web}"
case "$BENCH" in
  online-mind2web) TASK_FILE=openwebrl/data/eval/online-mind2web.jsonl ;;
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python
SERVER_PY=$ZRT/venv/bin/python
OWRL=/gpfs/scrubbed/piotrt/owrl
EVAL=$OWRL/eval
OUT=$EVAL/pass$PASS
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT

export OPENWEBRL_CACHE_ROOT=$OWRL/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=4
# Jobs inherit http(s)_proxy (tillicum-dtn01:3128), which the judge needs to reach
# OpenAI, but no_proxy only lists "localhost": sglang's self-check to 127.0.0.1
# went to Squid and got a 503. Keep loopback traffic off the proxy.
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
# Browser env servers reserve ports via lock files in a node-shared /tmp directory by
# default; files another user left there are unwritable to us, every port then looks
# taken, and all tasks fail in seconds (376803 on g014). Use a job-private lock dir;
# the code still probes each port for actual use, so jobs sharing a node cannot clash.
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=$OPENWEBRL_JOB_TMP/port-locks
# .env resolves the judge key with a shell command, so bash must expand it;
# arm_eval's own dotenv load is override=False and keeps this value.
set -a; source $REPO/.env; set +a
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL OPENAI_API_KEY did not resolve"; exit 1; }
cd $REPO
mkdir -p $OUT

# label | run root | checkpoint (relative to run root), or a bare model path for base
SPECS=(
  "base|$BASE|"
  "1a|$OWRL/c2-1a-retrain/full|student/epoch-1"
  "random|$OWRL/c2-random-control/full|student/epoch-1"
  "joint-sft|$OWRL/joint-retrain/runs/joint-v2-sft-2gpu|student/update-000174"
  "joint-dpo|$OWRL/joint-retrain/runs/joint-v2-dpo-2gpu|student/update-000174"
)
# SFT learning-rate grid (tillicum/c2_sft_grid.sbatch): already merged, so bare model paths.
for v in arm random; do for lr in 1e-5 5e-5 1e-4 2e-4; do
  SPECS+=("$v-lr$lr|$OWRL/sft-grid/$v-lr$lr/merged|")
done; done
# Joint DPO / RPO learning-rate sweep (tillicum/train_joint.sbatch RUN=...): merged here on first use.
for r in dpo-lr5e-5 dpo-lr1e-4 dpo-lr2e-4 rpo-a1-lr1e-5 rpo-a1-lr5e-5 rpo-a1-lr1e-4 rpo-a1-lr2e-4; do
  SPECS+=("$r|$OWRL/joint-retrain/runs/$r|student/update-000174")
done
SPECS+=("k2-rpo-a1-lr5e-5|$OWRL/joint-retrain/runs/k2-rpo-a1-lr5e-5|student/update-000271")
for b in 0.3 0.5 1.0; do SPECS+=("rpo-a1-lr5e-5-b$b|$OWRL/joint-retrain/runs/rpo-a1-lr5e-5-b$b|student/update-000174"); done
# Run-on fixes: matched action-count pairs (4,078 pairs -> 128 updates) and DPO without end of turn.
SPECS+=("rpo-matched-lr5e-5|$OWRL/joint-retrain/runs/rpo-matched-lr5e-5|student/update-000128")
# Ablation: is it the matched data or RPO? Same data, lr, beta and updates; SFT only / DPO only.
for r in sft-matched-lr5e-5 dpo-matched-lr5e-5; do SPECS+=("$r|$OWRL/joint-retrain/runs/$r|student/update-000128"); done
SPECS+=("rpo-noeot-lr5e-5|$OWRL/joint-retrain/runs/rpo-noeot-lr5e-5|student/update-000174")
SPECS+=("rpo-a2-lr5e-5|$OWRL/joint-retrain/runs/rpo-a2-lr5e-5|student/update-000174")
TASKS=()
if [ "$SMOKE" = 1 ]; then SPECS=("${SPECS[0]}"); TASKS=(--task-indices ${SMOKE_TASKS:-0,1}); fi
# MODELS=base,1a,... evaluates a subset; pass --gpus to match (one per model).
if [ -n "${MODELS:-}" ]; then
  KEEP=()
  for spec in "${SPECS[@]}"; do [[ ",$MODELS," == *",${spec%%|*},"* ]] && KEEP+=("$spec"); done
  [ ${#KEEP[@]} -eq $(tr ',' '\n' <<<"$MODELS" | wc -l) ] || { echo "FATAL unknown label in MODELS=$MODELS"; exit 1; }
  SPECS=("${KEEP[@]}")
fi
N=${#SPECS[@]}
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ ${#GPUS[@]} -eq $N ] || { echo "FATAL need $N GPUs, got ${#GPUS[@]}"; exit 1; }

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || kill $p 2>/dev/null || true; done; rm -rf "$OPENWEBRL_JOB_TMP"; }
trap cleanup EXIT

# 1. Merge each finished LoRA into a standalone model, once; pass 2 reuses them.
model_path() { local label=$1 root=$2 ckpt=${3:-}; [ -z "$ckpt" ] && echo $root || echo $EVAL/merged/$label; }
i=0; MERGE_PIDS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read label root ckpt <<<"$spec"
  if [ -n "$ckpt" ] && [ ! -f $EVAL/merged/$label/merge-manifest.json ]; then
    [ -f $root/student/complete.json ] || { echo "FATAL $label training not complete: $root"; exit 1; }
    mkdir -p $EVAL/merged
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $ARM_PY scripts/merge_arm_c2_student.py --run-root $root \
      --checkpoint $root/$ckpt --output $EVAL/merged/$label > $EVAL/merged/$label.log 2>&1 &
    MERGE_PIDS+=($!)
  fi
  i=$((i+1))
done
for p in "${MERGE_PIDS[@]}"; do wait $p || { echo "FATAL a merge failed; see $EVAL/merged/*.log"; exit 1; }; done
echo "[eval] models ready: $(ls $EVAL/merged 2>/dev/null | grep -v log | tr '\n' ' ')"

# 2. One sglang server per GPU, same flags as Zixian's worker. Ports are derived
# from the GPU slot in a range her scripts do not use (they take 19100-19699).
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)   # %e is local time
i=0
for spec in "${SPECS[@]}"; do
  IFS='|' read label root ckpt <<<"$spec"
  mkdir -p $OUT/$label
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $SERVER_PY -m sglang.launch_server --model-path "$(model_path "$label" "$root" "$ckpt")" \
    --host 127.0.0.1 --port $((20010 + 300*i)) --dtype bfloat16 --tp 1 --mem-fraction-static 0.45 \
    --context-length 32768 --max-running-requests 24 --chunked-prefill-size 4096 --disable-cuda-graph \
    > $OUT/$label/actor-server-$BENCH.log 2>&1 &
  PIDS+=($!); i=$((i+1))
done
for i in $(seq 0 $((N-1))); do
  port=$((20010 + 300*i)); t0=$(date +%s)
  until curl -sf http://127.0.0.1:$port/health_generate >/dev/null; do
    kill -0 ${PIDS[$i]} 2>/dev/null || { echo "FATAL server $i died"; tail -20 $OUT/*/actor-server-$BENCH.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL server $i not healthy after 15 min"; exit 1; }
    sleep 5
  done
done
echo "[eval] $N servers healthy"

# 3. All five evals at once; stop 5 min before the limit (results so far are kept
# and a resubmission skips them).
REMAIN=$(( END - $(date +%s) - 300 ))
[ $REMAIN -gt $([ "$SMOKE" = 1 ] && echo 300 || echo 1800) ] || { echo "FATAL only ${REMAIN}s left; check the deadline math"; exit 1; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 300 > $OUT/gpu-util.csv &
PIDS+=($!)
i=0; EVAL_PIDS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read label root ckpt <<<"$spec"
  b=$((20000 + 300*i))
  timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode baseline \
    --actor "$(model_path "$label" "$root" "$ckpt")" --actor-port $((b + 10)) --output $OUT/$label/$BENCH \
    --parallel 12 --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file .env \
    --browser-port-start $((b + 100)) --browser-port-end $((b + 299)) "${TASKS[@]}" > $OUT/$label/eval-$BENCH.log 2>&1 &
  EVAL_PIDS+=($!); i=$((i+1))
done
echo "[eval] pass $PASS running: ${SPECS[*]%%|*}"
rc=0; for p in "${EVAL_PIDS[@]}"; do wait $p || rc=$?; done

for spec in "${SPECS[@]}"; do
  label=${spec%%|*}; s=$OUT/$label/$BENCH/summary.json
  n=$(ls $OUT/$label/$BENCH/results 2>/dev/null | wc -l)
  [ -f $s ] && echo "[eval] $label: $(tr -d '\n ' < $s)" || echo "[eval] $label: incomplete ($n results)"
done
exit $rc
