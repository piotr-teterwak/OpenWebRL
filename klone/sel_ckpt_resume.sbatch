#!/bin/bash
#SBATCH --job-name=owrl-selck
# l40s, not a40: this eval is CPU-bound on 8 concurrent chromiums, and l40s
# nodes have 128 CPUs/8 GPUs (16 per GPU) vs a40's 52/8 (6.5 per GPU), so 16
# cpus-per-task is fair share here instead of 2.5x oversubscribed. Same 48GB,
# ample for a 4B model + KV. dan29 uses a40 only because it works once
# chromium launches fast; nothing about the workload requires it.
# NOT ckpt: scalar needs ~9h (measured 99s/task x 300) and preemption killed
# the one attempt that was working (39447296, clean at 26/300). zs completed
# here in 3:40 on the same 2-GPU shape.
# Preemptible variant. Safe to preempt ONLY because run_evaluate.py now takes
# --resume: it reuses the newest matching output directory and skips tasks that
# already have a usable record, so a requeue continues instead of restarting.
# Without that flag this partition cost us a clean 26/300 scalar run.
# p100 and l40 are excluded: p100 is sm60 (unsupported) and 16GB.
#SBATCH --account=ckpt-krishna
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt-gpu
#SBATCH --constraint="a100|a40|l40s|h200"
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=2
#SBATCH --mem=180G
# gpu-l40s has MaxTime=UNLIMITED. 12h was fine for arm (36 s/task) and zs
# (44 s/task) but NOT for scalar: its judge is an HF eager PeftModel rather
# than an sglang server, measured at ~185 s/task, i.e. ~15h for 300 tasks.
# A running job's limit cannot be raised without operator rights, so set it
# generously up front -- an over-long limit costs nothing here, a short one
# costs the whole run.
#SBATCH --time=40:00:00
#SBATCH --output=/gscratch/cse/piotrt/logs/eval-%j.out
#
# Online-Mind2Web eval of OpenWebRL-4B-SFT.
#   ARM=greedy   -> TEMPERATURE=0.0 TOP_P=1.0            (deterministic)
#   ARM=nucleus  -> TEMPERATURE=0.7 TOP_P=0.8 + pytorch sampling backend
# Hardening below is adapted from dan29's sbatch_eval_owrlsft_om2w.sh; each
# item there is documented as having killed at least one of their runs.
set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"; export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

export PROJ=/gscratch/cse/piotrt
export REPO=/gscratch/krishna/piotrt/openwebrl-eval   # worktree with the arbiter code
export PATH="$PROJ/bin:$PATH"; export MAMBA_ROOT_PREFIX=$PROJ/micromamba
eval "$(micromamba shell hook -s bash)"; micromamba activate $PROJ/envs/slime
source $REPO/.env
SP="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export SLIME_ENV_LIB_PATHS_HEAD="$SP/cudnn/lib:$SP/nccl/lib:$SP/cuda_runtime/lib:$SP/cublas/lib"
export LD_LIBRARY_PATH="$SP/cudnn/lib:$SP/nccl/lib:$SP/cuda_runtime/lib:$SP/cublas/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:$PROJ/openwebrl_build/Megatron-LM"
export CUDA_HOME="$CONDA_PREFIX"

# flashinfer JIT build env (same as training; without it sglang dies at capture)
NVD="$SP"; _inc="$CONDA_PREFIX/targets/x86_64-linux/include"; _lib="$CONDA_PREFIX/targets/x86_64-linux/lib"
for d in "$NVD"/*/include; do [ -d "$d" ] && _inc="$_inc:$d"; done
for d in "$NVD"/*/lib;     do [ -d "$d" ] && _lib="$_lib:$d"; done
export CPATH="$_inc:${CPATH:-}"
_drv=""; for d in /usr/lib64 /lib64; do [ -e "$d/libcuda.so" ] && { _drv="$d"; break; }; done
export LIBRARY_PATH="$_lib:${_drv:-$CONDA_PREFIX/targets/x86_64-linux/lib/stubs}:${LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="$PROJ"

# node-local scratch: GPFS-resident chromium/triton/tmp killed dan29's runs
export TMPDIR=/tmp/owrl_$SLURM_JOB_ID;             mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=$TMPDIR/triton;       mkdir -p "$TRITON_CACHE_DIR"
if rsync -a "$PROJ/ms-playwright/" "$TMPDIR/ms-playwright/" 2>/dev/null; then
  export PLAYWRIGHT_BROWSERS_PATH=$TMPDIR/ms-playwright
else
  echo "WARN: playwright staging failed; using GPFS bundle" >&2
fi


# Browser subprocesses must not inherit the training LD_LIBRARY_PATH: with
# $CONDA_PREFIX/lib on it chromium pulls 35 libs off GPFS (~40s cold launch,
# 0/8 under 8-way concurrency). The wrapper clears it so all 60 resolve from
# node-local /lib64 -- verified 0 GPFS / 0 missing. Matches dan29, whose
# launcher exports only PATH.
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON=$PROJ/browser_python.sh

ARM="${ARM:?set ARM=greedy or ARM=nucleus}"
case "$ARM" in
  greedy)  TEMPERATURE=0.0; TOP_P=1.0; EXTRA="" ;;
  nucleus) TEMPERATURE=0.7; TOP_P=0.8; EXTRA="--sampling-backend pytorch" ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
# No $SLURM_JOB_ID here: the run directory must be stable across a requeue so
# --resume can find the work already done. Requeue keeps the same job id, but
# a fresh submission after a hard failure would not.
RUN_NAME="${RUN_NAME:-owrlsft_om2w_sel-${SELECTOR:-none}_${ARM}_ckptresume}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=$SLIME_OUTPUT_ROOT/$RUN_NAME/env_logs
mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR"

# Chromium launch is node-dependent on klone: identical env/config, g3075 ran
# fine while g3046 failed 8/8 with "BrowserType.launch: Timeout 180000ms".
# dan29 hit the same on g3067/g3072/g3075 and keeps a BAD node list. Fail in
# ~60s on a bad node instead of burning the whole allocation on aborts.
echo "[eval] chromium preflight on $(hostname)..."
# MUST go through browser_python.sh: it clears LD_LIBRARY_PATH exactly as the
# real env_server does. Running the preflight under the training
# LD_LIBRARY_PATH tests the slow 35-GPFS-lib path the fix removed, and
# false-negatives nodes that would run fine (g3121 failed then passed 4min
# later purely because that path is marginal).
if ! timeout 150 $PROJ/browser_python.sh - <<'PYEOF'
import asyncio, sys
from playwright.async_api import async_playwright
ARGS = ["--disable-gpu","--no-sandbox","--disable-dev-shm-usage","--disable-logging",
        "--ignore-certificate-errors","--log-level=3","--silent"]
# 8 concurrent, matching N_PARALLEL: a single launch passes on nodes where the
# real 8-way workload still fails (measured: n=1 ok at ~40s, n=8 0/8 at 150s).
async def one(p):
    b = await p.chromium.launch(headless=True, args=ARGS, timeout=120000)
    await b.close()
async def main():
    async with async_playwright() as p:
        await asyncio.gather(*[one(p) for _ in range(8)])
asyncio.run(main())
PYEOF
then
  # browser launch is a node lottery (3s..213s cold, measured). rather than die,
  # put the job back in the queue to land somewhere healthy. capped so a
  # cluster-wide problem does not loop forever.
  # Count OUR OWN chromium requeues, not SLURM_RESTART_COUNT. That counter is
  # bumped by every restart including preemption, so on a preemptible queue the
  # budget reserved for browser retries gets silently spent by the scheduler --
  # which is exactly how 39450598 declared chromium "unusable after 9 requeues"
  # and quit in 2m48s on a node that had passed preflight the run before.
  CHROME_TRIES_FILE="$PROJ/.chromium_retries_${SLURM_JOB_ID}"
  RC=$(cat "$CHROME_TRIES_FILE" 2>/dev/null || echo 0)
  if [ "$RC" -lt 5 ]; then
    echo "$((RC+1))" > "$CHROME_TRIES_FILE"
    echo "[eval] chromium unusable on $(hostname); requeueing (browser attempt $((RC+1))/5)"
    scontrol requeue "$SLURM_JOB_ID" && sleep 30 && exit 0
  fi
  echo "[eval] FATAL: chromium unusable after $RC browser requeues; giving up"
  rm -f "$CHROME_TRIES_FILE"
  exit 1
fi
rm -f "$PROJ/.chromium_retries_${SLURM_JOB_ID}"
echo "[eval] chromium preflight OK"

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ARM=$ARM TEMPERATURE=$TEMPERATURE TOP_P=$TOP_P EXTRA='$EXTRA' RUN=$RUN_NAME"


# ==== ACTION-SELECTION JUDGE (GPU 1) =========================================
# SELECTOR=arm    -> trained selection ARM  (PTeterwak/OpenWebRL-4B-SelectionARM)
# SELECTOR=zs     -> untrained zero-shot judge (stock OpenWebRL-4B-SFT)
# SELECTOR=scalar -> trained value-head RM  (PTeterwak/OpenWebRL-4B-ScalarRM-LoRA)
#                    (needs `pip install peft` in the slime env, once)
# The policy samples n=5 at temp 0.7 per step; the judge picks. Trajectory
# format/judging downstream is unchanged. Sidecars in $OWRL_ARBITER_LOG_DIR.
SELECTOR="${SELECTOR:?set SELECTOR=arm|zs|scalar}"
# derive both ports from the job id: two selector jobs can land on the same
# node (they did), and run_evaluation_local.sh defaults SGLANG_PORT to a
# fixed 9000 -- a collision there silently cross-wires the two evals.
JPORT=$((21000 + SLURM_JOB_ID % 900))
export SGLANG_PORT=$((9000 + SLURM_JOB_ID % 900))
echo "[sel] ports: policy=$SGLANG_PORT judge=$JPORT"
export OWRL_ARBITER=1
export OWRL_ARBITER_N="${OWRL_ARBITER_N:-5}"
export OWRL_ARBITER_TEMP="${OWRL_ARBITER_TEMP:-0.7}"
export OWRL_ARBITER_LOG_DIR="$SLIME_OUTPUT_ROOT/$RUN_NAME/arbiter_logs"
mkdir -p "$OWRL_ARBITER_LOG_DIR"
JLOG="$SLIME_OUTPUT_ROOT/$RUN_NAME/judge_server.log"
case "$SELECTOR" in
  arm|zs)
    if [ "$SELECTOR" = arm ]; then JUDGE_HF="${ARM_HF:-PTeterwak/OpenWebRL-4B-SelectionARM}";
    else JUDGE_HF="${ZS_HF:-/gscratch/cse/piotrt/models/OpenWebRL/OpenWebRL-4B-SFT}"; fi
    CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
      --model-path "$JUDGE_HF" --port "$JPORT" --host 127.0.0.1 \
      --mem-fraction-static 0.80 --context-length 32768 > "$JLOG" 2>&1 &
    JPID=$!
    export OWRL_ARBITER_URL="http://127.0.0.1:$JPORT/v1"
    export OWRL_ARBITER_API_KEY=EMPTY
    export OWRL_ARBITER_MODEL="$JUDGE_HF"
    export OWRL_ARBITER_COORDS=normalized
    HEALTH="http://127.0.0.1:$JPORT/health"
    ;;
  scalar)
    CUDA_VISIBLE_DEVICES=1 python "$REPO/klone/scalar_server.py" --port "$JPORT" > "$JLOG" 2>&1 &
    JPID=$!
    export OWRL_SCALAR_URL="http://127.0.0.1:$JPORT"
    HEALTH="http://127.0.0.1:$JPORT/health"
    ;;
  *) echo "unknown SELECTOR=$SELECTOR"; exit 1;;
esac
trap 'kill $JPID 2>/dev/null' EXIT
echo "[sel] judge ($SELECTOR) starting on GPU1 port $JPORT"
for _ in $(seq 1 120); do
  kill -0 $JPID 2>/dev/null || { echo "[sel] FATAL judge died:"; tail -30 "$JLOG"; exit 1; }
  curl -sf "$HEALTH" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "$HEALTH" >/dev/null || { echo "[sel] FATAL judge never healthy"; tail -30 "$JLOG"; exit 1; }
echo "[sel] judge healthy"
# arbiter steps are ~5-6x slower than n=1
export TASK_TIMEOUT_SECS_OVERRIDE=1
export RESUME=1
# =============================================================================

cd "$REPO"

# NOTE: TASK_TIMEOUT_SECS=9000 below must stay >= BROWSER_MAX_STEPS(30) *
# INFERENCE_STEP_TIMEOUT_SECS(120) = 3600, else the task-level timeout fires
# before the per-step ones and infrastructure slowness is miscounted as agent
# failure. Our run_evaluation_local.sh defaults it to 600. Do NOT put comments
# inside the backslash-continued env chain below -- it silently breaks it.
MODEL_PATH_CHECK="$SLIME_MODEL_ROOT/OpenWebRL/OpenWebRL-4B-SFT"
[ -f "$MODEL_PATH_CHECK/config.json" ] || { echo "FATAL: no model at $MODEL_PATH_CHECK"; exit 1; }
echo "[eval] model=$MODEL_PATH_CHECK"
MODEL_PATH="$SLIME_MODEL_ROOT/OpenWebRL/OpenWebRL-4B-SFT" \
TASK_FILE="openwebrl/data/online-mind2web.jsonl" \
OUTPUT_ROOT="$SLIME_OUTPUT_ROOT/$RUN_NAME" \
N_PARALLEL=8 \
SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=8 \
SGLANG_TP=1 SGLANG_DP=1 \
JUDGE_MODEL=gpt-4.1 \
JUDGE_API_MODE=served \
JUDGE_API_BASE=https://api.openai.com/v1 \
TEMPERATURE="$TEMPERATURE" \
TOP_P="$TOP_P" \
SGLANG_EXTRA_ARGS="$EXTRA" \
INFERENCE_STEP_TIMEOUT_SECS=120 \
TASK_TIMEOUT_SECS=9000 \
SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=300 \
SGLANG_STARTUP_TIMEOUT_SECS=1800 \
SGLANG_CONTEXT_LENGTH=65536 \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_evaluation_local.sh
echo "EVAL_EXIT=$?"
