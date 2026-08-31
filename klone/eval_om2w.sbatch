#!/bin/bash
#SBATCH --job-name=owrl-eval
# l40s, not a40: this eval is CPU-bound on 8 concurrent chromiums, and l40s
# nodes have 128 CPUs/8 GPUs (16 per GPU) vs a40's 52/8 (6.5 per GPU), so 16
# cpus-per-task is fair share here instead of 2.5x oversubscribed. Same 48GB,
# ample for a 4B model + KV. dan29 uses a40 only because it works once
# chromium launches fast; nothing about the workload requires it.
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
export REPO=/gscratch/krishna/piotrt/openwebrl
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
export TMPDIR=/scr/$SLURM_JOB_ID;             mkdir -p "$TMPDIR"
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
RUN_NAME="${RUN_NAME:-owrlsft_om2w_${ARM}_${SLURM_JOB_ID}}"
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
  echo "[eval] FATAL: chromium cannot launch on $(hostname); resubmit to land elsewhere"
  exit 1
fi
echo "[eval] chromium preflight OK"

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ARM=$ARM TEMPERATURE=$TEMPERATURE TOP_P=$TOP_P EXTRA='$EXTRA' RUN=$RUN_NAME"

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
bash scripts/run_evaluation_local.sh
echo "EVAL_EXIT=$?"
