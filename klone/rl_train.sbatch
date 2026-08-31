#!/bin/bash
#SBATCH --job-name=owrl-rl
#SBATCH --account=ckpt-krishna
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a100
#SBATCH --requeue
# slime's --actor-num-nodes 1 assumes both GPUs on ONE node. Without this,
# Slurm may satisfy --gpus=2 as 1 GPU on each of 2 nodes and ray start dies:
# 'Attempting to start raylet with 2 GPU, but CUDA_VISIBLE_DEVICES contains [0]'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=14
# 200G was OOM-killed (MaxRSS hit the cap exactly). Host RAM holds the
# cpu-offloaded Adam states (~48GB for 4B), the pinned rollout tensors
# (CHANGES.md fix #1), and the profiler's event buffers. l40s nodes have
# 1511GB/8 GPUs = 189GB per GPU, so 350G for 2 GPUs is within fair share.
#SBATCH --mem=350G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/cse/piotrt/logs/rl-%j.out
set -o pipefail
# conda activate.d hooks read these unguarded; nounset would abort here
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

export PROJ=/gscratch/cse/piotrt
export REPO=/gscratch/krishna/piotrt/openwebrl
export PATH="$PROJ/bin:$PATH"; export MAMBA_ROOT_PREFIX=$PROJ/micromamba
eval "$(micromamba shell hook -s bash)"; micromamba activate $PROJ/envs/slime
source $REPO/.env

# ---- klone equivalents of the weka paths baked into the launcher ----
export SLIME_MEGATRON_PATH="$PROJ/openwebrl_build/Megatron-LM"
SP="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
# pip nvidia dirs MUST precede $CONDA_PREFIX/lib. The env has two cuDNNs
# (conda 9.10.2 and pip 9.16.0.29); TransformerEngine dlopens pip's
# libcudnn_cnn.so.9 by absolute path, and if conda's lib dir comes first its
# libcudnn_graph dependency resolves to 9.10.2 -> undefined symbol
# _ZTVN5cudnn7backend20SdpaForwardOperationE at import.
export SLIME_ENV_LIB_PATHS="$SP/cudnn/lib:$SP/nccl/lib:$SP/cuda_runtime/lib:$SP/cublas/lib:$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$SLIME_ENV_LIB_PATHS:${LD_LIBRARY_PATH:-}"

# ---- flashinfer runtime-JIT build env ----
# set_up.sh uninstalls flashinfer-jit-cache, so sglang JIT-compiles attention
# kernels at cuda-graph capture. That compile needs CUDA headers on CPATH and
# CUDA libs on LIBRARY_PATH (link-time -L). Without them the link step fails
# with `cannot find -lcuda` and every SGLangEngine dies in its scheduler.
NVD="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
_lib="$CONDA_PREFIX/targets/x86_64-linux/lib"
for d in "$NVD"/*/include; do [ -d "$d" ] && _inc="$_inc:$d"; done
for d in "$NVD"/*/lib;     do [ -d "$d" ] && _lib="$_lib:$d"; done
export CPATH="$_inc:${CPATH:-}"

# -lcuda is the driver lib. Prefer the real one shipped by the node's driver;
# fall back to the CUDA stub (link-time only -- never put stubs on
# LD_LIBRARY_PATH or the stub gets loaded at runtime instead of the driver).
_drv=""
for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /lib64; do
  [ -e "$d/libcuda.so" ] && { _drv="$d"; break; }
done
if [ -n "$_drv" ]; then
  echo "[smoke] -lcuda from node driver: $_drv"
  export LIBRARY_PATH="$_lib:$_drv:${LIBRARY_PATH:-}"
else
  echo "[smoke] no driver libcuda.so; using CUDA stub for link step"
  export LIBRARY_PATH="$_lib:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
fi

# keep the JIT cache off $HOME (10GB quota) and persist it across jobs so the
# ~8min kernel compile happens once, not every run
export FLASHINFER_WORKSPACE_BASE="$PROJ"
mkdir -p "$PROJ/.cache/flashinfer"
echo "[smoke] flashinfer cache: $PROJ/.cache/flashinfer"
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONPATH="$REPO:$SLIME_MEGATRON_PATH"

# Chromium cold-start off GPFS was AI2's blocker (300s+/launch); stage on node-local /scr.
export TMPDIR=/scr/$SLURM_JOB_ID; mkdir -p "$TMPDIR"
if rsync -a "$PROJ/ms-playwright/" "$TMPDIR/ms-playwright/" 2>/dev/null; then
  export PLAYWRIGHT_BROWSERS_PATH="$TMPDIR/ms-playwright"
  echo "[smoke] playwright staged on node-local /scr"
else
  echo "[smoke] WARN: staging failed, using GPFS playwright path (slow cold start)"
fi

# Ray/sglang scratch on node-local disk, not GPFS
export RAY_TMPDIR=$TMPDIR/ray; mkdir -p "$RAY_TMPDIR"
# env_server stderr must survive the job: node-local /scr is unreadable once
# the allocation ends, and this is the log that explains rollout failures.
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=$SLIME_SAVE_ROOT/browser_env_logs_${SLURM_JOB_ID}
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=$TMPDIR/browser_ports
mkdir -p "$SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR" "$SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR"

export MASTER_ADDR=127.0.0.1
# STABLE save dir (no job id!) so ckpt-partition requeues resume the same run
RUN_NAME="${RUN_NAME:-owrl4b_rl_prod}"
export SLIME_SAVE_DIR="$SLIME_SAVE_ROOT/$RUN_NAME"
mkdir -p "$SLIME_SAVE_DIR"

# 80GB A100s, not 180GB B200s -> leave more headroom than the script's default.
# The launcher default 0.6 was tuned for 180GB B200; I had dropped it to 0.45
# out of caution, but on an 80GB A100 that just starves the KV cache and slows
# generation. 0.6 => ~48GB KV + ~8GB weights, comfortable in 80GB, and sglang
# offloads during the training step anyway (--colocate).
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"
# 2 prompts x 4 samples = 8 episodes; default cap of 16 would oversubscribe 14 cores
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=8

# env_server startup on klone: measured 87s warm / ~460s cold for a single
# server with no contention. The launcher default is 60s, so every episode
# aborted with "env_server was not healthy within 60.0s". The cost is not
# env_server itself (6.3s) - importing openwebrl pulls torch (115s),
# transformers (105s) and slime.rollout.sglang_rollout (50s) off GPFS.
# AI2 ran this off weka where 60s sufficed.
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=600
export SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS=1800
# episode budget must cover that startup plus the actual browsing
export SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS=2400
export KILL_STALE_SGLANG=1

# ---- per-job Ray ports (klone nodes are SHARED between users) ----
# Upstream hardcodes 127.0.0.1:8265; on a shared node that can be another
# user's Ray head, and `ray job submit` then submits OUR job into THEIR cluster
# (observed on g3083, job 39049971). Ray also needs unique agent/manager ports
# or the dashboard agent fails to bind and job submit returns
# "No available agent to submit job" (job 39050803).
# Scan for a genuinely free contiguous block rather than hashing the job id.
find_free_base() {
  local base
  for base in $(seq 20000 250 60000); do
    local busy=0 prt
    for prt in $(seq $base $((base+11))); do
      ss -ltn 2>/dev/null | grep -q ":${prt} " && { busy=1; break; }
    done
    [ $busy -eq 0 ] && { echo $base; return 0; }
  done
  return 1
}
RAY_BASE=$(find_free_base) || { echo "[tuned] FATAL: no free port block on $(hostname)"; exit 1; }
export RAY_DASHBOARD_PORT=$((RAY_BASE))
export RAY_GCS_PORT=$((RAY_BASE+1))
export RAY_AGENT_LISTEN_PORT=$((RAY_BASE+2))
export RAY_AGENT_GRPC_PORT=$((RAY_BASE+3))
export RAY_NODE_MANAGER_PORT=$((RAY_BASE+4))
export RAY_OBJECT_MANAGER_PORT=$((RAY_BASE+5))
export RAY_RUNTIME_ENV_AGENT_PORT=$((RAY_BASE+6))
export RAY_METRICS_EXPORT_PORT=$((RAY_BASE+7))
export RAY_MIN_WORKER_PORT=$((RAY_BASE+20))
export RAY_MAX_WORKER_PORT=$((RAY_BASE+220))
echo "[tuned] ray port block ${RAY_BASE}-$((RAY_BASE+220)) on $(hostname)"


# ---- production settings (memory savers stay ON: OOM postmortem 39051888) ----
export RECOMPUTE=1
export OPTIMIZER_CPU_OFFLOAD=1
echo "[rl] recompute=on cpu_offload=on sglang_mem=$SGLANG_MEM_FRACTION_STATIC"

# ---- auto-resume: latest iter_* in SAVE_DIR (requeue/preemption-safe) ----
if [ -z "${SLIME_LOAD_CHECKPOINT:-}" ]; then
  latest=$(ls -d "$SLIME_SAVE_DIR"/iter_* 2>/dev/null | sed 's/.*iter_0*//' | sort -n | tail -1)
  if [ -n "$latest" ]; then
    export SLIME_LOAD_CHECKPOINT="$SLIME_SAVE_DIR"
    export SLIME_CKPT_STEP="$latest"
    echo "[rl] RESUMING from iter $latest"
  else
    echo "[rl] fresh run (no checkpoints in $SLIME_SAVE_DIR)"
  fi
fi

# ---- chromium hardening (same as eval_om2w.sbatch) ----
# browser subprocesses must not inherit the training LD_LIBRARY_PATH:
# with $CONDA_PREFIX/lib on it chromium pulls 35 libs off GPFS.
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON=$PROJ/browser_python.sh
echo "[stage] chromium preflight (8-way) on $(hostname)..."
if ! timeout 200 $PROJ/browser_python.sh - <<'PYEOF'
import asyncio
from playwright.async_api import async_playwright
ARGS=["--disable-gpu","--no-sandbox","--disable-dev-shm-usage","--disable-logging",
      "--ignore-certificate-errors","--log-level=3","--silent"]
async def one(p):
    b=await p.chromium.launch(headless=True,args=ARGS,timeout=120000); await b.close()
async def main():
    async with async_playwright() as p:
        await asyncio.gather(*[one(p) for _ in range(8)])
asyncio.run(main())
PYEOF
then echo "[stage] FATAL: chromium cannot launch on $(hostname); resubmit"; exit 1; fi
echo "[stage] chromium preflight OK"

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "SAVE_DIR=$SLIME_SAVE_DIR"; echo "JUDGE=$JUDGE_MODEL mode=$JUDGE_API_MODE"

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "SAVE_DIR=$SLIME_SAVE_DIR resume=${SLIME_CKPT_STEP:-none}"

cd "$REPO"
# fail-fast sentinel (from SCC bringup: dead engines look like activity).
# grep -c exits 1 on zero matches -- || true is LOAD-BEARING under pipefail.
LIVE="$SLIME_SAVE_DIR/train_${SLURM_JOB_ID}.log"
bash scripts/run_browser_realrun_2gpu.sh > "$LIVE" 2>&1 &
LPID=$!
for i in $(seq 1 60); do
  sleep 30
  kill -0 $LPID 2>/dev/null || break
  ok=$(grep -cE "accepted_groups=[1-9]" "$LIVE" 2>/dev/null || true)
  [ "${ok:-0}" -ge 1 ] && { echo "[sentinel] healthy (groups accepted)"; break; }
  e503=$(grep -c "503 Service Unavailable" "$LIVE" 2>/dev/null || true)
  fails=$(grep -c "generate_turn_sample failed" "$LIVE" 2>/dev/null || true)
  if [ "${e503:-0}" -gt 30 ] || [ "${fails:-0}" -gt 300 ]; then
    echo "[sentinel] FATAL dead-engine signature (503s=$e503 fails=$fails)"; tail -40 "$LIVE"
    kill $LPID 2>/dev/null; exit 1
  fi
done
tail -f "$LIVE" --pid=$LPID &
wait $LPID
echo "RL_EXIT=$?"
