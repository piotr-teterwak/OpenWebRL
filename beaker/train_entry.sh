#!/usr/bin/env bash
# ON-NODE entry for OpenWebRL RL training (runs inside the Beaker/Gantry container).
# Uses the weka-resident `webrl` conda env + from-source backend built under
# openwebrl_build/ (no install needed). Judge = GPT-4.1 via OPENAI_API_KEY (Beaker secret).
set -euo pipefail

REPO=/weka/oe-training-default/zixianm/xunmei/OpenWebRL
BUILD=/weka/oe-training-default/zixianm/openwebrl_build   # read-only: Megatron-LM source on PYTHONPATH
CONDA=/weka/oe-training-default/zixianm/.conda/envs/webrl
NV="$CONDA/lib/python3.12/site-packages/nvidia"
: "${OPENAI_API_KEY:?OPENAI_API_KEY (Beaker secret) required for the gpt-4.1 judge}"

export PATH="$CONDA/bin:$PATH"
# webrl/lib provides libnss/libnuma; nvidia cu12 dirs provide cudnn/nccl/cuda runtime.
export LD_LIBRARY_PATH="$CONDA/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
# Chromium cold-start off weka NFS was the Jul-16 blocker (318-347s/launch, 226s avg
# pool wait). Copy the browser bundle to node-local disk once per job (~4G, minutes),
# fall back to the weka path only if the copy fails.
if rsync -a /weka/oe-training-default/zixianm/.cache/ms-playwright/ /tmp/ms-playwright/ 2>/dev/null; then
  export PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright
  echo "[train_entry] playwright bundle copied to node-local /tmp/ms-playwright"
else
  export PLAYWRIGHT_BROWSERS_PATH=/weka/oe-training-default/zixianm/.cache/ms-playwright
  echo "[train_entry] WARN: local copy failed; using weka playwright path (slow cold-start)"
fi
export PYTHONPATH="$REPO:$BUILD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

# CUDA headers for flashinfer's RUNTIME JIT compile. On Blackwell/B200 (titan, sm_100a)
# sglang JIT-builds trtllm-gen decode kernels and nvcc needs nvrtc.h etc. on the include
# path (on H100 prebuilt kernels are used, so this is silent there). Mirror the build env.
export CUDA_HOME="$CONDA"
_cuda_inc="$CONDA/targets/x86_64-linux/include"; _cuda_lib="$CONDA/targets/x86_64-linux/lib"
for d in "$NV"/*/include; do case "$d" in *"/cu13/"*) continue;; esac; _cuda_inc="$_cuda_inc:$d"; done
for d in "$NV"/*/lib;     do case "$d" in *"/cu13/"*) continue;; esac; _cuda_lib="$_cuda_lib:$d"; done
export CPATH="$_cuda_inc:${CPATH:-}"
export LD_LIBRARY_PATH="$_cuda_lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$_cuda_lib:${LIBRARY_PATH:-}"   # link-time (-L) search for the flashinfer JIT c++ link step (libnvrtc etc.)

# Beaker base image lacks some Chromium system deps -> install once (idempotent).
if ! dpkg -s libgbm1 >/dev/null 2>&1; then
  echo "[train_entry] installing chromium system deps (one-time, container-local)"
  apt-get update -qq || true
  apt-get install -y --no-install-recommends \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libxcb-dri3-0 libatspi2.0-0 libnss3 libnspr4 libcups2 libatk1.0-0 \
    libatk-bridge2.0-0 libpango-1.0-0 libcairo2 libasound2 libpangocairo-1.0-0 \
    >/dev/null 2>&1 || echo "[train_entry] WARN: apt deps non-zero (LD_LIBRARY_PATH webrl/lib still covers libnss)"
fi

EXP_NAME="${EXP_NAME:-openwebrl_rl_$(date +%m%d_%H%M)}"
# wandb: log to Xunmei's entity (the inherited default lacked permissions).
# WANDB_API_KEY comes from the beaker secret wired in launch_train.sh; empty -> offline.
export WANDB_ENTITY="${WANDB_ENTITY:-xunmel-university-of-washington}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ---- paths the launcher reads ----
export SLIME_MODEL_ROOT="${SLIME_MODEL_ROOT:-/weka/oe-training-default/zixianm/OpenWebRL/models}"   # read-only: reuse her model copy, don't duplicate 9G
export SLIME_SAVE_ROOT="${SLIME_SAVE_ROOT:-$REPO/outputs/$EXP_NAME}"
# STABLE save dir keyed off EXP_NAME (not a per-attempt timestamp) so every beaker
# retry / preemption restart writes to & reads from the same checkpoints.
export SLIME_SAVE_DIR="${SLIME_SAVE_DIR:-$SLIME_SAVE_ROOT/ckpt}"
mkdir -p "$SLIME_SAVE_DIR"

# ---- auto-resume: if a checkpoint exists in the stable save dir, load the latest
# iteration so a restart continues instead of starting over at rollout 0. ----
if [ -f "$SLIME_SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  export SLIME_LOAD_CHECKPOINT="$SLIME_SAVE_DIR"
  export SLIME_CKPT_STEP="$(tr -dc '0-9' < "$SLIME_SAVE_DIR/latest_checkpointed_iteration.txt")"
  echo "[train_entry] RESUMING from $SLIME_SAVE_DIR @ iter ${SLIME_CKPT_STEP}"
else
  echo "[train_entry] fresh start (no checkpoint in $SLIME_SAVE_DIR)"
fi

# ---- judge: plain OpenAI key -> served mode against api.openai.com ----
export JUDGE_API_MODE=served
export JUDGE_API_BASE="https://api.openai.com/v1"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
export JUDGE_API_KEY="$OPENAI_API_KEY"

# ---- memory: dedicated beaker GPUs; TP splits the model, so 0.5 leaves room for the train actor ----
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"
# ---- reduce CUDA fragmentation on the train GPUs (disaggregated only): the disaggregated
# train step OOMs by a few hundred MiB with ~700MiB fragmented-free; expandable_segments
# recovers it. ONLY when disaggregated (ROLLOUT_NUM_GPUS set) — colocate uses
# torch_memory_saver, which expandable_segments can conflict with. ----
if [ -n "${ROLLOUT_NUM_GPUS:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
# Allocator-snapshot diagnostic, off by default. It was how we root-caused the TP1
# OOM (recompute args never reaching the bridge provider); leaving it on would record
# every allocation for the rest of the run for nothing, since the peak now sits at
# ~39GiB and never reaches the 45GiB dump threshold. Set to 1 to re-arm.
export SLIME_MEM_SNAPSHOT="${SLIME_MEM_SNAPSHOT:-0}"
# ---- local_process browser throughput: more concurrent env_server browsers for the full rollout ----
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-8}"   # 32 caused the 63-deep acquire queue on Jul 16; 8 is right for a 2-GPU smoke

SCRIPT="${SCRIPT:-scripts/run_browser_smoke_2gpu.sh}"
# use all GPUs the scheduler granted (gantry forwards GPUS_PER_NODE); script default is 4
export NUM_GPUS="${NUM_GPUS:-${GPUS_PER_NODE:-4}}"
cd "$REPO"
echo "[train_entry] EXP=$EXP_NAME script=$SCRIPT model_root=$SLIME_MODEL_ROOT save=$SLIME_SAVE_ROOT gpus=$(nvidia-smi -L | wc -l)"
exec bash "$SCRIPT" 2>&1 | tee -a "$SLIME_SAVE_ROOT/run.log"   # -a: keep prior attempts (a clobbered log cost us a crash diagnosis)
