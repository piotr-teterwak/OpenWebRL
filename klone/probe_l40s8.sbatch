#!/bin/bash
#SBATCH --job-name=owrl-l40s8
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=850G      # friend's working 8xL40S value
#SBATCH --time=10:00:00
#SBATCH --output=/gscratch/cse/piotrt/logs/l40s8-%j.out
# Throughput probe: identical workload on each queue's best configuration.
# One rollout iteration at reduced scale, then the training phases. What we want
# out of it is the perf dict (rollout_time / log_probs_time / actor_train_time /
# tokens_per_gpu_per_sec) so the queues can be compared on equal work.
# slime's --actor-num-nodes 1 assumes both GPUs on ONE node. Without this,
# Slurm may satisfy --gpus=2 as 1 GPU on each of 2 nodes and ray start dies:
# 'Attempting to start raylet with 2 GPU, but CUDA_VISIBLE_DEVICES contains [0]'
# 200G was OOM-killed (MaxRSS hit the cap exactly). Host RAM holds the
# cpu-offloaded Adam states (~48GB for 4B), the pinned rollout tensors
# (CHANGES.md fix #1), and the profiler's event buffers. l40s nodes have
# 1511GB/8 GPUs = 189GB per GPU, so 350G for 2 GPUs is within fair share.
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
export TMPDIR=/tmp/owrl_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
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
export SLIME_SAVE_DIR="$SLIME_SAVE_ROOT/phase_${SLURM_JOB_ID}"
mkdir -p "$SLIME_SAVE_DIR"

# 80GB A100s, not 180GB B200s -> leave more headroom than the script's default.
# The launcher default 0.6 was tuned for 180GB B200; I had dropped it to 0.45
# out of caution, but on an 80GB A100 that just starves the KV cache and slows
# generation. 0.6 => ~48GB KV + ~8GB weights, comfortable in 80GB, and sglang
# offloads during the training step anyway (--colocate).
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"   # authors value
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

# ---- PHASE PROFILING ----
# slime's --profile-target train_actor/train_log_probs are dead code: nothing in
# actor.py calls TrainProfiler.iterate_*. Instead model.py now carries an
# env-gated one-shot torch profile on forward_only (the forward-only log-prob
# pass) and train_one_step (forward+backward). Each prints a key_averages table
# straight into this log, so no trace parsing is needed.
mkdir -p "$SLIME_PHASE_PROFILE_DIR"
echo "[phase] profile dir: $SLIME_PHASE_PROFILE_DIR"

# ---- PROFILER RUN ----
# Question: what makes a turn sample cost 40.6s when the arithmetic says ~3s?
# log_probs (fwd only) costs 86% of actor_train (fwd+bwd), so the cost is paid
# per pass regardless of gradients. Suspects: H2D copy of pixel_values
# (CHANGES.md fix #1 keeps them pinned on CPU and moves per micro-batch), or a
# frozen vision tower whose forward runs in both passes.
# Profile both phases separately; 4 steps is plenty and keeps the trace small.

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

# ---- THE EXPERIMENT ----
# Baseline (job 38896851) was 82.3s per turn sample, with log_probs (forward
# only) costing the same as actor_train (fwd+bwd) => overhead-bound, not
# FLOP-bound, on a node with NVLINK_COUNT=0. Change all four suspected causes
# at once; if this moves the number, attribute afterwards.
export TP=2                  # friend's working value; DP=4 on 8 GPUs
unset SEQUENCE_PARALLEL           # SP requires TP>1
export MICRO_BATCH_SIZE=1         # authors' value
# global_batch_size stays at the baseline 8 -- it IS a hyperparameter
# (gradient noise / effective LR) and must not move in a throughput test.
# 8 % (MBS 4 x DP 2) == 0 already, so no divisibility problem existed.
export GLOBAL_BATCH_SIZE=256      # authors' value
# OOM POSTMORTEM (job 39051888): dropping cpu-offload at TP=1 is what blew
# memory. Adam fp32 moments for 4B params ~32GB; at TP=2 they were sharded
# 16GB/GPU, at TP=1 each GPU carries all 32GB. SGLang had correctly offloaded
# (held only 1.99GB), so all ~78GB was Megatron. Keep both memory savers on and
# still test the two big levers (TP=1 removes PCIe collectives, MBS=4 amortises
# per-micro-batch overhead).
export RECOMPUTE=1                # keep: activation recompute
export OPTIMIZER_CPU_OFFLOAD=1    # keep: Adam states stay on CPU
echo "[tuned] TP=$TP MBS=$MICRO_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE recompute=${RECOMPUTE:-off} cpu_offload=${OPTIMIZER_CPU_OFFLOAD:-off} sglang_mem=$SGLANG_MEM_FRACTION_STATIC"


# ---- chromium hardening (same as eval_om2w.sbatch) ----
# browser subprocesses must not inherit the training LD_LIBRARY_PATH:
# with $CONDA_PREFIX/lib on it chromium pulls 35 libs off GPFS.
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON=$PROJ/browser_python.sh
if ! "$PROJ/chromium_preflight.sh" 16; then
  # Browser launch time varies ~45x across nodes (3s..138s cold). A slow node
  # aborts every rollout task at env_server's 180s /reset cap, so the job is
  # worthless there -- put it back in the queue to land somewhere else.
  RC=${SLURM_RESTART_COUNT:-0}
  if [ "$RC" -lt 5 ]; then
    echo "[stage] chromium unusable on $(hostname); requeueing (attempt $((RC+1))/5)"
    scontrol requeue "$SLURM_JOB_ID" && sleep 30 && exit 0
  fi
  echo "[stage] FATAL: chromium unusable after $RC requeues; giving up"; exit 1
fi

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "SAVE_DIR=$SLIME_SAVE_DIR"; echo "JUDGE=$JUDGE_MODEL mode=$JUDGE_API_MODE"

# ---- continuous logging ----
# Python line-buffers to a TTY but block-buffers to a pipe, and sbatch stdout is
# a pipe: an OOM-kill or segfault then discards minutes of output. The 200G and
# 240G runs died with no record of the RAM trajectory, so we could not tell which
# phase was growing.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1   # C-level traceback on segfault/abort
export RAY_DEDUP_LOGS=0       # ray suppresses repeated worker lines by default

# Heartbeat. memory.current is the number Slurm OOM-kills on, so read the cgroup
# when it is available and fall back to summing our RSS.
_jobmem() {
  local rel f
  rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
  for f in "/sys/fs/cgroup${rel}/memory.current" \
           "/sys/fs/cgroup/memory${rel}/memory.usage_in_bytes"; do
    [ -r "$f" ] && { awk '{printf "%.1f", $1/1073741824}' "$f"; return; }
  done
  ps -u "$USER" -o rss= --no-headers 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1048576}'
}
_heartbeat() {
  while true; do
    echo "[hb $(date +%H:%M:%S)] job_mem=$(_jobmem)GB/${SLURM_MEM_PER_NODE:-?}M gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | grep -E "^[0-9]" | paste -sd" @ " || echo n/a)"
    sleep 20
  done
}
_heartbeat & _HB_PID=$!

# Ray actors write to per-worker files, not the driver stream, so the Megatron
# actors' output never reaches this log. /scr is wiped at job end -- archive it.
_archive() {
  kill "$_HB_PID" 2>/dev/null
  mkdir -p "$SLIME_SAVE_DIR/raylogs"
  # ray nests its session dir under an extra ray/ level; find is layout-agnostic
  find "$RAY_TMPDIR" -maxdepth 4 -type d -name logs -exec cp -r {} "$SLIME_SAVE_DIR/raylogs/" \; 2>/dev/null
  echo "[stage] ray logs archived to $SLIME_SAVE_DIR/raylogs"
}
trap _archive EXIT

# ---- authors' reference configuration (run_browser_Qwen3VL_4B_Instruct.sh) ----
export NUM_GPUS=4
export ACTOR_GPUS=4
export ROLLOUT_BATCH_SIZE=48
export N_SAMPLES_PER_PROMPT=5
export ROLLOUT_MAX_CONTEXT_LEN=32768
export PPO_EPOCHS=2
export BROWSER_MAX_STEPS=30
# NOT the authors' 100: enough rollouts to measure step time, not to train.
export NUM_ROLLOUT=${NUM_ROLLOUT:-2}
echo "[full8] NUM_GPUS=$NUM_GPUS TP=$TP MBS=$MICRO_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE rollout=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT} ctx=$ROLLOUT_MAX_CONTEXT_LEN ppo_epochs=$PPO_EPOCHS num_rollout=$NUM_ROLLOUT"

cd "$REPO"

# ---- identical workload on every queue ----
export NUM_GPUS=8
export ACTOR_GPUS=8
export TP=2                  # friend's working value; DP=4 on 8 GPUs
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=64
export ROLLOUT_BATCH_SIZE=16
export N_SAMPLES_PER_PROMPT=5
export ROLLOUT_MAX_CONTEXT_LEN=32768
export PPO_EPOCHS=1
export BROWSER_MAX_STEPS=30
export NUM_ROLLOUT=1
# browser + inference concurrency sized to this queue's CPU budget (~2 cpu/browser)
export SGLANG_SERVER_CONCURRENCY=96
export SGLANG_MAX_RUNNING_REQUESTS=96
export SLIME_BROWSER_ROLLOUT_CONCURRENCY=96   # friend runs 96 slots on 8xL40S
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES=96
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=96
echo "[l40s8] 8xL40S TP=$TP cpus=128 browsers=96 playwright=$PLAYWRIGHT_BROWSERS_PATH"
echo "[probe] nproc=$(nproc)"

cd "$REPO"
stdbuf -oL -eL bash scripts/run_browser_full.sh
echo "PROBE_EXIT=$?"
