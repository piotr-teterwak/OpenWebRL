#!/bin/bash
#SBATCH --job-name=owrl-train
# l40s, not a40: this eval is CPU-bound on 8 concurrent chromiums, and l40s
# nodes have 128 CPUs/8 GPUs (16 per GPU) vs a40's 52/8 (6.5 per GPU), so 16
# cpus-per-task is fair share here instead of 2.5x oversubscribed. Same 48GB,
# ample for a 4B model + KV. dan29 uses a40 only because it works once
# chromium launches fast; nothing about the workload requires it.
# NOT ckpt: scalar needs ~9h (measured 99s/task x 300) and preemption killed
# the one attempt that was working (39447296, clean at 26/300). zs completed
# here in 3:40 on the same 2-GPU shape.
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Right-sized from measured use, not guessed: the completed zs eval 39429421
# burned 19.44 CPU-h over 3.67 h on 20 allocated CPUs -- a mean of 5.3 busy
# cores, 27% utilisation. 12 leaves ~2x headroom over that mean and matches
# the a40/a100 node shape (6.5 CPU per GPU); l40s is 16 and had room to spare.
#SBATCH --cpus-per-task=12
#SBATCH --gpus=2
#SBATCH --mem=180G
# gpu-l40s has MaxTime=UNLIMITED. 12h was fine for arm (36 s/task) and zs
# (44 s/task) but NOT for scalar: its judge is an HF eager PeftModel rather
# than an sglang server, measured at ~185 s/task, i.e. ~15h for 300 tasks.
# A running job's limit cannot be raised without operator rights, so set it
# generously up front -- an over-long limit costs nothing here, a short one
# costs the whole run.
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/cse/piotrt/logs/train-%j.out
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

# ---------------------------------------------------------------------------
# Joint-data offline training on klone: filtered SFT or DPO over the same
# chosen/rejected pairs. OBJECTIVE=sft|dpo|dpo_action.
#
# SFT and DPO differ ONLY in the objective -- identical data, order, batch,
# updates and seed -- which is what makes that pair the one clean contrast in
# the offline table. Everything else comes from frozen_config() in
# train_arm_joint_ddp.py, so hyperparameters match the runs it reports.
#
# Paths: train_arm_joint_sft.py defaults to tillicum; the OWRL_* env vars below
# repoint it at klone without changing those defaults.
# ---------------------------------------------------------------------------
OBJECTIVE="${OBJECTIVE:?set OBJECTIVE=sft|dpo|dpo_action}"
case "$OBJECTIVE" in sft|dpo|dpo_action) ;; *) echo "bad OBJECTIVE=$OBJECTIVE"; exit 1;; esac

export OWRL_ACTOR="${OWRL_ACTOR:-$SLIME_MODEL_ROOT/OpenWebRL/OpenWebRL-4B-SFT}"
export OWRL_JOINT_DATA="${OWRL_JOINT_DATA:?set OWRL_JOINT_DATA to the built joint dataset}"
export OWRL_RUNTIME="${OWRL_RUNTIME:-$SLIME_OUTPUT_ROOT/owrl_runtime}"
mkdir -p "$OWRL_RUNTIME/runs"

[ -d "$OWRL_JOINT_DATA" ] || { echo "[train] FATAL joint data not found: $OWRL_JOINT_DATA"; exit 1; }
echo "[train] objective=$OBJECTIVE"
echo "[train] actor=$OWRL_ACTOR"
echo "[train] data=$OWRL_JOINT_DATA"
echo "[train] runtime=$OWRL_RUNTIME"

# Leave a wall-clock margin: --stop-unix tells the trainer when to checkpoint
# and exit cleanly rather than being killed mid-write.
MARGIN_S="${MARGIN_S:-900}"
LIMIT_S=$(( $(squeue -h -j "$SLURM_JOB_ID" -o "%l" | awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}') ))
STOP_UNIX=$(python -c "import time;print(time.time()+${LIMIT_S:-43200}-${MARGIN_S})")
echo "[train] stop-unix=$STOP_UNIX (margin ${MARGIN_S}s)"

cd "$REPO"
python scripts/run_arm_joint_pipeline.py --objective "$OBJECTIVE" --prepare \
  || { echo "[train] FATAL prepare failed"; exit 1; }

python scripts/train_arm_joint_ddp.py \
  --config "$OWRL_RUNTIME/runs/joint-v2-${OBJECTIVE}-2gpu/config.json" \
  --stop-unix "$STOP_UNIX" ${TRAIN_RESUME:+--resume} \
  || { echo "[train] FATAL training failed"; exit 1; }

echo "[train] DONE -> $OWRL_RUNTIME/runs/joint-v2-${OBJECTIVE}-2gpu"
