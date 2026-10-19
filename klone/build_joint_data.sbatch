#!/bin/bash
#SBATCH --job-name=owrl-joint
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
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
# gpu-l40s has MaxTime=UNLIMITED. 12h was fine for arm (36 s/task) and zs
# (44 s/task) but NOT for scalar: its judge is an HF eager PeftModel rather
# than an sglang server, measured at ~185 s/task, i.e. ~15h for 300 tasks.
# A running job's limit cannot be raised without operator rights, so set it
# generously up front -- an over-long limit costs nothing here, a short one
# costs the whole run.
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/cse/piotrt/logs/joint-%j.out
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
# Build the joint dataset = C2 (self-generated, ARM-selected) + Piotr (GPT-5.5
# labelled) into the chosen/rejected pairs that joint SFT and joint DPO share.
#
# This is the step between collection and training. prepare_arm_joint_data.py
# does the pairing, dedup, ambiguity quarantine and task-group split described
# in openwebrl/docs/ARM_JOINT_DATA.md; its own report is the authority on the
# final counts, not the numbers in that doc.
#
# No GPU, no torch, no model load -- it uses the Rust tokenizer and image
# headers only, hence gpus=0.
#
# NOTE on the upstream guardrails: the script sets RLIMIT_AS to 1 GiB and
# RLIMIT_CPU to 240s itself. Those were sized on tillicum; GPFS here is slower,
# so a CPU-time abort is the most likely first failure. If it dies on SIGXCPU,
# raise the limit in the script rather than papering over it in the sbatch --
# the limit is a deliberate bound, and changing it should be a visible edit.
# ---------------------------------------------------------------------------
C2_DIR="${C2_DIR:?set C2_DIR to the output of klone/c2_collect.sbatch}"
HF_DIR="${HF_DIR:-/gscratch/scrubbed/piotrt/arm_data}"
OUT_DIR="${OUT_DIR:-$SLIME_OUTPUT_ROOT/joint_data_$SLURM_JOB_ID}"
MODEL="${MODEL:-$SLIME_MODEL_ROOT/OpenWebRL/OpenWebRL-4B-SFT}"

for d in "$C2_DIR" "$HF_DIR" "$MODEL"; do
  [ -e "$d" ] || { echo "[joint] FATAL missing: $d"; exit 1; }
done
[ -f "$HF_DIR/openwebrl_actor/states_full.jsonl" ] || {
  echo "[joint] FATAL $HF_DIR does not look like the PTeterwak release"; exit 1; }

echo "[joint] c2     = $C2_DIR"
echo "[joint] hf     = $HF_DIR"
echo "[joint] model  = $MODEL"
echo "[joint] output = $OUT_DIR"

cd "$REPO"
python scripts/prepare_arm_joint_data.py \
  --hf "$HF_DIR" --c2 "$C2_DIR" --output "$OUT_DIR" \
  --model "$MODEL" \
  --eval openwebrl/data/eval/online-mind2web.jsonl \
  ${JOINT_RESUME:+--resume} \
  || { echo "[joint] FATAL prepare failed (SIGXCPU => raise RLIMIT_CPU in the script)"; exit 1; }

echo "[joint] verifying"
python scripts/verify_arm_joint_data.py --data "$OUT_DIR" \
  || { echo "[joint] FATAL verification failed"; exit 1; }

echo "[joint] DONE -> $OUT_DIR"
echo "[joint] next: OBJECTIVE=sft OWRL_JOINT_DATA=$OUT_DIR sbatch klone/train_joint.sbatch"
