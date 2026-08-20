#!/usr/bin/env bash
# Launch OpenWebRL RL training on Beaker via Gantry.
# Code + webrl conda env + from-source backend live on weka (mounted), so we run
# the on-node entry directly — no git-clone / image build. Override any VAR via env:
#   EXP_NAME=my_run GPUS=2 PRIORITY=high bash beaker/launch_train.sh
set -euo pipefail

# Site-specific identifiers (budget / workspace / clusters) live in an
# untracked beaker/site.env; copy beaker/site.env.example to create one.
# Environment variables still take precedence over anything set there.
_SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/site.env"
[ -f "$_SITE" ] && . "$_SITE"

BUDGET=${BUDGET:-}
WORKSPACE=${WORKSPACE:-}
CLUSTER=${CLUSTER:-}
CLUSTER2=${CLUSTER2:-}
CLUSTER3=${CLUSTER3:-}
CLUSTER4=${CLUSTER4:-}   # titan (B200) is reserved for other users -> unschedulable; opt in explicitly
PRIORITY=${PRIORITY:-high}
IMAGE=${IMAGE:-sanghol/molmo2-torch2.7.1-cuda12.8}   # CUDA12.8 base; entry uses the weka webrl env
GPUS=${GPUS:-4}                                       # default 4 GPUs (full run: TP=2, DP=2)
EXP_NAME=${EXP_NAME:-rl_$(date +%m%d_%H%M)}
SCRIPT=${SCRIPT:-scripts/run_browser_full.sh}
ENTRY=/weka/oe-training-default/zixianm/xunmei/OpenWebRL/beaker/train_entry.sh
NAME="${NAME:-run-$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')}"
DESCRIPTION="${DESCRIPTION:-batch run}"

# Preemptible jobs may NOT set --min-runtime (gantry rejects the combo). Preemptible
# is required to backfill onto unallocated capacity (e.g. titan) where this budget
# has no guaranteed allocation.
if [ -n "${PREEMPTIBLE:-}" ]; then
  SCHED_ARGS=(--preemptible)
else
  SCHED_ARGS=(--min-runtime "${MIN_RUNTIME:-1h}")
fi

NAME_ARGS=()
if [ -z "${IMPLICIT_NAME:-}" ]; then NAME_ARGS=(--name "$NAME" --task-name "$NAME"); fi
# The code that actually runs comes from weka, not from a git clone, so the ref
# gantry records is metadata. Our own commits live only in a private fork, so pin
# the ref to the last commit shared with origin -- otherwise gantry refuses to
# launch with "does not appear to exist on the remote".
REF="${REF:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." merge-base HEAD origin/main 2>/dev/null || true)}"

gantry run \
  --budget "$BUDGET" --workspace "$WORKSPACE" \
  "${NAME_ARGS[@]}" --description "$DESCRIPTION" \
  --cluster "$CLUSTER" --cluster "$CLUSTER2" --cluster "$CLUSTER3" ${CLUSTER4:+--cluster "$CLUSTER4"} \
  --gpus "$GPUS" --priority "$PRIORITY" --retries "${RETRIES:-3}" --allow-dirty ${REF:+--ref "$REF"} \
  "${SCHED_ARGS[@]}" \
  --beaker-image "$IMAGE" \
  --weka oe-training-default:/weka/oe-training-default \
  --shared-memory 128GiB \
  ${MEMORY:+--memory "${MEMORY}"} \
  --env-secret OPENAI_API_KEY=OPENAI_API_KEY_ZM \
  --env-secret WANDB_API_KEY="${WANDB_API_KEY_SECRET:-WANDB_API_KEY_XL}" \
  --env EXP_NAME="$EXP_NAME" \
  --env SCRIPT="$SCRIPT" \
  --env GPUS_PER_NODE="$GPUS" \
  --env WANDB_PROJECT="${WANDB_PROJECT:-verl_agent_webgym}" --env WANDB_ENTITY="${WANDB_ENTITY:-xunmel-university-of-washington}" \
  --env JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}" \
  --env SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}" \
  ${ROLLOUT_MAX_CONTEXT_LEN:+--env ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN}"} \
  ${SGLANG_ATTENTION_BACKEND:+--env SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND}"} \
  ${NUM_ROLLOUT:+--env NUM_ROLLOUT="${NUM_ROLLOUT}"} \
  ${SAVE_INTERVAL:+--env SAVE_INTERVAL="${SAVE_INTERVAL}"} \
  ${TP:+--env TP="${TP}"} \
  ${ACTOR_GPUS:+--env ACTOR_GPUS="${ACTOR_GPUS}"} \
  ${ROLLOUT_NUM_GPUS:+--env ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS}"} \
  ${SLIME_SAVE_DIR:+--env SLIME_SAVE_DIR="${SLIME_SAVE_DIR}"} \
  ${BROWSER_MAX_STEPS:+--env BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS}"} \
  ${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:+--env SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}"} \
  ${TRAIN_DATA:+--env TRAIN_DATA="${TRAIN_DATA}"} \
  ${CONVERT_INPUT:+--env CONVERT_INPUT="${CONVERT_INPUT}"} \
  ${CONVERT_OUTPUT:+--env CONVERT_OUTPUT="${CONVERT_OUTPUT}"} \
  ${EVAL_ITERS:+--env EVAL_ITERS="${EVAL_ITERS}"} \
  ${TASK_FILE:+--env TASK_FILE="${TASK_FILE}"} \
  ${OUTPUT_ROOT:+--env OUTPUT_ROOT="${OUTPUT_ROOT}"} \
  --no-python \
  -- /bin/bash "$ENTRY"
