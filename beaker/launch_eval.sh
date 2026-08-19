#!/usr/bin/env bash
# Launch OpenWebRL eval on Beaker via Gantry. Evaluates training checkpoints
# "every 10 iters" by default. Override knobs via env, e.g.:
#   EVAL_ITERS="9 19 29" N_EVAL_TASKS=30 GPUS=1 bash beaker/launch_eval.sh
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
CLUSTER4=${CLUSTER4:-}
PRIORITY=${PRIORITY:-high}
IMAGE=${IMAGE:-sanghol/molmo2-torch2.7.1-cuda12.8}
GPUS=${GPUS:-1}                       # 4B serve fits on 1 GPU (tp=1)
EXP_NAME=${EXP_NAME:-eval_$(date +%m%d_%H%M)}
ENTRY=/weka/oe-training-default/zixianm/OpenWebRL/beaker/eval_entry.sh
NAME="openwebrl-eval-${EXP_NAME}"

if [ -n "${PREEMPTIBLE:-}" ]; then SCHED=(--preemptible); else SCHED=(--min-runtime "${MIN_RUNTIME:-1h}"); fi

gantry run \
  --budget "$BUDGET" --workspace "$WORKSPACE" \
  --name "$NAME" --task-name "$NAME" --description "openwebrl eval ${EXP_NAME}" \
  --cluster "$CLUSTER" --cluster "$CLUSTER2" --cluster "$CLUSTER3" --cluster "$CLUSTER4" \
  --gpus "$GPUS" --priority "$PRIORITY" --retries "${RETRIES:-1}" --allow-dirty \
  "${SCHED[@]}" \
  --beaker-image "$IMAGE" \
  --weka oe-training-default:/weka/oe-training-default \
  --shared-memory 128GiB \
  --env-secret OPENAI_API_KEY=OPENAI_API_KEY_ZM \
  --env-secret GOOGLE_API_KEY=GOOGLE_API_KEY_ZM \
  --env EXP_NAME="$EXP_NAME" \
  ${JUDGES:+--env JUDGES="${JUDGES}"} \
  ${GEMINI_MODEL:+--env GEMINI_MODEL="${GEMINI_MODEL}"} \
  ${CKPT_RUN_DIR:+--env CKPT_RUN_DIR="${CKPT_RUN_DIR}"} \
  ${ORIGIN_HF_DIR:+--env ORIGIN_HF_DIR="${ORIGIN_HF_DIR}"} \
  ${EVAL_ITERS:+--env EVAL_ITERS="${EVAL_ITERS}"} \
  ${TASK_FILE:+--env TASK_FILE="${TASK_FILE}"} \
  ${N_EVAL_TASKS:+--env N_EVAL_TASKS="${N_EVAL_TASKS}"} \
  ${BROWSER_MAX_STEPS:+--env BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS}"} \
  ${JUDGE_MODEL:+--env JUDGE_MODEL="${JUDGE_MODEL}"} \
  --no-python \
  -- /bin/bash "$ENTRY"
