#!/usr/bin/env bash
# Staged eval for the native stack: convert one or more Megatron distcp checkpoints
# to HF, then evaluate them AND the SFT base back-to-back in a single beaker job.
#
# One job on purpose: the cluster has ~0 free GPU slots, so every extra experiment
# costs another 10-40h in the queue. Converting + serving + evaluating every arm
# inside one slot is far cheaper than one job per arm.
#
#   EVAL_ITERS="0000009" GPUS=1 SCRIPT=scripts/run_staged_eval.sh bash beaker/launch_train.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs/owrl_native_frepro/ckpt}"
EVAL_ITERS="${EVAL_ITERS:-0000009}"          # space-separated iter suffixes
BASE_HF="${BASE_HF:-/weka/oe-training-default/zixianm/OpenWebRL/models/OpenWebRL/OpenWebRL-4B-SFT}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"            # evaluate the SFT base as the reference arm
TASK_FILE="${TASK_FILE:-/weka/oe-training-default/zixianm/xunmei/zixian_molmoweb-rl/_data/eval_owrlpool_heldout120_gs18.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/staged_eval/heldout120}"

# Match the TRAINING judge configuration exactly, so eval numbers are comparable
# to the reward curve rather than to a differently-configured judge.
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
export JUDGE_API_MODE="${JUDGE_API_MODE:-served}"
export JUDGE_PROMPT_VARIANT="${JUDGE_PROMPT_VARIANT:-action_history}"
export CONTEXT_NUM_SCREENSHOTS="${CONTEXT_NUM_SCREENSHOTS:-1}"
export JUDGE_MAX_ATTACHED_IMGS="${JUDGE_MAX_ATTACHED_IMGS:-3}"
export BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-16}"
export SGLANG_TP="${SGLANG_TP:-1}"
export SGLANG_DP="${SGLANG_DP:-1}"

echo "[staged_eval] iters      = ${EVAL_ITERS}"
echo "[staged_eval] task file  = ${TASK_FILE} ($(wc -l < "${TASK_FILE}") tasks)"
echo "[staged_eval] output     = ${OUTPUT_ROOT}"
nvidia-smi -L

MODELS=""
[ "${INCLUDE_BASE}" = "1" ] && MODELS="${BASE_HF}"

for it in ${EVAL_ITERS}; do
  IN="${CKPT_ROOT}/iter_${it}"
  OUT="${IN}/hf_converted"
  if [ ! -f "${OUT}/config.json" ]; then
    echo "[staged_eval] converting iter_${it} -> ${OUT}"
    python tools/convert_torch_dist_to_hf_bridge.py \
      --input-dir "${IN}" --output-dir "${OUT}" --origin-hf-dir "${BASE_HF}" -f
  else
    echo "[staged_eval] reusing existing conversion at ${OUT}"
  fi
  MODELS="${MODELS} ${OUT}"
done

echo "[staged_eval] evaluating:${MODELS}"
TASK_FILE="${TASK_FILE}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
MODEL_LIST_OVERRIDE="${MODELS}" \
bash scripts/run_evaluation.sh
