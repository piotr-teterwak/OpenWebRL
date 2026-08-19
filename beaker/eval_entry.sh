#!/usr/bin/env bash
# ON-NODE entry for OpenWebRL eval (Beaker/Gantry). For each target iter:
#   convert torch_dist -> HF, serve via sglang (triton on B200), then for each
#   judge in JUDGES (openai=gpt-4.1, gemini=gemini-2.5-flash) run run_evaluate.py
#   and record the success rate. Default targets ~ every 10 training iters.
set -uo pipefail

REPO=/weka/oe-training-default/zixianm/OpenWebRL
BUILD=/weka/oe-training-default/zixianm/openwebrl_build
CONDA=/weka/oe-training-default/zixianm/.conda/envs/webrl
NV="$CONDA/lib/python3.12/site-packages/nvidia"
: "${OPENAI_API_KEY:?OPENAI_API_KEY (Beaker secret) required}"

export PATH="$CONDA/bin:$PATH"
export CUDA_HOME="$CONDA"
_inc="$CONDA/targets/x86_64-linux/include"; _lib="$CONDA/targets/x86_64-linux/lib"
for d in "$NV"/*/include; do case "$d" in *"/cu13/"*) continue;; esac; _inc="$_inc:$d"; done
for d in "$NV"/*/lib;     do case "$d" in *"/cu13/"*) continue;; esac; _lib="$_lib:$d"; done
export CPATH="$_inc:${CPATH:-}"; export LIBRARY_PATH="$_lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA/lib:$_lib:${LD_LIBRARY_PATH:-}"
export PLAYWRIGHT_BROWSERS_PATH=/weka/oe-training-default/zixianm/.cache/ms-playwright
export PYTHONPATH="$REPO:$BUILD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

if ! dpkg -s libgbm1 >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y --no-install-recommends libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libxcb-dri3-0 libatspi2.0-0 libnss3 libnspr4 libcups2 libatk1.0-0 \
    libatk-bridge2.0-0 libpango-1.0-0 libcairo2 libasound2 libpangocairo-1.0-0 >/dev/null 2>&1 || true
fi

export SLIME_BROWSER_ENV_MODE=local_process
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-16}"

# ---- what to evaluate ----
CKPT_RUN_DIR="${CKPT_RUN_DIR:-$BUILD/beaker_out/full4_resume_0623_0443/ckpt}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-$REPO/models/OpenWebRL/OpenWebRL-4B-SFT}"
EVAL_ITERS="${EVAL_ITERS:-9 19 29}"
TASK_FILE="${TASK_FILE:-openwebrl/data/webgym_test_attempted414_openwebrl.jsonl}"
N_EVAL_TASKS="${N_EVAL_TASKS:-30}"
BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"
N_PARALLEL="${N_PARALLEL:-12}"
JUDGES="${JUDGES:-openai gemini_rubric}"    # openai|gemini|openai_rubric|gemini_rubric (rubric = molmoweb-matching)
OUT="${EVAL_OUT:-$BUILD/eval_out/$(basename "$CKPT_RUN_DIR")}"; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"; [ -f "$SUMMARY" ] || echo -e "iter\tjudge\tsuccess_rate\ttasks\toutput" > "$SUMMARY"

# resolve judge env for a backend
set_judge() {
  export JUDGE_API_MODE=served
  local backend="$1" variant="action_history"
  case "$backend" in *_rubric) variant="webgym_rubric"; backend="${backend%_rubric}" ;; esac
  export JUDGE_PROMPT_VARIANT="$variant"
  case "$backend" in
    openai) export JUDGE_MODEL=gpt-4.1 JUDGE_API_BASE="https://api.openai.com/v1" JUDGE_API_KEY="$OPENAI_API_KEY"; unset JUDGE_API_BASE_NO_V1 ;;
    gemini) : "${GOOGLE_API_KEY:?GOOGLE_API_KEY (Beaker secret) required for gemini judge}"
            export JUDGE_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" JUDGE_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai" JUDGE_API_KEY="$GOOGLE_API_KEY" JUDGE_API_BASE_NO_V1=1 ;;
    *) echo "[eval] unknown judge backend: $1"; return 1 ;;
  esac
}

PORT="${SGLANG_PORT:-9000}"; IP=127.0.0.1; serve_pid=""
stop_serve(){ [ -n "$serve_pid" ] && kill -TERM -- "-$serve_pid" 2>/dev/null; sleep 4; [ -n "$serve_pid" ] && kill -KILL -- "-$serve_pid" 2>/dev/null; serve_pid=""; }
trap stop_serve EXIT
TASK_IDX="$(seq -s, 0 $((N_EVAL_TASKS-1)))"
cd "$REPO"

for it in $EVAL_ITERS; do
  ITERDIR="$CKPT_RUN_DIR/iter_$(printf '%07d' "$it")"
  [ -f "$ITERDIR/common.pt" ] || { echo "[eval] skip iter $it (no checkpoint)"; continue; }
  echo "==================== EVAL iter $it ===================="
  bash scripts/run_convert_hf.sh "$ITERDIR" "$ORIGIN_HF_DIR" || { echo "[eval] convert failed iter $it"; continue; }
  HF="$(find "$ITERDIR" -maxdepth 1 -type d -name '*_converted' | head -1)"
  [ -n "$HF" ] || { echo "[eval] no converted dir iter $it"; continue; }

  stop_serve
  setsid bash -lc "exec python -m sglang.launch_server --model-path '$HF' --host 0.0.0.0 --port $PORT \
    --trust-remote-code --dtype bfloat16 --mem-fraction-static 0.8 --context-length 32768 \
    --tp 1 --dp 1 --attention-backend triton" > "$OUT/serve_iter${it}.log" 2>&1 &
  serve_pid=$!
  for _ in $(seq 1 300); do curl -sf "http://$IP:$PORT/health_generate" >/dev/null 2>&1 && break; kill -0 "$serve_pid" 2>/dev/null || { echo "[eval] sglang died:"; tail -30 "$OUT/serve_iter${it}.log"; break; }; sleep 2; done

  for jb in $JUDGES; do
    set_judge "$jb" || continue
    ODIR="$OUT/iter_${it}_${jb}"; mkdir -p "$ODIR"
    echo "-------- iter $it judge=$jb ($JUDGE_MODEL) --------"
    python openwebrl/run_evaluate.py \
      --task-file "$TASK_FILE" --task-indices "$TASK_IDX" \
      --hf-checkpoint "$HF" --output "$ODIR" \
      --sglang-ip "$IP" --sglang-port "$PORT" --n-parallel "$N_PARALLEL" \
      --judge-model "$JUDGE_MODEL" --judge-api-mode served --judge-prompt-variant "$JUDGE_PROMPT_VARIANT" \
      --context-num-screenshots 1 --judge-max-attached-imgs 3 \
      --browser-response-format-mode browser_env --browser-include-tool-response 1 \
      --max-steps "$BROWSER_MAX_STEPS" --turn-level 2>&1 | tee "$ODIR/eval.log"
    SR="$(grep -aoE 'Success rate: [0-9.]+%' "$ODIR/eval.log" | tail -1 | grep -aoE '[0-9.]+')"
    echo -e "${it}\t${jb}\t${SR:-NA}%\t${N_EVAL_TASKS}\t${ODIR}" | tee -a "$SUMMARY"
  done
  stop_serve
done

echo "==================== EVAL SUMMARY ===================="; cat "$SUMMARY"
