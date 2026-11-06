#!/usr/bin/env bash
#SBATCH --job-name=om2w-cells
#SBATCH --account=vlm
#SBATCH --qos=normal
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/webarena/jobs/om2w-%x-%j.out
# A18 harness on live Online-Mind2Web (om2w_episode.py), one task at a time, no site hosting or network namespace.
#   METHOD=no_search|arbiter|selectionarm INDICES="0 1 2" TAG=N18v720 WA_POLICY_STOP=full WA_DETECTOR=off \
#   [WA_SAMPLING=shepherd] WA_VIEWPORT=1280x720 sbatch [-p ... --gpus=1 ...] om2w_cells.sbatch
set -uo pipefail
METHOD=${METHOD:?}; INDICES=${INDICES:?}; TAG=${TAG:-}
R=/gpfs/home/piotrt/OpenWebRL/webarena_repro; W=/gpfs/scrubbed/piotrt/owrl/webarena
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
export WA_POLICY_STOP=${WA_POLICY_STOP:-} WA_DETECTOR=${WA_DETECTOR:-} WA_SAMPLING=${WA_SAMPLING:-} WA_VIEWPORT=${WA_VIEWPORT:-1280x720}
export OM2W_JUDGE_PRICE=${OM2W_JUDGE_PRICE:-0,0}
RUN=$W/runs/om2w/$METHOD${TAG:+-$TAG}-$SLURM_JOB_ID; mkdir -p $RUN
LOCAL=${TMPDIR:-/tmp}/om2w-$SLURM_JOB_ID; mkdir -p $LOCAL; export WEBPRM_BROWSER_TMP_ROOT=/tmp
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'")
export OPENAI_API_KEY; [ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
ARB=""
port() { python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()"; }
wait_healthy() {  # url log name
  t0=$(date +%s)
  until curl -sf --noproxy '*' $1 >/dev/null; do sleep 10; kill -0 $ARB || { echo "FATAL_$3_SERVER"; tail -20 $2; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 1500 ] || { echo "FATAL_$3_NOT_HEALTHY_25MIN"; tail -20 $2; exit 1; }; done; echo "$3 healthy"; }
if [ "$METHOD" = selectionarm ]; then
  source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
  APORT=$(port)
  (cd /gpfs/home/piotrt/OpenWebRL && setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/serve_arm.py --mode selection \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89 \
    --base /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --port $APORT > $RUN/selectionarm-server.log 2>&1) & ARB=$!
  wait_healthy http://127.0.0.1:$APORT/health $RUN/selectionarm-server.log SELECTIONARM
  export WA_SELECTOR_URL=http://127.0.0.1:$APORT
fi
if [ "$METHOD" = arbiter ]; then
  source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
  APORT=$(port)
  setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -m sglang.launch_server --model-path /gpfs/scrubbed/piotrt/owrl/models/WebArbiter-7B \
    --served-model-name WebArbiter --host 127.0.0.1 --port $APORT --dtype bfloat16 --tp 1 --mem-fraction-static 0.6 \
    --max-running-requests 8 --random-seed 42 --disable-cuda-graph > $RUN/webarbiter-server.log 2>&1 & ARB=$!
  wait_healthy http://127.0.0.1:$APORT/health_generate $RUN/webarbiter-server.log ARBITER
  export OPENAI_API_URL=http://127.0.0.1:$APORT/v1
fi
cleanup() { [ -n "$ARB" ] && kill -- -$ARB 2>/dev/null; rm -rf $LOCAL; }
trap cleanup EXIT
echo "node $(hostname -s) method $METHOD indices: $INDICES"
echo "policy_stop ${WA_POLICY_STOP:-0} detector ${WA_DETECTOR:-released} sampling ${WA_SAMPLING:-released} viewport $WA_VIEWPORT"
for I in $INDICES; do
  t0=$(date +%s)
  timeout ${TASK_TIMEOUT:-3000} $PY $R/om2w_episode.py --index $I --method $METHOD --out $RUN/task-$I > $RUN/episode-$I.log 2>&1
  rc=$?; if [ ! -f $RUN/task-$I/summary.json ]; then mkdir -p $RUN/task-$I; printf '{"index": %s, "method": "%s", "valid": false, "success": null, "steps": 0, "error": "killed at the %ss task limit (rc %s)", "usd": {}, "seconds": %s}\n' $I $METHOD ${TASK_TIMEOUT:-3000} $rc $(( $(date +%s) - t0 )) > $RUN/task-$I/summary.json; fi
  echo "task $I: rc=$rc $(cat $RUN/task-$I/summary.json 2>/dev/null | tr -d '\n' | cut -c1-500) wall=$(( $(date +%s) - t0 ))s"
done
echo CELLS_DONE
