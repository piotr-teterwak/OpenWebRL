#!/usr/bin/env bash
#SBATCH --job-name=live-cells
#SBATCH --account=vlm
#SBATCH --qos=normal
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/webarena/jobs/live-%x-%j.out
# A18 harness on live OM2W / WebVoyager / DeepShop (live_episode.py). Generalizes om2w_cells.sbatch (unchanged):
# several methods per job (each selector server is started once, only if a listed method needs it) and a few
# parallel lanes.
#   JOBS="method:benchmark:index ..."   explicit episode list, or
#   METHODS="no_search arbiter selectionarm" BENCHMARK=webvoyager INDICES="0 1 2"   (method-major product)
#   LANES=1 (parallel episodes; lane k runs JOBS k, k+LANES, ...)  TAG=...  TASK_TIMEOUT=1500 (s per episode)
#   USD_CAP=5 (per episode; USD_CAP_<method> overrides)  LIVE_SHADOW_JUDGE_IMAGES=30 (0 = no trajectory verdict)
#   WA_POLICY_STOP=full WA_DETECTOR=off WA_SAMPLING=shepherd WA_VIEWPORT=1280x720   (A18; no_search always gets the
#   released single sample unless WA_SAMPLING_no_search is set, as in the OM2W N18 row)
# GPU only if arbiter or selectionarm is listed: sbatch --gpus=1 --cpus-per-task=8 --mem=120G --time=... live_cells.sbatch
# no_search alone: sbatch -p gpu-h200-mig --gpus=1 --cpus-per-task=2 --mem=32G ... (no GPU work; MIG is the cheapest slot)
set -uo pipefail
R=/gpfs/home/piotrt/OpenWebRL/webarena_repro; W=/gpfs/scrubbed/piotrt/owrl/webarena
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
export WA_POLICY_STOP=${WA_POLICY_STOP:-} WA_DETECTOR=${WA_DETECTOR:-} WA_SAMPLING=${WA_SAMPLING:-} WA_VIEWPORT=${WA_VIEWPORT:-1280x720}
export OM2W_JUDGE_PRICE=${OM2W_JUDGE_PRICE:-0,0} LIVE_SHADOW_JUDGE_IMAGES=${LIVE_SHADOW_JUDGE_IMAGES:-30}
TAG=${TAG:-}; LANES=${LANES:-1}; TASK_TIMEOUT=${TASK_TIMEOUT:-1500}
if [ -z "${JOBS:-}" ]; then
  : ${METHODS:?} ${BENCHMARK:?} ${INDICES:?}; JOBS=""
  for M in $METHODS; do for I in $INDICES; do JOBS="$JOBS $M:$BENCHMARK:$I"; done; done
fi
RUN=$W/runs/live/${TAG:+$TAG-}$SLURM_JOB_ID; mkdir -p $RUN
export WEBPRM_BROWSER_TMP_ROOT=/tmp
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'")
export OPENAI_API_KEY; [ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
SERVERS=""
cleanup() { for p in $SERVERS; do kill -- -$p 2>/dev/null; done; }
trap cleanup EXIT
port() { python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()"; }
wait_healthy() {  # url log name pid
  t0=$(date +%s)
  until curl -sf --noproxy '*' $1 >/dev/null; do sleep 15; kill -0 $4 || { echo "FATAL_$3_SERVER"; tail -20 $2; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 1500 ] || { echo "FATAL_$3_NOT_HEALTHY_25MIN"; tail -20 $2; exit 1; }; done; echo "$3 healthy after $(( $(date +%s) - t0 ))s"; }
if [[ " $JOBS" == *" selectionarm:"* || " $JOBS" == *" arbiter:"* ]]; then
  source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
fi
if [[ " $JOBS" == *" selectionarm:"* ]]; then  # our ADM: SelectionARM 81b452d8 (same command as om2w_cells.sbatch)
  SPORT=$(port)
  (cd /gpfs/home/piotrt/OpenWebRL && setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/serve_arm.py --mode selection \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89 \
    --base /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --port $SPORT > $RUN/selectionarm-server.log 2>&1) & SPID=$!
  SERVERS="$SERVERS $SPID"; export WA_SELECTOR_URL=http://127.0.0.1:$SPORT
fi
if [[ " $JOBS" == *" arbiter:"* ]]; then  # WebArbiter-7B (same command as om2w_cells.sbatch)
  APORT=$(port)
  setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -m sglang.launch_server --model-path /gpfs/scrubbed/piotrt/owrl/models/WebArbiter-7B \
    --served-model-name WebArbiter --host 127.0.0.1 --port $APORT --dtype bfloat16 --tp 1 --mem-fraction-static 0.6 \
    --max-running-requests 8 --random-seed 42 --disable-cuda-graph > $RUN/webarbiter-server.log 2>&1 & APID=$!
  SERVERS="$SERVERS $APID"; export OPENAI_API_URL=http://127.0.0.1:$APORT/v1
fi
[ -n "${SPID:-}" ] && wait_healthy http://127.0.0.1:$SPORT/health $RUN/selectionarm-server.log SELECTIONARM $SPID
[ -n "${APID:-}" ] && wait_healthy http://127.0.0.1:$APORT/health_generate $RUN/webarbiter-server.log ARBITER $APID
echo "node $(hostname -s) lanes $LANES jobs:$JOBS"
echo "policy_stop ${WA_POLICY_STOP:-0} detector ${WA_DETECTOR:-released} sampling ${WA_SAMPLING:-released} (no_search: ${WA_SAMPLING_no_search:-released}) viewport $WA_VIEWPORT shadow_judge_images $LIVE_SHADOW_JUDGE_IMAGES"
read -ra LIST <<< "$JOBS"
lane() {
  for ((k=$1; k<${#LIST[@]}; k+=LANES)); do
    IFS=: read -r M B I <<< "${LIST[$k]}"; D=$RUN/$B/$M; mkdir -p $D; t0=$(date +%s)
    CAPVAR=USD_CAP_$M; CAP=${!CAPVAR:-${USD_CAP:-5}}
    # A18 sampling is per row: Web-Shepherd (20 samples, 5 most frequent) for the selector rows, the released single
    # sample for no_search (OM2W N18 job 422923: "sampling released"; Shepherd's 5 BLOCKS fail no_search's one-BLOCK check)
    SAMP=${WA_SAMPLING:-}; [ "$M" = no_search ] && SAMP=${WA_SAMPLING_no_search:-}
    WA_SAMPLING=$SAMP timeout $TASK_TIMEOUT $PY $R/live_episode.py --benchmark $B --index $I --method $M --usd-cap $CAP --out $D/task-$I > $D/episode-$I.log 2>&1
    rc=$?; if [ ! -f $D/task-$I/summary.json ]; then mkdir -p $D/task-$I; printf '{"benchmark": "%s", "index": %s, "method": "%s", "valid": false, "success": null, "steps": 0, "error": "no summary: rc %s (124 = killed at the %ss task limit)", "usd": {}, "seconds": %s}\n' $B $I $M $rc $TASK_TIMEOUT $(( $(date +%s) - t0 )) > $D/task-$I/summary.json; fi
    echo "lane $1 $M $B $I: rc=$rc $(tr -d '\n' < $D/task-$I/summary.json | cut -c1-400) wall=$(( $(date +%s) - t0 ))s"
  done
}
LPIDS=""; for ((l=0; l<LANES; l++)); do lane $l & LPIDS="$LPIDS $!"; sleep 20; done
wait $LPIDS  # not a bare wait: the selector servers are children too
echo CELLS_DONE $RUN
