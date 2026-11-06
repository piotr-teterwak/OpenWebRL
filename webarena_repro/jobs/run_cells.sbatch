#!/usr/bin/env bash
#SBATCH --job-name=wa-cells
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/webarena/jobs/cells-%x-%j.out
# WebArena-Lite cells on one site, one at a time, each from a fresh site copy (no GPU used for no_search; MIG slices
# are the cheapest CPU+RAM on this cluster). Model calls go out through wa_relay.py; the browser stays in the netns.
#   SITE=cms TASKS="0 2 8" METHOD=no_search sbatch run_cells.sbatch
set -uo pipefail
SITE=${SITE:?}; TASKS=${TASKS:?}; METHOD=${METHOD:-no_search}; export REWARD_DECODING=${REWARD_DECODING:-}
R=/gpfs/home/piotrt/OpenWebRL/webarena_repro; W=/gpfs/scrubbed/piotrt/owrl/webarena
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
TAG=${TAG:-}; export WA_JUDGE=${WA_JUDGE:-}  # TAG names a repeat/ablation (r1, r2, A3-steps10, ...)
POLICY=${POLICY:-}; export WA_ARBITER_FORMAT=${WA_ARBITER_FORMAT:-} WA_POLICY_STOP=${WA_POLICY_STOP:-} WA_TYPE_UNQUOTE=${WA_TYPE_UNQUOTE:-} WA_DETECTOR=${WA_DETECTOR:-} WA_SAMPLING=${WA_SAMPLING:-} WA_SAMPLES=${WA_SAMPLES:-} WA_MERGE_STOP=${WA_MERGE_STOP:-} WA_VIEWPORT=${WA_VIEWPORT:-} EPISODE_SCRIPT=${EPISODE_SCRIPT:-wa_episode.py} WA_ARM_CROP=${WA_ARM_CROP:-} WA_ARM_WINDOW=${WA_ARM_WINDOW:-}  # POLICY: e.g. gpt-4o-mini; WA_ARBITER_FORMAT=browsergym (A11); WA_SAMPLING=shepherd (A17)
[ -z "$WA_JUDGE" ] && unset WA_JUDGE
LOCAL=${TMPDIR:-/tmp}/wa-$SLURM_JOB_ID; RUN=$W/runs/cells/$SITE-$METHOD${TAG:+-$TAG}-$SLURM_JOB_ID
mkdir -p $LOCAL $RUN
# Shopping: unpack the packed pristine base (pack_site.sbatch) to node-local disk once per job; each cell then copies
# from there (wa_site.py WA_SITE_BASE_SHOPPING) instead of reading 67.6 GB of small files off GPFS per task.
if [ "$SITE" = shopping ] && [ -f $W/sandbox/shopping.tar.zst ]; then
  t0=$(date +%s); mkdir -p $LOCAL/base
  unshare --user --map-root-user sh -c "zstd -dc $W/sandbox/shopping.tar.zst | tar -C $LOCAL/base --numeric-owner --xattrs -xf -" \
    || { echo FATAL_UNPACK_SHOPPING; df -h $LOCAL; exit 1; }
  export WA_SITE_BASE_SHOPPING=$LOCAL/base; echo "unpacked shopping base in $(( $(date +%s) - t0 )) s; $(df -h $LOCAL | tail -1)"
fi
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'")
export OPENAI_API_KEY
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
python3 $R/wa_relay.py out --sock $LOCAL/egress.sock --ready $LOCAL/relay-out.json & RELAY=$!
ARB=""; ARB_IN=""
if [ "$METHOD" = selectionarm ]; then
  # Piotr's SelectionARM (scripts/serve_arm.py --mode selection, checkpoint 81b452d on OpenWebRL-4B-SFT) on this job's GPU,
  # reachable inside the namespace at 127.0.0.1:8003 through a second relay pair (wa_selectionarm.py calls /select).
  source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  APORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  (cd /gpfs/home/piotrt/OpenWebRL && setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/serve_arm.py --mode selection \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89 \
    --base /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --port $APORT --max-pixels ${ARM_MAX_PIXELS:-262144} > $RUN/selectionarm-server.log 2>&1) & ARB=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:$APORT/health >/dev/null; do sleep 10; kill -0 $ARB || { echo FATAL_SELECTIONARM_SERVER; tail -20 $RUN/selectionarm-server.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 1500 ] || { echo "FATAL_SELECTIONARM_NOT_HEALTHY_25MIN"; tail -20 $RUN/selectionarm-server.log; exit 1; }; done
  echo "SelectionARM healthy on $APORT: $(curl -s http://127.0.0.1:$APORT/health | cut -c1-200)"
  python3 $R/wa_relay.py out --sock $LOCAL/arbiter.sock --to 127.0.0.1:$APORT & ARB_RELAY=$!
  ARB_IN="$PY $R/wa_relay.py in --sock $LOCAL/arbiter.sock --listen 127.0.0.1:8003 & export WA_SELECTOR_URL=http://127.0.0.1:8003;"
fi
if [ "$METHOD" = arbiter ]; then
  # WebArbiter-7B @ 6518479b on this job's GPU, served as "WebArbiter" (the released routing name); reachable inside the
  # namespace at 127.0.0.1:8002 (released default OPENAI_API_URL) through a second relay pair.
  source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh; export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  APORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  setsid /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -m sglang.launch_server --model-path /gpfs/scrubbed/piotrt/owrl/models/WebArbiter-7B \
    --served-model-name WebArbiter --host 127.0.0.1 --port $APORT --dtype bfloat16 --tp 1 --mem-fraction-static 0.6 \
    --max-running-requests 8 --random-seed 42 --disable-cuda-graph > $RUN/webarbiter-server.log 2>&1 & ARB=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:$APORT/health_generate >/dev/null; do sleep 10; kill -0 $ARB || { echo FATAL_ARBITER_SERVER; tail -20 $RUN/webarbiter-server.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 1500 ] || { echo "FATAL_ARBITER_SERVER_NOT_HEALTHY_25MIN (job 414459 idled 5.4 h this way)"; tail -20 $RUN/webarbiter-server.log; exit 1; }; done
  echo "WebArbiter-7B healthy on $APORT"
  python3 $R/wa_relay.py out --sock $LOCAL/arbiter.sock --to 127.0.0.1:$APORT & ARB_RELAY=$!
  ARB_IN="$PY $R/wa_relay.py in --sock $LOCAL/arbiter.sock --listen 127.0.0.1:8002 & export OPENAI_API_URL=http://127.0.0.1:8002/v1;"
fi
cleanup() { kill $RELAY ${ARB_RELAY:-} 2>/dev/null; [ -n "$ARB" ] && kill -- -$ARB 2>/dev/null; touch $LOCAL/site/stop 2>/dev/null; sleep 15; unshare --user --map-root-user rm -rf $LOCAL; }
trap cleanup EXIT
echo "node $(hostname -s) site $SITE method $METHOD tasks: $TASKS"; echo "policy ${POLICY:-default} arbiter_format ${WA_ARBITER_FORMAT:-released} policy_stop ${WA_POLICY_STOP:-0} type_unquote ${WA_TYPE_UNQUOTE:-0} detector ${WA_DETECTOR:-released} sampling ${WA_SAMPLING:-released} samples ${WA_SAMPLES:-default} merge_stop ${WA_MERGE_STOP:-0} episode ${EPISODE_SCRIPT:-wa_episode.py} viewport ${WA_VIEWPORT:-1024x2490}"
for T in $TASKS; do
  t0=$(date +%s); CELL=$RUN/task-$T
  unshare --user --map-root-user rm -rf $LOCAL/state $LOCAL/site
  python3 $R/wa_site.py copy $SITE $LOCAL/state > $RUN/copy-$T.log 2>&1 || { echo "task $T: COPY_FAILED"; continue; }
  python3 $R/wa_site.py serve $SITE $LOCAL/state $LOCAL/site --hold 7200 > $RUN/serve-$T.log 2>&1 & SERVE=$!
  for i in $(seq 1 600); do [ -f $LOCAL/site/ready.json ] && break; kill -0 $SERVE 2>/dev/null || break; sleep 3; done
  if [ ! -f $LOCAL/site/ready.json ]; then echo "task $T: SITE_NOT_READY"; tail -5 $RUN/serve-$T.log; kill $SERVE 2>/dev/null
    mkdir -p $RUN/fail-$T; cp -r $LOCAL/site/*.log $LOCAL/site/svc-logs $RUN/fail-$T/ 2>/dev/null; continue; fi  # node-local, wiped next task
  cp $LOCAL/site/ready.json $RUN/ready-$T.json
  python3 $R/wa_site.py exec $LOCAL/site -- /bin/bash -c "
    $PY $R/wa_relay.py in --sock $LOCAL/egress.sock & RIN=\$!; $ARB_IN sleep 1
    $PY $R/${EPISODE_SCRIPT:-wa_episode.py} --task-id $T --method $METHOD --out $CELL \${REWARD_DECODING:+--reward-decoding \$REWARD_DECODING} ${POLICY:+--policy $POLICY}; rc=\$?
    kill \$RIN; exit \$rc" > $RUN/episode-$T.log 2>&1
  echo "task $T: rc=$? $(cat $CELL/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400) wall=$(( $(date +%s) - t0 ))s"
  touch $LOCAL/site/stop; wait $SERVE 2>/dev/null
done
echo CELLS_DONE
