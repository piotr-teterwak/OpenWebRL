#!/usr/bin/env bash
#SBATCH --job-name=pro-cua-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/pro-cua-%j.out
# PRO-CUA-4B standalone on our benchmark tasks with our judges (pro_cua_eval.py). One SGLang server (same flags as
# tillicum/eval_concurrent.sbatch, context 12800 as their vLLM launch), then the agent loop.
#   BENCHES="webvoyager deepshop" EXCLUDE="allrecipe github" LIMIT= sbatch --exclude=g011,g013,g015,g021 pro_cua_eval.sbatch
set -uo pipefail
BENCHES=${BENCHES:-deepshop}; EXCLUDE=${EXCLUDE:-}; LIMIT=${LIMIT:-}; PAR=${PAR:-8}; TAG=${TAG:-run}; REJUDGE=${REJUDGE:-}
REPO=/gpfs/home/piotrt/OpenWebRL; ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
MODEL=/gpfs/scrubbed/piotrt/owrl/models/PRO-CUA-4B; OUT=/gpfs/scrubbed/piotrt/owrl/pro-cua-eval/$TAG
source $REPO/scripts/h200_env.sh   # CUDA_HOME for sglang's kernels, private job TMPDIR (as eval_concurrent)
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
export TRITON_CACHE_DIR=/gpfs/scrubbed/piotrt/owrl/cache/triton; mkdir -p $TRITON_CACHE_DIR  # h200_env points at Zixian's (read-only to us)
export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
set -a; source $REPO/.env; set +a
# our .env key is stale since Zixian's Oct 31 rotation; the live one is in Zixian's .env (as webarena run_cells.sbatch)
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' /gpfs/projects/krishna/zixianma/OpenWebRL/.env | head -1 | cut -d= -f2- | tr -d "\"'"); export OPENAI_API_KEY
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL no OpenAI key"; exit 1; }
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
setsid $SERVER_PY -m sglang.launch_server --model-path $MODEL --served-model-name pro-cua-4b --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --tp 1 --mem-fraction-static 0.8 --context-length 12800 --max-running-requests 24 \
  --chunked-prefill-size 4096 --disable-cuda-graph > $OUT-server.log 2>&1 & SRV=$!
trap 'kill -- -$SRV 2>/dev/null; sleep 5; rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
mkdir -p $(dirname $OUT); t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health_generate >/dev/null; do sleep 10; kill -0 $SRV || { echo "FATAL server died"; tail -20 $OUT-server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL server not healthy"; exit 1; }; done
echo "[pro-cua] server healthy on $PORT after $(( $(date +%s) - t0 )) s"
cd $REPO
for B in $BENCHES; do
  $ARM_PY pro_cua/pro_cua_eval.py --bench $B --model-url http://127.0.0.1:$PORT/v1 --model-name pro-cua-4b \
    --out $OUT/$B --parallel $PAR ${LIMIT:+--limit $LIMIT} ${EXCLUDE:+--exclude $EXCLUDE} ${REJUDGE:+--rejudge} >> $OUT-$B.log 2>&1
  echo "[pro-cua] $B: $(tail -1 $OUT-$B.log)"
done
