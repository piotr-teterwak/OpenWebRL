#!/usr/bin/env bash
#SBATCH --job-name=sft-cands
#SBATCH --account=vlm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/sft-cands-%j.out
#
# Mixed-pool Round 2 (2026-11-19): N (3) fresh candidates from ACTOR (default the SFT model) at every stored state of
# the half-B Round-2 collection exports, with the exact actor prompt the collection actor saw (scripts/sample_sft_candidates.py).
# One SGLang server as in c2_collect.sbatch; seeds are sampled one after another into OUTDIR/sft-s<seed>.jsonl (resumable).
#   SEEDS="42 43" LIMIT=50 sbatch tillicum/sample_sft_candidates.sbatch                         # pilot
#   SEEDS=42 sbatch tillicum/sample_sft_candidates.sbatch; SEEDS=43 sbatch ...                   # all states, one job per seed
# Works on a MIG slice too (pilot, ~0.2 states/s): MEMF=0.8 MAXREQ=24 PARALLEL=8 sbatch -p gpu-h200-mig --gpus=1 --cpus-per-task=2 --mem=32G --qos=debug ...
set -euo pipefail
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python; SERVER_PY=$ZRT/venv/bin/python
C=/gpfs/scrubbed/piotrt/owrl/c2-collect
ACTOR=${ACTOR:-/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT}
PREFIX=${PREFIX:-sol2-Br1}; SEEDS=${SEEDS:-42 43}; N=${N:-3}
OUTDIR=${OUTDIR:-$C/sol2-mix}
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
cd $REPO; mkdir -p $OUTDIR
PORT=$((28000 + SLURM_JOB_ID % 1000))
# MIG slices have ~18 GB: a larger static fraction leaves the KV cache room there; on an H200 0.4 as in c2_collect.
MEMF=${MEMF:-0.4}
setsid $SERVER_PY -m sglang.launch_server --model-path $ACTOR --host 127.0.0.1 --port $PORT --dtype bfloat16 --tp 1 \
  --mem-fraction-static $MEMF --context-length 32768 --max-running-requests ${MAXREQ:-96} --chunked-prefill-size 4096 \
  --cuda-graph-max-bs ${MAXREQ:-96} > $OUTDIR/sglang-$SLURM_JOB_ID.log 2>&1 &
SPID=$!; trap 'kill -- -$SPID 2>/dev/null || true; rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health_generate >/dev/null; do
  kill -0 $SPID 2>/dev/null || { echo "FATAL sglang died"; tail -30 $OUTDIR/sglang-$SLURM_JOB_ID.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL sglang not healthy after 15 min"; exit 1; }
  sleep 5
done
echo "[sft-cands] server healthy after $(( $(date +%s) - t0 ))s; actor=$ACTOR"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 60 > $OUTDIR/gpu-util-$SLURM_JOB_ID.csv &
for s in $SEEDS; do
  $ARM_PY scripts/sample_sft_candidates.py --dataset $C/$PREFIX-s$s-all-shard1/training.jsonl $C/$PREFIX-s$s-all-shard2/training.jsonl \
    --port $PORT --out $OUTDIR/sft-s$s${LIMIT:+-pilot}.jsonl --n $N --seed $s --parallel ${PARALLEL:-32} --model $ACTOR ${LIMIT:+--limit $LIMIT}
done
echo "[sft-cands] done in $(( $(date +%s) - t0 ))s"
