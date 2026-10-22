#!/usr/bin/env bash
#SBATCH --job-name=owrl-joint-build
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only work; a CPU-only job on this partition queues for days, so take the debug
# GPU slot for a few minutes. The builder caps itself at 1 GiB / 240 CPU-s per process.
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/joint-build-%j.out
#
# tillicum port of klone/build_joint_data.sbatch: rebuild the joint C2 + Piotr pairs with
# scripts/prepare_arm_joint_data.py, verify, and write a data-driven train-config.json.
# K=1 must reproduce Zixian's joint-data-v2 pairs exactly; K=2 adds a second rejected
# candidate per training state.
#   NEGATIVES="1 2" sbatch tillicum/build_joint_data.sbatch
set -euo pipefail
NEGATIVES="${NEGATIVES:-1 2}"
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction
PY=$ZRUN/venv/bin/python
# verify_arm_joint_data.py reads <data>/../joint-data-inputs-0d83b48, so outputs live next to a link
# to it; the builder gets the real path so image/source paths match v2 byte for byte.
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data
V2=$ZRUN/joint-data-v2-20260910
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd $REPO
# Reuse v2's verified prompt tokenisations (her build needed them to fit the CPU cap);
# copy so the builder's SQLite recovery never touches her file.
cp $V2/working.sqlite $OPENWEBRL_JOB_TMP/context-cache.sqlite
for K in $NEGATIVES; do
  OUT=$ROOT/k$K
  [ -e $OUT ] && { echo "[build] $OUT exists; remove it to rebuild"; continue; }
  echo "[build] K=$K -> $OUT"
  $PY scripts/prepare_arm_joint_data.py --hf $ZRUN/joint-data-inputs-0d83b48 --c2 $ZRUN/runs/c2-full-282782-20260908T075414Z \
    --output $OUT --context-cache $OPENWEBRL_JOB_TMP/context-cache.sqlite --negatives-per-state $K \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --eval openwebrl/data/eval/online-mind2web.jsonl \
    || { echo "[build] FATAL prepare K=$K failed (SIGXCPU => raise RLIMIT_CPU in the script, visibly)"; exit 1; }
  $PY scripts/verify_arm_joint_data.py --data $OUT | tail -3
  for f in joint_pairs.train joint_pairs.validation joint_sft.train validation-panel-c2 validation-panel-piotr; do
    echo "[build] K=$K $f: $(wc -l < $OUT/$f.jsonl) rows, $(cmp -s $OUT/$f.jsonl $V2/$f.jsonl && echo identical-to-v2 || echo differs-from-v2)"
  done
  OWRL_JOINT_DATA=$OUT $PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[build] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
done
