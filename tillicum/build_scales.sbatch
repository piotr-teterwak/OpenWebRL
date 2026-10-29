#!/usr/bin/env bash
#SBATCH --job-name=owrl-scales
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (tokenizer + JSON); a MIG slice bills 1/7 of an H200 and CPU-only jobs queue for days.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/scales-%j.out
#
# Data-scaling datasets for matched-pair DPO/RPO, all sharing joint-data-v2's validation split:
#   1. build K=1 with every Piotr teacher draw kept (--draws-per-state 16; C2 has one draw per state)
#   2. matched filter (equal chosen/rejected action counts)
#   3. scale points, nested: fraction of task groups at D=1, then D extra draws per state
#   4. verify each and write its train-config.json
# Outputs: /gpfs/scrubbed/piotrt/owrl/joint-data/scale-<name>. Rerunning skips finished steps.
set -euo pipefail
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction
PY=$ZRUN/venv/bin/python
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data
V2=$ZRUN/joint-data-v2-20260910
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd $REPO
FULL=$ROOT/k1-d16; MATCHED=$ROOT/matched-d16
if [ ! -f $FULL/.built ]; then
  rm -rf $FULL
  cp $V2/working.sqlite $OPENWEBRL_JOB_TMP/context-cache.sqlite
  $PY scripts/prepare_arm_joint_data.py --hf $ZRUN/joint-data-inputs-0d83b48 --c2 $ZRUN/runs/c2-full-282782-20260908T075414Z \
    --output $FULL --context-cache $OPENWEBRL_JOB_TMP/context-cache.sqlite --draws-per-state 16 --cpu-seconds 7200 \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --eval openwebrl/data/eval/online-mind2web.jsonl
  $PY scripts/verify_arm_joint_data.py --data $FULL --cpu-seconds 3600 | tail -3
  for f in joint_pairs.validation joint_sft.train joint_sft.validation; do
    echo "[scales] $f vs v2: $(cmp -s $FULL/$f.jsonl $V2/$f.jsonl && echo identical || echo DIFFERS)"
  done
  touch $FULL/.built   # not trained on directly, so no train-config (its final batch may be odd)
fi
[ -d $MATCHED ] || $PY scripts/filter_joint_pairs.py --src $FULL --out $MATCHED
for spec in f0.25:1:0.25 f0.5:1:0.5 d1:1:1 d2:2:1 d4:4:1 d16:16:1; do
  IFS=: read name draws frac <<<"$spec"; OUT=$ROOT/scale-$name
  [ -f $OUT/train-config.json ] && { echo "[scales] $name exists"; continue; }
  rm -rf $OUT
  $PY scripts/subsample_joint_pairs.py --src $MATCHED --out $OUT --max-draw $draws --fraction $frac
  $PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 3600 | tail -1
  $PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[scales] $name', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
done
# d1 must reproduce the 4,078-pair matched set used so far.
cmp -s $ROOT/scale-d1/joint_pairs.train.jsonl $ROOT/matched/joint_pairs.train.jsonl && echo "[scales] d1 == matched" || echo "[scales] NOTE d1 differs from matched"
