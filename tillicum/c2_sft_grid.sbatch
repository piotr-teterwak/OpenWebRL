#!/usr/bin/env bash
#SBATCH --job-name=owrl-grid
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
# Same shape as 1A (285567): 4 CPUs, 120G.
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
# ~236 updates at ~42 s (measured on 372697/8) = 2h45, plus merge and held-out scoring.
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/grid-%j.out
#
# One cell of the C2 SFT learning-rate grid: the pinned 1A trainer and recipe, except
# learning rate, on the host-grouped train split (scripts/build_c2_heldout_grid.py),
# with ARM-picked or random targets. Then merge the adapter and score the model on the
# held-out states (scripts/score_c2_heldout.py). VARIANT=base only scores the base actor.
#
#   VARIANT=arm LR=1e-4 sbatch tillicum/c2_sft_grid.sbatch
#   VARIANT=base sbatch --time=00:45:00 tillicum/c2_sft_grid.sbatch
set -euo pipefail
VARIANT="${VARIANT:?set VARIANT=arm|random|base}"
case "$VARIANT" in arm|random|base) ;; *) echo "bad VARIANT=$VARIANT"; exit 1;; esac
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction
TRAINER=$ZRUN/action-reward-models-c2/actor_distillation/train_openwebrl_c2.py
TRAINER_SHA256=704dc25f95a807428c12def2866a71f66dd1f7e74d62107c1be5b2a5ae0b0363
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
GRID=/gpfs/scrubbed/piotrt/owrl/sft-grid
DATA=$GRID/data
PY=$ZRUN/venv/bin/python

export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd $REPO
[ -f $DATA/split.json ] || { echo "FATAL $DATA not built; run scripts/build_c2_heldout_grid.py first"; exit 1; }
mkdir -p $GRID/scores

if [ "$VARIANT" = base ]; then
  $PY scripts/score_c2_heldout.py --heldout $DATA/heldout.jsonl --model $BASE --label base --out $GRID/scores/base.jsonl
  exit 0
fi

LR="${LR:?set LR, e.g. 1e-4}"
LABEL=$VARIANT-lr$LR
RUN=$GRID/$LABEL
[ "$(sha256sum $TRAINER | cut -d' ' -f1)" = "$TRAINER_SHA256" ] || { echo "FATAL 1A trainer checksum changed"; exit 1; }
mkdir -p $RUN
cp $DATA/$VARIANT/dataset-audit.json $RUN/dataset-audit.json
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
END_UTC=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)  # %e is local time; no -u
# Stop training early enough to leave ~25 min for merge + scoring.
STOP=$(( END_UTC - 1800 ))
[ $(( STOP - $(date +%s) )) -gt 1800 ] || { echo "FATAL stop time is <30 min away; check the deadline math"; exit 1; }
CONFIG=$RUN/frozen-config-$SLURM_JOB_ID.json
$PY - "$CONFIG" <<PYEOF
import json, sys
from datetime import datetime, timezone
recipe = json.load(open('$ZRUN/runs/c2-ablation-1a/frozen-config.json'))['training']
rows = sum(1 for _ in open('$DATA/$VARIANT/training.jsonl'))
# Only the learning rate is swept; the cosine horizon follows 1A's rule (two dataset sizes),
# which the trainer enforces, on the smaller train split.
recipe.update(ablation='$LABEL', learning_rate=float('$LR'), cosine_horizon_examples=2 * rows)
json.dump({'allocation': '$SLURM_JOB_ID', 'gpu_uuid': '$GPU_UUID', 'node': '$(hostname -s)', 'output': '$RUN',
           'task_count': 2091, 'stop_utc': datetime.fromtimestamp($STOP, timezone.utc).isoformat(),
           'teacher_manifest': {'actor': '$BASE'}, 'training': recipe, 'trainer_sha256': '$TRAINER_SHA256',
           'variant': '$VARIANT'}, open(sys.argv[1], 'w'), indent=1)
PYEOF

RESUME=()
if [ -f $RUN/student/latest-checkpoint.json ] && [ ! -f $RUN/student/complete.json ]; then
  RESUME=(--resume "$($PY -c "import json;print(json.load(open('$RUN/student/latest-checkpoint.json'))['path'])")")
fi
if [ ! -f $RUN/student/complete.json ]; then
  echo "[grid] $LABEL train node=$(hostname -s) ${RESUME[*]}"
  $PY $TRAINER --config $CONFIG "${RESUME[@]}"
fi
[ -f $RUN/student/complete.json ] || { echo "[grid] $LABEL paused before the end of the epoch; resubmit to resume"; exit 0; }

[ -f $RUN/merged/merge-manifest.json ] || $PY scripts/merge_arm_c2_student.py --run-root $RUN \
  --checkpoint $RUN/student/epoch-1 --output $RUN/merged
$PY scripts/score_c2_heldout.py --heldout $DATA/heldout.jsonl --model $RUN/merged --label $LABEL --out $GRID/scores/$LABEL.jsonl
echo "[grid] $LABEL DONE"
