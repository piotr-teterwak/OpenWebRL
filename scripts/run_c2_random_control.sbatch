#!/usr/bin/env bash
#SBATCH --job-name=c2-sft-1gpu
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
# Same shape as 1A (285567): 4 CPUs, 120G. Its MaxRSS was 41G including eval.
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
# 1A trained 263 updates in 2h53m on one H200; 4h leaves room to load and save.
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/c2-sft-%j.out
#
# Random-selection control for C2 ablation 1A: the 1A recipe and the pinned 1A
# trainer, on the same 8,394 states with each target replaced by a random
# candidate (scripts/build_c2_random_control.py). Training only -- evaluation
# is a separate concurrent job against the base actor and 1A.
#
# VARIANT=arm retrains 1A itself on the original ARM-pick data, for reproduction.
#
#   MODE=smoke sbatch --qos=debug --time=00:45:00 scripts/run_c2_random_control.sbatch
#   MODE=full  [VARIANT=arm] sbatch scripts/run_c2_random_control.sbatch   # resumes a paused run
set -euo pipefail
MODE="${MODE:?set MODE=smoke or MODE=full}"
VARIANT="${VARIANT:-random}"
case "$VARIANT" in random|arm) ;; *) echo "bad VARIANT=$VARIANT"; exit 1;; esac
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction
TRAINER=$ZRUN/action-reward-models-c2/actor_distillation/train_openwebrl_c2.py
TRAINER_SHA256=704dc25f95a807428c12def2866a71f66dd1f7e74d62107c1be5b2a5ae0b0363
C2=$ZRUN/runs/c2-full-282782-20260908T075414Z
if [ "$VARIANT" = random ]; then ROOT=/gpfs/scrubbed/piotrt/owrl/c2-random-control; else ROOT=/gpfs/scrubbed/piotrt/owrl/c2-1a-retrain; fi
DATA=$ROOT/data
RUN=$ROOT/$MODE

# Her env script, but with every writable cache in our scratch: her cache dirs
# are not writable by us.
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
PY=$ZRUN/venv/bin/python
cd $REPO

[ "$(sha256sum $TRAINER | cut -d' ' -f1)" = "$TRAINER_SHA256" ] || { echo "FATAL 1A trainer checksum changed"; exit 1; }

if [ ! -f $DATA/dataset-audit.json ]; then
  if [ "$VARIANT" = random ]; then
    $PY scripts/build_c2_random_control.py --c2 $C2 --output $DATA --seed 42
  else
    # 1A's own data, unchanged: the trainer re-verifies the sha256 against this audit.
    mkdir -p $DATA
    $PY -c "import json;a=json.load(open('$C2/dataset-audit.json'));a.update(ablation='1A-retrain',source_audit='$C2/dataset-audit.json');json.dump(a,open('$DATA/dataset-audit.json','w'),indent=2)"
  fi
fi

mkdir -p $RUN
cp $DATA/dataset-audit.json $RUN/dataset-audit.json
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
END_UTC=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)  # %e is local time; no -u
# full: pause 15 min before the limit; the trainer checkpoints and a resubmit resumes.
# smoke: ~20 min of real updates is enough to see load, loss, speed and a pause checkpoint.
if [ "$MODE" = smoke ]; then STOP=$(( $(date +%s) + 1200 )); else STOP=$(( END_UTC - 900 )); fi
[ "$MODE" = smoke ] || [ $(( STOP - $(date +%s) )) -gt 1800 ] || { echo "FATAL stop time $(date -d @$STOP) is <30 min away; check the deadline math"; exit 1; }
CONFIG=$RUN/frozen-config-$SLURM_JOB_ID.json
$PY - "$CONFIG" <<PYEOF
import json, sys
from datetime import datetime, timezone
recipe = json.load(open('$ZRUN/runs/c2-ablation-1a/frozen-config.json'))['training']
recipe['ablation'] = 'random-candidate' if '$VARIANT' == 'random' else '1A'
json.dump({'allocation': '$SLURM_JOB_ID', 'gpu_uuid': '$GPU_UUID', 'node': '$(hostname -s)', 'output': '$RUN',
           'task_count': 2091, 'stop_utc': datetime.fromtimestamp($STOP, timezone.utc).isoformat(),
           'teacher_manifest': {'actor': '/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'},
           'training': recipe, 'trainer_sha256': '$TRAINER_SHA256', 'mode': '$MODE', 'variant': '$VARIANT'},
          open(sys.argv[1], 'w'), indent=1)
PYEOF

RESUME=()
if [ -f $RUN/student/latest-checkpoint.json ] && [ ! -f $RUN/student/complete.json ]; then
  RESUME=(--resume "$($PY -c "import json;print(json.load(open('$RUN/student/latest-checkpoint.json'))['path'])")")
fi
echo "[rc] mode=$MODE node=$(hostname -s) gpu=$GPU_UUID config=$CONFIG ${RESUME[*]}"
$PY $TRAINER --config $CONFIG "${RESUME[@]}"
echo "[rc] trainer exit 0"; tail -2 $RUN/student/metrics.jsonl
