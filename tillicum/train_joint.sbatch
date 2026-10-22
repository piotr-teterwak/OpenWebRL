#!/usr/bin/env bash
#SBATCH --job-name=owrl-joint
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
# Zixian's joint runs asked for 16 CPUs but used a mean of 3.5-3.8 busy cores
# (287447, 287477); the trainer pins 4 torch threads per rank. MaxRSS 63-74G.
# Memory comes at 30G/CPU here, so 8 CPUs = 240G.
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
# Her sft job ran 1h54m including a 300-task eval; dpo trained 2h02m on top of
# a reference pass (~0.5-1h) that her r2 reused from r1 and we cannot. Pass
# --time on the command line: 03:00:00 for sft, 05:00:00 for dpo.
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/joint-%j.out
#
# tillicum port of klone/train_joint.sbatch: joint-data SFT or DPO with the
# CURRENT trainer (scripts/train_arm_joint_ddp.py), on Zixian's existing
# joint-data-v2. Training only; evaluation is a separate concurrent job.
#
#   OBJECTIVE=sft sbatch tillicum/train_joint.sbatch
#   OBJECTIVE=dpo sbatch --time=05:00:00 tillicum/train_joint.sbatch
# Resubmitting the same OBJECTIVE resumes from the latest checkpoint.
#
# Sweeps: LR overrides the learning rate (default 1e-5), SFT_WEIGHT adds RPO's
# length-normalised NLL on the chosen response to DPO, and RUN names the run
# directory so cells do not collide:
#   OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 RUN=rpo-a1-lr5e-5 sbatch --time=05:00:00 tillicum/train_joint.sbatch
# BETA overrides DPO's beta (default 0.1), e.g. BETA=0.5 RUN=rpo-a1-lr5e-5-b0.5.
set -euo pipefail
OBJECTIVE="${OBJECTIVE:?set OBJECTIVE=sft|dpo|dpo_action}"
case "$OBJECTIVE" in sft|dpo|dpo_action) ;; *) echo "bad OBJECTIVE=$OBJECTIVE"; exit 1;; esac
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction

# Her env script with every writable cache moved to our scratch.
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=4
# No W&B key here; keep the run log local (sync later with `wandb sync` if wanted).
export WANDB_MODE=offline
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
PY=$ZRUN/venv/bin/python

export OWRL_ACTOR=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
export OWRL_JOINT_DATA=${OWRL_JOINT_DATA:-$ZRUN/joint-data-v2-20260910}   # e.g. /gpfs/scrubbed/piotrt/owrl/joint-data/k2
export OWRL_RUNTIME=/gpfs/scrubbed/piotrt/owrl/joint-retrain
cd $REPO

# The config is frozen on first submission and reused on resume: the trainer
# keys its reference cache to the config's hash, so rewriting it would orphan
# the cache. frozen_config() pins hashes of the trainer code, data and model
# metadata, so a code change between submissions fails loudly instead of mixing.
CONFIG=$($PY - <<PYEOF
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from train_arm_joint_ddp import frozen_config
config = frozen_config('$OBJECTIVE')
if '${RUN:-}':
    config['output'] = str(Path(config['output']).parent / '${RUN:-}')
if '${LR:-}': config['learning_rate'] = float('${LR:-}')
if '${SFT_WEIGHT:-}': config['auxiliary_sft_weight'] = float('${SFT_WEIGHT:-}')
if '${BETA:-}': config['beta'] = float('${BETA:-}')
if '${EXCLUDE_EOT:-}' == '1': config['dpo_exclude_end_of_turn'] = True
path = Path(config['output']) / 'frozen-config.json'
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1) + '\n')
print(path)
PYEOF
)
RUN=$(dirname $CONFIG)
END_UTC=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)  # %e is local time; no -u
STOP=$(( END_UTC - 900 ))
[ $(( STOP - $(date +%s) )) -gt 1800 ] || { echo "FATAL stop time $(date -d @$STOP) is <30 min away; check the deadline math"; exit 1; }
RESUME=""
if [ -f $RUN/student/latest-checkpoint.json ] && [ ! -f $RUN/student/complete.json ]; then RESUME=--resume; fi
echo "[joint] objective=$OBJECTIVE node=$(hostname -s) config=$CONFIG stop=$(date -u -d @$STOP +%FT%TZ) $RESUME"
echo "[joint] code=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts | wc -l)"

$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_arm_joint_ddp.py --config $CONFIG --stop-unix $STOP $RESUME
cat $RUN/training-status.json; echo
