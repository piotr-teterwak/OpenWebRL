#!/usr/bin/env bash
#SBATCH --job-name=arm-scale-train
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --exclude=g013,g021
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/arm-scale-train-%j.out
#
# Train one Selection-ARM scale point with action-reward-models' training/llamafactory/arm_lora.yaml,
# then merge it with arm_merge.yaml. Every hyperparameter comes from the repo yaml; overridden are
# only paths, the batch layout (4 GPUs x 2 x accumulation 2 = the yaml's effective batch 16 on
# 1 GPU x 2 x 8), step checkpoints for resuming, and report_to. ~5 samples/s on 4 H200s
# (Zixian's runs of the same recipe), so the full 35.9k-draw point is ~4 h; size --time to the point.
#   sbatch --time=06:00:00 tillicum/arm_scaling_train.sbatch states-1
# SMOKE=1: 32 samples, 2 steps, debug QoS, no merge.
set -euo pipefail
POINT=${1:?scale point, e.g. states-0.25 or draws-2}
ARM_REPO=/gpfs/scrubbed/piotrt/owrl/arm-scaling/action-reward-models
DATA=/gpfs/scrubbed/piotrt/owrl/arm-scaling/data/sft-$POINT
RUN=/gpfs/scrubbed/piotrt/owrl/arm-scaling/runs/$POINT${SMOKE:+-smoke}
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
V=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv-refresh
R=/gpfs/scrubbed/zixianma/openwebrl-runtime
[ -f $DATA/dataset_info.json ] || { echo "FATAL no dataset at $DATA"; exit 1; }
mkdir -p $RUN
NGPU=$(nvidia-smi -L | wc -l)
# Zixian's working environment for this recipe on tillicum (sft-actions-20261021/arm-lora).
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=$NGPU \
  MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000)) TRITON_CACHE_DIR=$RUN/triton-cache WANDB_MODE=disabled \
  CPATH=$R/src/python-headers/Include:$R/src/python-headers PATH=$R/arm-reproduction/refresh-torch28/bin:$V/bin:$PATH
$V/bin/python - <<PYEOF
import yaml
c = yaml.safe_load(open('$ARM_REPO/training/llamafactory/arm_lora.yaml'))
assert c['per_device_train_batch_size'] * c['gradient_accumulation_steps'] == 16
c.update(model_name_or_path='$BASE', dataset_dir='$DATA', media_dir='$DATA', output_dir='$RUN/adapter',
         per_device_train_batch_size=2, gradient_accumulation_steps=16 // (2 * $NGPU),
         save_strategy='steps', save_steps=300, save_total_limit=2, overwrite_output_dir=False,
         save_only_model=False, report_to='none', run_name='arm-scale-$POINT', seed=42)
if '${SMOKE:-}': c.update(max_samples=32, max_steps=2, save_strategy='no')
yaml.safe_dump(c, open('$RUN/arm_lora.yaml', 'w'), sort_keys=False)
m = yaml.safe_load(open('$ARM_REPO/training/llamafactory/arm_merge.yaml'))
m.update(model_name_or_path='$BASE', adapter_name_or_path='$RUN/adapter', export_dir='$RUN/merged')
yaml.safe_dump(m, open('$RUN/arm_merge.yaml', 'w'), sort_keys=False)
PYEOF
echo "[arm-train] $POINT gpus=$NGPU repo=$(git -C $ARM_REPO rev-parse --short HEAD) train=$(python3 -c "import json;print(len(json.load(open('$DATA/owrl_selection_train.json'))))")"
extra=()
last=$(ls -d $RUN/adapter/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[ -n "$last" ] && extra=(resume_from_checkpoint=$last) && echo "[arm-train] resuming from $last"
cd $RUN
llamafactory-cli train $RUN/arm_lora.yaml "${extra[@]}"
[ -n "${SMOKE:-}" ] && { echo SMOKE_TRAIN_DONE; exit 0; }
test -s $RUN/adapter/adapter_model.safetensors
llamafactory-cli export $RUN/arm_merge.yaml
test -s $RUN/merged/config.json && echo TRAIN_MERGE_DONE
