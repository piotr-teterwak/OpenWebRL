#!/usr/bin/env bash
# Requires explicit approval: 1 H200 x 3 hours = 3 GPU-hours, estimated $2.70. No API calls, no browsers.
# Selection head phase 1: frozen-SFT features (branching eval + Piotr panels), Piotr SelectionARM on
# the branching states, then listwise and pointwise heads. Every worker is awaited before exit.
#SBATCH --job-name=openwebrl-selection-head
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-head-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source            # frozen copy made at submission; the job never reads the live checkout
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export HF_HOME=$R/hf-cache PYTHONPATH=$SRC TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p $R/logs
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $R/logs/gpu-$SLURM_JOB_ID.txt

# 1. Branching-state features; aborts unless cached features match full forwards.
$PY scripts/extract_selection_head_features.py --panels $R/branch124/panels.jsonl \
    --states $R/branch124/states.jsonl --output $R/features-branch124 --chunk-states 124

# 2. Released SelectionARM on the same states, sharing the GPU with step 3.
$PY scripts/score_selection_arm_offline.py --panels $R/branch124/panels.jsonl \
    --output $R/piotr_arm.jsonl > $R/logs/piotr-arm-$SLURM_JOB_ID.log 2>&1 &
arm_pid=$!

# 3. Teacher-labeled Piotr panels (resumable chunks).
$PY scripts/extract_selection_head_features.py --panels $R/panels/panels.jsonl --output $R/features-piotr
wait $arm_pid

# 4. Heads: joint comparison versus one-candidate-at-a-time; branching states are evaluation only.
for variant in listwise pointwise; do
    extra=""; [[ $variant == pointwise ]] && extra="--pointwise"
    $PY scripts/train_selection_head.py --panels $R/panels/panels.jsonl --features $R/features-piotr \
        --branch-panels $R/branch124/panels.jsonl --branch-features $R/features-branch124 \
        --external-picks $R/piotr_arm.jsonl --output $R/heads/$variant $extra \
        > $R/logs/train-$variant-$SLURM_JOB_ID.log 2>&1
done
echo "phase1 complete $(date -Is)" | tee $R/logs/complete-$SLURM_JOB_ID.txt
