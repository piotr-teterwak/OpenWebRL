#!/usr/bin/env bash
#SBATCH --job-name=arm-scale-data
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (JSON + image copies); a MIG slice bills 1/7 of an H200 and CPU-only jobs queue for days.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/arm-scale-data-%j.out
#
# Selection-ARM data scaling, built with action-reward-models' own pipeline:
# data_generation/openwebrl_actor/build_selection_sft.py, unchanged, importing the pushed
# inference/openwebrl/frontier_arbiter.py, on Piotr's released states/candidates/GPT-5.5 labels.
# scripts/arm_scaling_labels.py cuts the labels file into the scale points (task-disjoint test split,
# states axis and draws-per-state axis); the builder runs once per point with --holdout 0.
set -euo pipefail
ARM_REPO=/gpfs/scrubbed/piotrt/owrl/arm-scaling/action-reward-models
RELEASE=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor
DATA=/gpfs/scrubbed/piotrt/owrl/arm-scaling/data
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
cd /gpfs/home/piotrt/OpenWebRL
echo "[arm-data] action-reward-models $(git -C $ARM_REPO rev-parse --short HEAD)"
[ -f $DATA/subsets.json ] || $PY scripts/arm_scaling_labels.py --out $DATA
# states-1 and draws-16 are the same full pool; build it once.
for p in test states-0.0625 states-0.125 states-0.25 states-0.5 states-1 draws-1 draws-2 draws-4 draws-8; do
  OUT=$DATA/sft-$p
  [ -f $OUT/dataset_info.json ] && { echo "[arm-data] $p exists"; continue; }
  rm -rf $OUT
  # The builder prepends the author's /projectnb path; the import falls through to PYTHONPATH.
  PYTHONPATH=$ARM_REPO/inference $PY $ARM_REPO/data_generation/openwebrl_actor/build_selection_sft.py \
    --states $DATA/states_local.jsonl --candidates $RELEASE/candidates_merged.jsonl \
    --labels $DATA/labels_$p.jsonl --out-dir $OUT --holdout 0 | sed "s/^/[arm-data] $p: /"
done
du -sh $DATA
