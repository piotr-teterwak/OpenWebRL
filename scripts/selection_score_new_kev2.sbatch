#!/usr/bin/env bash
# Approved 2026-10-10: 1 H200, 2 CPUs, 120 GiB, 30 minutes (~$0.45). No API calls, no browsers.
# After job 353914 ends: score the Kev-style continuation's best checkpoint on the new branching states
# (pilot32, fresh64, fresh64 successor) with the same evaluation path as selection_score_new_branch.sbatch.
# 353914 scores the original 124 states itself.
#SBATCH --job-name=openwebrl-selection-score-kev2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-score-kev2-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-score-new
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
KEV=${KEV:-$R/kev2-353914}
O=$R/score-new-kev2-$SLURM_JOB_ID
mkdir -p $O
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
for lineage in pilot32 fresh64 fresh64succ; do
    dir=$R/branch-new/$lineage
    $PY scripts/train_selection_adapter.py --panels $R/panels/panels.jsonl --branch-panels $dir/panels.jsonl \
        --branch-states $dir/states.jsonl --init-head $R/heads/pointwise/head.pt --top-layers 36 --kev \
        --resume $KEV --output $O/$lineage --eval-only --eval-checkpoint best --final-val-states 0 \
        > $O/$lineage.log 2>&1
    echo "$lineage scored $(date -Is)"
done
echo "score-kev2 complete $(date -Is)" | tee $O/complete.txt
