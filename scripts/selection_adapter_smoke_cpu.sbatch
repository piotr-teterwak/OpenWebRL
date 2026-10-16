#!/usr/bin/env bash
# CPU-only (no GPU request, Slurm cost $0): end-to-end smoke test of the scoring-adapter trainer
# with the real 4B actor on CPU — two optimizer steps, tiny evaluations. Validates data joins,
# processor inputs, adapter wiring, checkpoint save/reload and branch evaluation before GPU use.
#SBATCH --job-name=openwebrl-adapter-smoke
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-adapter-smoke-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-adapter
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1 HF_HOME=$R/hf-cache
cd $SRC
$PY scripts/train_selection_adapter.py --panels $R/panels/panels.jsonl \
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl \
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl \
    --init-head $R/heads/pointwise/head.pt --output $R/adapter-smoke-$SLURM_JOB_ID --device cpu \
    --panels-per-state 1 --accum 1 --max-steps 2 --eval-every 1 --eval-states 1 \
    --final-val-states 1 --branch-limit 2 --batch-size 5
echo "smoke complete $(date -Is)"
