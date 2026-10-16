#!/usr/bin/env bash
# Approved 2026-10-10: 1 H200, about 45 minutes (~$0.70). No API calls, no browsers.
# Runs after job 353512 ends. (1) If the Kev-style run did not finish its final evaluations, score its best
# checkpoint on the 124 branch states and full validation. (2) Score the joint run's latest checkpoint,
# including the trained comparison pointer's own picks, on the same states.
#SBATCH --job-name=openwebrl-selection-evalonly
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-evalonly-%j.out
set -uo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-evalonly
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $SRC
common=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head $R/heads/pointwise/head.pt --eval-only)
if grep -q '"validation"' $R/kev-353512/metrics.json 2>/dev/null; then
  echo "kev-style evaluations already complete; skipping"
else
  $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 36 --kev --resume $R/kev-353512 \
      --output $R/kev-353512-eval --eval-checkpoint best && echo "kev-style evaluation complete $(date -Is)"
fi
$PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 8 --joint --resume $R/joint-353389 \
    --output $R/joint-353389-last-eval --eval-checkpoint last && echo "joint latest-checkpoint evaluation complete $(date -Is)"
