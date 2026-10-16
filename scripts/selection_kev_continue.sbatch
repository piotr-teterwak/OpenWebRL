#!/usr/bin/env bash
# Approved 2026-10-10: 1 H200, 2 CPUs, 120 GiB, 6 hours (~$5.40), including retries. No API calls, no browsers.
# Continues the Kev-style decision pass (job 353512) from its latest checkpoint (step 244) with unchanged
# settings: a 250-state probe every 150 steps, best and latest checkpoints saved, then the best checkpoint
# scored on the 124 branch states first and full validation second. A short smoke run of the same resume
# path goes first and aborts the job on failure.
#SBATCH --job-name=openwebrl-selection-kev2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-kev2-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-kev2
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
args=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head $R/heads/pointwise/head.pt --top-layers 36 --kev --grad-checkpoint
    --lr 1e-4 --pointer-lr 1e-3 --resume $R/kev-353512)
S=$R/kev2-smoke-$SLURM_JOB_ID
$PY scripts/train_selection_adapter.py "${args[@]}" --output $S --max-steps 245 \
    --panels-per-state 4 --accum 1 --eval-every 1 --eval-states 1 --final-val-states 1 --branch-limit 2 > $S.log 2>&1
grep -q '"event": "resumed".*"step": 244' $S.log
echo "kev continuation smoke passed $(date -Is)"
$PY scripts/train_selection_adapter.py "${args[@]}" --output $R/kev2-$SLURM_JOB_ID \
    --eval-states 250 --eval-every 150 --train-seconds ${TRAIN_SECONDS:-19200}
echo "kev continuation complete $(date -Is)"
