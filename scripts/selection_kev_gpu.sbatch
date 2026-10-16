#!/usr/bin/env bash
# Requires explicit approval: 1 H200 x 3 hours = 3 GPU-hours, estimated $2.70. No API calls, no browsers.
# Kev-style decision pass on the actor's weights: packed [prompt + screenshot][panel candidates, <decide>]...
# through all 36 layers with LoRA on every language layer, isolated candidates/panels with shared positions,
# pure pointer readout, no warm start. Stages: smoke at full panel size incl. resume (aborts on failure);
# training with a fixed 250-state probe every 150 steps; best-checkpoint full validation and 124-branch
# evaluation. Retries must fit the same 3 GPU-hour total.
#SBATCH --job-name=openwebrl-selection-kev
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-kev-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-kevstyle
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
args=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head $R/heads/pointwise/head.pt --top-layers 36 --kev --grad-checkpoint)
smoke=(--panels-per-state 4 --accum 1 --eval-every 1 --eval-states 1 --final-val-states 1 --branch-limit 2)
S=$R/kev-smoke-$SLURM_JOB_ID
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-a --max-steps 1 > $S.log 2>&1
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-b --max-steps 2 --resume $S-a >> $S.log 2>&1
grep -q '"event": "resumed".*"step": 1' $S.log
nvidia-smi --query-gpu=memory.used --format=csv
echo "kev smoke and resume passed $(date -Is)"
$PY scripts/train_selection_adapter.py "${args[@]}" --output $R/kev-$SLURM_JOB_ID \
    --lr 1e-4 --pointer-lr 1e-3 --eval-states 250 --eval-every 150 --train-seconds 8400
echo "kev complete $(date -Is)"
