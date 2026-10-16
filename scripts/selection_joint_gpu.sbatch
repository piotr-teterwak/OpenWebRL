#!/usr/bin/env bash
# Requires explicit approval: 1 H200 x 3 hours = 3 GPU-hours, estimated $2.70. No API calls, no browsers.
# Kev-style joint top-layer scorer: the frozen actor's lower 28 layers per candidate, then the LoRA-adapted
# top 8 layers over [candidates, <decide>] (isolated candidates, shared positions) with a pointer readout
# added to the warm-started head. Stages: smoke incl. resume (aborts on failure); training with the head
# frozen for 100 steps and a fixed 250-state probe every 150 steps; best-checkpoint full validation and
# 124-branch evaluation. Retries must fit the same 3 GPU-hour total.
#SBATCH --job-name=openwebrl-selection-joint
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-joint-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-joint
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
args=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head $R/heads/pointwise/head.pt --top-layers 8 --joint)
smoke=(--panels-per-state 1 --accum 1 --eval-every 1 --eval-states 1 --final-val-states 1 --branch-limit 2)
S=$R/joint-smoke-$SLURM_JOB_ID
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-a --max-steps 1 > $S.log 2>&1
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-b --max-steps 2 --resume $S-a >> $S.log 2>&1
grep -q '"event": "resumed".*"step": 1' $S.log
echo "joint smoke and resume passed $(date -Is)"
$PY scripts/train_selection_adapter.py "${args[@]}" --output $R/joint-$SLURM_JOB_ID \
    --lr 5e-5 --head-lr 1e-5 --pointer-lr 1e-3 --freeze-head-steps 100 \
    --eval-states 250 --eval-every 150 --train-seconds ${TRAIN_SECONDS:-8400}
echo "joint complete $(date -Is)"
