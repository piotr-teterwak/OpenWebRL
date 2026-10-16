#!/usr/bin/env bash
# Uses the remaining approved adapter budget (1 H200; approved 4 h total, 1 h 47 m 40 s used by 353150/353205).
# Stages: (1) smoke, including a resume from a saved last/ checkpoint; (2) adapter training with the head
# frozen for the first 100 steps, a fixed 250-state validation probe every 150 steps, latest and best
# checkpoints saved; (3) best-checkpoint full validation and 124-branch evaluation. No API calls or browsers.
#SBATCH --job-name=openwebrl-selection-adapter
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=02:10:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-adapter-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-adapter-v2
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
args=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head $R/heads/pointwise/head.pt --top-layers 8)
smoke=(--panels-per-state 1 --accum 1 --eval-every 1 --eval-states 1 --final-val-states 1 --branch-limit 2)
S=$R/adapter-smoke-$SLURM_JOB_ID
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-a --max-steps 1 > $S.log 2>&1
$PY scripts/train_selection_adapter.py "${args[@]}" "${smoke[@]}" --output $S-b --max-steps 2 --resume $S-a >> $S.log 2>&1
grep -q '"event": "resumed".*"step": 1' $S.log
echo "smoke and resume passed $(date -Is)"
$PY scripts/train_selection_adapter.py "${args[@]}" --output $R/adapter-$SLURM_JOB_ID \
    --lr 5e-5 --head-lr 1e-5 --freeze-head-steps 100 --eval-states 250 --eval-every 150 --train-seconds 6300
echo "adapter complete $(date -Is)"
