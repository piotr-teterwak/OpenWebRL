#!/usr/bin/env bash
# Requires explicit approval: 1 H200 x 4 hours = 4 GPU-hours, estimated $3.60. No API calls, no browsers.
# Stages: (1) GPU smoke run of the adapter trainer, aborting the job on failure; (2) frozen-feature
# head sweep in the background (GPU, small); (3) top-layer scoring adapter + head, warm-started from
# the frozen-feature head, then full validation and 124-branch evaluation. Every worker is awaited.
# Retries must fit the same 4 GPU-hour total. Two CPUs fit the partition's idle-CPU slots.
#SBATCH --job-name=openwebrl-selection-adapter
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-adapter-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-adapter
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
adapter_args=(--panels $R/panels/panels.jsonl
    --branch-panels $R/branch124/panels.jsonl --branch-states $R/branch124/states.jsonl
    --reference-picks $R/branch124/luna_before.jsonl --external-picks $R/piotr_arm.jsonl
    --init-head ${INIT_HEAD:-$R/heads/pointwise/head.pt} --top-layers ${TOP_LAYERS:-8})

# 1. Smoke: two optimizer steps and tiny evaluations; set -e aborts the job on failure.
$PY scripts/train_selection_adapter.py "${adapter_args[@]}" --output $R/adapter-smoke-$SLURM_JOB_ID \
    --panels-per-state 1 --accum 1 --max-steps 2 --eval-every 1 --eval-states 1 \
    --final-val-states 1 --branch-limit 2 > $R/logs/adapter-smoke-$SLURM_JOB_ID.log 2>&1
echo "smoke passed $(date -Is)"

# 2. Frozen-feature sweep in the background; checkpoints chosen on GPT-5.5 validation only.
(
  common=(--panels $R/panels/panels.jsonl --features $R/features-piotr
          --branch-panels $R/branch124/panels.jsonl --branch-features $R/features-branch124
          --external-picks $R/piotr_arm.jsonl --reference-picks $R/branch124/luna_before.jsonl
          --epochs 10 --pointwise)
  for entry in "L09|--layers 9" "L18|--layers 18" "L27|--layers 27" "L36|--layers 36" \
               "L09-36|--layers 9,18,27,36" "L18-36_action|--layers 18,27,36 --pools action_mean" \
               "L18-36_thought|--layers 18,27,36 --pools thought_mean" "L18-36_last|--layers 18,27,36 --pools last" \
               "L18-36_small|--layers 18,27,36 --d-model 128 --dropout 0.3 --weight-decay 0.1"; do
    name=${entry%%|*}; extra=${entry#*|}
    [[ -f $R/sweep/$name/metrics.json ]] && continue
    # shellcheck disable=SC2086
    nice -n 10 $PY scripts/train_selection_head.py "${common[@]}" --output $R/sweep/$name $extra \
        > $R/logs/sweep-$name-$SLURM_JOB_ID.log 2>&1 || echo "sweep $name failed"
    echo "sweep $name done $(date -Is)"
  done
) &
sweep_pid=$!

# 3. Adapter training, then best-checkpoint validation and branch evaluation.
$PY scripts/train_selection_adapter.py "${adapter_args[@]}" --output $R/adapter-$SLURM_JOB_ID \
    --train-seconds ${TRAIN_SECONDS:-8100} --lr ${LR:-1e-4} --head-lr ${HEAD_LR:-1e-4}
wait $sweep_pid
echo "adapter complete $(date -Is)"
