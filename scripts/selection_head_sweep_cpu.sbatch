#!/usr/bin/env bash
# CPU-only (no GPU request, Slurm cost $0): frozen-feature selection-head sweep over layers,
# pools and head size, on features already extracted by job 351647. Checkpoints are chosen by
# held-out GPT-5.5 agreement; Luna agreement and branch outcomes are reported, never selected on.
#SBATCH --job-name=openwebrl-selection-sweep
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-sweep-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-sweep
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export PYTHONPATH=$SRC OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
cd $SRC
common=(--panels $R/panels/panels.jsonl --features $R/features-piotr
        --branch-panels $R/branch124/panels.jsonl --branch-features $R/features-branch124
        --external-picks $R/piotr_arm.jsonl --reference-picks $R/branch124/luna_before.jsonl
        --device cpu --epochs 10 --pointwise)
# name | extra args
configs=(
  "L09|--layers 9"
  "L18|--layers 18"
  "L27|--layers 27"
  "L36|--layers 36"
  "L09-36|--layers 9,18,27,36"
  "L18-36_action|--layers 18,27,36 --pools action_mean"
  "L18-36_thought|--layers 18,27,36 --pools thought_mean"
  "L18-36_last|--layers 18,27,36 --pools last"
  "L18-36_small|--layers 18,27,36 --d-model 128 --dropout 0.3 --weight-decay 0.1"
)
for entry in "${configs[@]}"; do
  name=${entry%%|*}; extra=${entry#*|}
  out=$R/sweep/$name
  [[ -f $out/metrics.json ]] && { echo "skip $name (done)"; continue; }
  echo "start $name $(date -Is)"
  # shellcheck disable=SC2086
  $PY scripts/train_selection_head.py "${common[@]}" --output $out $extra > $R/logs/sweep-$name-$SLURM_JOB_ID.log 2>&1
  echo "done $name $(date -Is)"
done
echo "sweep complete $(date -Is)"
