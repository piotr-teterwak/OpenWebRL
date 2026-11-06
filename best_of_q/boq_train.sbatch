#!/usr/bin/env bash
#SBATCH --job-name=boq-train
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-train-%j.out
# Train one Best-of-Q Q-function (train_q.py: paper IQL settings, tau 0.7 for Qwen-scale failure-heavy data, 50k steps,
# best held-out V(s0) AUC checkpoint) on the union of the listed run directories. One MIG slice, 2 CPUs (billing ~1/7).
#   NAME=q1 RUNS="seed-s1 seed-s2 seed-s3 seed-s4 seed-s5" sbatch boq_train.sbatch
#   MAX_STATES=1380: train on a seeded subset of whole episodes holding >= 1,380 executed states (data-matched)
set -euo pipefail
NAME=${NAME:?}; RUNS=${RUNS:?}; ROOT=${BOQ_ROOT:-/gpfs/scrubbed/piotrt/owrl/best_of_q_v2}
ARM_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export BOQ_EMBEDDER=${BOQ_EMBEDDER:-/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct}
export OMP_NUM_THREADS=2
cd /gpfs/home/piotrt/OpenWebRL
DIRS=(); for r in $RUNS; do [ -s $ROOT/runs/$r/boq/emb/tasks.npy ] || { echo "FATAL $r has no embeddings"; exit 1; }; DIRS+=($ROOT/runs/$r); done
DATA=$ROOT/train/$NAME; rm -rf $DATA; mkdir -p $DATA
$ARM_PY best_of_q/merge_runs.py --out $DATA ${MAX_STATES:+--max-states $MAX_STATES --seed ${SUBSAMPLE_SEED:-0}} "${DIRS[@]}"
$ARM_PY best_of_q/train_q.py --rows $DATA/rows.jsonl --emb $DATA/emb --out $ROOT/q/$NAME --tau ${TAU:-0.7} --steps ${STEPS:-50000}
rm -rf $DATA/emb   # merged copies of per-run embeddings; the per-run originals stay
# train_q saves q_function_best.pt only when the held-out V(s0) AUC is defined; never leave the next cycle without a Q
[ -f $ROOT/q/$NAME/q_function_best.pt ] || { echo "[boq] $NAME: no AUC-selected checkpoint, using the final one"; cp $ROOT/q/$NAME/q_function.pt $ROOT/q/$NAME/q_function_best.pt; }
echo "[boq] $NAME trained on: $RUNS -> $ROOT/q/$NAME/q_function_best.pt"
