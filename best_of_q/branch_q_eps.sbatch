#!/usr/bin/env bash
#SBATCH --job-name=boq-branch-eps
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-branch-eps-%j.out
# Same states and embeddings as branch_q.sbatch; epsilon-greedy simulation: one executed candidate per state (eps 0.5).
set -euo pipefail
D=/gpfs/scrubbed/piotrt/owrl/best_of_q_branch; ARM_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export BOQ_EMBEDDER=/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct OMP_NUM_THREADS=2
cd /gpfs/home/piotrt/OpenWebRL/best_of_q && $ARM_PY train_branch_q.py --rows $D/data/rows.jsonl --emb $D/data/emb --out $D/q-eps0.5 --seed 0 --epsilon 0.5
