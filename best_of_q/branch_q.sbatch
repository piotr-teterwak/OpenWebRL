#!/usr/bin/env bash
#SBATCH --job-name=boq-branch
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-branch-%j.out
# Best-of-Q Q-network on Zixian's 1,367 branch training states (branch_rows.py: one sampled continuation outcome per
# candidate), embedded with the frozen Qwen2.5-VL-3B, trained with train_branch_q.py. One MIG slice, 2 CPUs.
set -euo pipefail
D=/gpfs/scrubbed/piotrt/owrl/best_of_q_branch; ARM_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export BOQ_EMBEDDER=/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct OMP_NUM_THREADS=2
cd /gpfs/home/piotrt/OpenWebRL
$ARM_PY best_of_q/embed.py states --rows $D/data/rows.jsonl --out $D/data/emb --batch 8
$ARM_PY best_of_q/embed.py texts --rows $D/data/rows.jsonl --out $D/data/emb
cd best_of_q && $ARM_PY train_branch_q.py --rows $D/data/rows.jsonl --emb $D/data/emb --out $D/q --seed 0
