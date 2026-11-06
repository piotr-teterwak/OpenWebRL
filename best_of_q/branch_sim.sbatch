#!/usr/bin/env bash
#SBATCH --job-name=boq-branch-sim
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-branch-sim-%j.out
# Offline Best-of-Q iterative simulation on Zixian's branch states (branch_sim.py): embed both folds' rows with the
# frozen Qwen2.5-VL-3B, then round 0 eps-greedy + 4 argmax-Q rounds, held-out fold-4 scoring after each round.
set -euo pipefail
D=/gpfs/scrubbed/piotrt/owrl/best_of_q_branch/sim; ARM_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export BOQ_EMBEDDER=/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct OMP_NUM_THREADS=2
cd /gpfs/home/piotrt/OpenWebRL
[ -s $D/emb/tasks.npy ] || { $ARM_PY best_of_q/embed.py states --rows $D/rows.jsonl --out $D/emb --batch 8; $ARM_PY best_of_q/embed.py texts --rows $D/rows.jsonl --out $D/emb; }
cd best_of_q && $ARM_PY branch_sim.py --rows $D/rows.jsonl --emb $D/emb --out $D/q --cycles 4 --epsilon 0.5 --seed 0
