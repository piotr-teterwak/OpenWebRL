#!/usr/bin/env bash
#SBATCH --job-name=boq-branch-thought
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/boq-branch-thought-%j.out
# Branch simulation (branch_sim.py) with each candidate's own reasoning in its action text (branch_thoughts.py), as the
# paper's action embedding. State embeddings are reused from sim/emb (same rows, same order); only texts re-embedded.
set -euo pipefail
S=/gpfs/scrubbed/piotrt/owrl/best_of_q_branch/sim; D=/gpfs/scrubbed/piotrt/owrl/best_of_q_branch/sim-thought
ARM_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export BOQ_EMBEDDER=/gpfs/scrubbed/piotrt/owrl/models/Qwen2.5-VL-3B-Instruct OMP_NUM_THREADS=2 BOQ_TEXT_BATCH=8
cd /gpfs/home/piotrt/OpenWebRL
python3 best_of_q/branch_thoughts.py --rows $S/rows.jsonl --out $D/rows.jsonl
mkdir -p $D/emb; for f in $S/emb/states-*.npy; do ln -sf $f $D/emb/; done
$ARM_PY best_of_q/embed.py texts --rows $D/rows.jsonl --out $D/emb
cd best_of_q && $ARM_PY branch_sim.py --rows $D/rows.jsonl --emb $D/emb --out $D/q --cycles 4 --epsilon 0.5 --seed 0
