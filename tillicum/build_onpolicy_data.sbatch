#!/usr/bin/env bash
#SBATCH --job-name=onpolicy-data
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (hashing + verification); a MIG slice bills 1/7 of an H200.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/onpolicy-data-%j.out
#
# Dan's on-policy pairs (HF davidan03/openwebrl-arm-onpolicy-pairs, downloaded to joint-data/onpolicy-hf
# on the login node) -> joint-data/onpolicy: absolute image paths, chosen-only SFT view, held-out
# validation from joint-data/matched, verification and a data-driven train-config.json.
set -euo pipefail
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data; OUT=$ROOT/onpolicy
cd /gpfs/home/piotrt/OpenWebRL
[ -d $OUT ] && { echo "$OUT exists; remove it to rebuild"; exit 1; }
$PY scripts/localize_onpolicy_pairs.py --src $ROOT/onpolicy-hf --val-src $ROOT/matched --out $OUT
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 1800 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[onpolicy] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
