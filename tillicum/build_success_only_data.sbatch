#!/bin/bash
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=bld-succ
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/build-succ-%j.out
# CPU-only (filtering, verification, hashing); a MIG slice bills 1/7 of an H200.
# Successful-episode subset of a joint dataset (terminal_success True), for SFT-only ablations: the rows RPO's SFT term
# trains on when sft_successful_only is set. Validation files and panels copied unchanged.
#   SRC=sol2-Br1 OUT=sol2-Br1-succ sbatch tillicum/build_success_only_data.sbatch
set -euo pipefail
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data; SRC=$ROOT/${SRC:?}; OUT=$ROOT/${OUT:?}
PY=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/venv/bin/python
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd /gpfs/home/piotrt/OpenWebRL
[ -e $OUT ] && { echo "$OUT exists; remove it to rebuild"; exit 1; }
python3 - <<PYEOF
import json, shutil
from pathlib import Path
src, out = Path('$SRC'), Path('$OUT'); out.mkdir(parents=True)
rows = [l for l in (src / 'joint_pairs.train.jsonl').open() if json.loads(l).get('terminal_success') is True]
if len(rows) % 32 % 2: rows = rows[:-1]          # 2-rank trainer needs an even final batch
keep = {json.loads(l)['id'] for l in rows}
(out / 'joint_pairs.train.jsonl').write_text(''.join(rows))
sft = [l for l in (src / 'joint_sft.train.jsonl').open() if json.loads(l)['id'] in keep]
(out / 'joint_sft.train.jsonl').write_text(''.join(sft))
for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl', 'validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl']:
    shutil.copy2(src / name, out / name)
used = {im['path'] for l in rows for im in json.loads(l)['images']}
man = [x for x in json.loads((src / 'image-manifest.json').read_text()) if x['path'] in used or True]
(out / 'image-manifest.json').write_text(json.dumps(man) + '\n')
rep = dict(source=str(src), filter='terminal_success is True (successful-episode pairs only)', pairs=len(rows), sft=len(sft))
(out / 'filter-report.json').write_text(json.dumps(rep, indent=2) + '\n'); print('[succ]', json.dumps(rep))
PYEOF
python3 scripts/repoint_zixian_paths.py $OUT
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 2400 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[succ] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
