#!/bin/bash
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=bld-subset
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/build-subset-%j.out
# CPU-only (filtering, verification, hashing); a MIG slice bills 1/7 of an H200.
# Subset of a joint dataset: FILTER=success (terminal_success True; the rows RPO's SFT term trains on) or FILTER=random:F
# (a seeded random fraction F of the pairs, e.g. Round-1 replay). Validation files and panels copied unchanged.
#   SRC=sol2-Br1 OUT=sol2-Br1-succ FILTER=success sbatch tillicum/build_subset_data.sbatch
#   SRC=sol-Aall-m0.3 OUT=sol-Aall-m0.3-r50 FILTER=random:0.5 sbatch tillicum/build_subset_data.sbatch
set -euo pipefail
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data; SRC=$ROOT/${SRC:?}; OUT=$ROOT/${OUT:?}; FILTER=${FILTER:-success}
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
import random
flt = '$FILTER'; allrows = list((src / 'joint_pairs.train.jsonl').open())
if flt == 'success':
    rows = [l for l in allrows if json.loads(l).get('terminal_success') is True]; desc = 'terminal_success is True (successful-episode pairs only)'
elif flt.startswith('random:'):
    frac = float(flt.split(':')[1]); idx = sorted(random.Random(0).sample(range(len(allrows)), round(frac * len(allrows))))
    rows = [allrows[i] for i in idx]; desc = f'seeded random {frac:g} of the pairs (seed 0)'
else:
    raise SystemExit(f'unknown FILTER {flt}')
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
rep = dict(source=str(src), filter=desc, pairs=len(rows), sft=len(sft))
(out / 'filter-report.json').write_text(json.dumps(rep, indent=2) + '\n'); print('[subset]', json.dumps(rep))
PYEOF
python3 scripts/repoint_zixian_paths.py $OUT
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 2400 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[subset] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
