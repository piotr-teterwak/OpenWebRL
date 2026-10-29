#!/usr/bin/env bash
#SBATCH --job-name=combined-data
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (concatenation + verification); a MIG slice bills 1/7 of an H200.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/combined-data-%j.out
#
# Round-1 (Dan's on-policy pairs, joint-data/onpolicy) + round-2 pairs (joint-data/round2-b0.3s1) in one
# training set, for a single DPO pass from base instead of iterating. Ids are already distinct
# (C2:, C2p2:, C2r2:); validation is onpolicy's (= joint-data/matched's) unchanged.
#   A=onpolicy B=round2-b0.3s1 OUT=onpol-plus-r2 sbatch tillicum/build_combined_data.sbatch
set -euo pipefail
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data
A=$ROOT/${A:-onpolicy}; B=$ROOT/${B:-round2-b0.3s1}; OUT=$ROOT/${OUT:-onpol-plus-r2}
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd /gpfs/home/piotrt/OpenWebRL
[ -e $OUT ] && { echo "$OUT exists; remove it to rebuild"; exit 1; }
python3 - <<PYEOF
import json, shutil
from pathlib import Path
a, b, out = Path('$A'), Path('$B'), Path('$OUT')
out.mkdir(parents=True)
ids, n = set(), {}
for name in ['joint_pairs.train.jsonl', 'joint_sft.train.jsonl']:
    rows = [l for src in (a, b) for l in (src / name).open()]
    if name.startswith('joint_pairs'):
        ids = [json.loads(l)['id'] for l in rows]
        assert len(ids) == len(set(ids)), 'duplicate pair ids across the two sets'
        if len(rows) % 32 % 2: rows = rows[:-1]          # 2-rank trainer needs an even final batch
        keep = {json.loads(l)['id'] for l in rows}
    else:
        rows = [l for l in rows if json.loads(l)['id'] in keep]
    (out / name).write_text(''.join(rows)); n[name] = len(rows)
for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl']: shutil.copy2(a / name, out / name)
manifest = {x['path']: x for src in (a, b) for x in json.loads((src / 'image-manifest.json').read_text())}
(out / 'image-manifest.json').write_text(json.dumps(list(manifest.values())) + '\n')
report = dict(sources=[str(a), str(b)], counts=n, validation_from=str(a))
(out / 'filter-report.json').write_text(json.dumps(report, indent=2) + '\n'); print('[combined]', json.dumps(report))
PYEOF
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 2400 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[combined] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
