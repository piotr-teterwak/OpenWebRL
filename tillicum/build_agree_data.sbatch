#!/usr/bin/env bash
#SBATCH --job-name=agree-data
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (filter + verification); a MIG slice bills 1/7 of an H200. MIG allows at most 2 CPUs per GPU.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/agree-data-%j.out
#
# Keep only the pairs whose chosen response a second selector also picks (scripts/relabel_with_selector.py
# labels over the same stored panel), e.g. round-2 ARM pairs where gpt-6-luna agrees with the ARM.
# Validation is the source's (joint-data/matched's) unchanged.
#   SRC=round2-b0.3s1 LABELS=<labels.jsonl> OUT=round2-b0.3s1-agree sbatch tillicum/build_agree_data.sbatch
set -euo pipefail
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data
SRC=$ROOT/${SRC:?}; OUT=$ROOT/${OUT:?}; LABELS=${LABELS:?}
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source /gpfs/home/piotrt/OpenWebRL/scripts/h200_env.sh
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd /gpfs/home/piotrt/OpenWebRL
[ -e $OUT ] && { echo "$OUT exists; remove it to rebuild"; exit 1; }
python3 - <<PYEOF
import json, shutil
from pathlib import Path
src, out = Path('$SRC'), Path('$OUT'); out.mkdir(parents=True)
labels = {r['id']: r['selected_index'] for r in map(json.loads, Path('$LABELS').read_text().splitlines())}
def state_id(pid):  # round-2 ids are C2r2:<task>:<turn>; labels use the collection's C2:<task>:<turn>
    return 'C2:' + pid.split(':', 1)[1]
pairs = [json.loads(l) for l in (src / 'joint_pairs.train.jsonl').open()]
keep = sorted((p for p in pairs if labels.get(state_id(p['id'])) == p['chosen_index']), key=lambda p: p['id'])
dropped = keep.pop()['id'] if len(keep) % 32 % 2 else None   # 2-rank trainer needs an even final batch
ids = {p['id'] for p in keep}
with (out / 'joint_pairs.train.jsonl').open('w') as f:
    for line in (src / 'joint_pairs.train.jsonl').open():
        if json.loads(line)['id'] in ids: f.write(line)
with (out / 'joint_sft.train.jsonl').open('w') as f:
    for line in (src / 'joint_sft.train.jsonl').open():
        if json.loads(line)['id'] in ids: f.write(line)
for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl', 'image-manifest.json']: shutil.copy2(src / name, out / name)
report = dict(source=str(src), labels='$LABELS', source_pairs=len(pairs), kept_pairs=len(keep),
              filter='second selector picks the same candidate as the pair\'s chosen', dropped_for_even_final_batch=dropped)
(out / 'filter-report.json').write_text(json.dumps(report, indent=2) + '\n'); print('[agree]', json.dumps(report))
PYEOF
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 1800 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[agree] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
