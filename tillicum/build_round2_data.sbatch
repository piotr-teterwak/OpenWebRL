#!/usr/bin/env bash
#SBATCH --job-name=round2-data
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (tokenizing new contexts, hashing, verification); a MIG slice bills 1/7 of an H200.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/round2-data-%j.out
#
# Round-2 on-policy pairs, same recipe as Dan's (davidan03/openwebrl-arm-onpolicy-pairs): from the
# sharded ARM-selection collection, chosen = executed ARM pick in a successful episode, rejected =
# another candidate at that state, equal tool-call counts only. Validation is joint-data/matched's,
# unchanged, so held-out numbers stay comparable; the builder's own split keeps validation task
# groups out of train, and verify checks the groups are disjoint.
#   NAME=round2-b0.3s1 sbatch tillicum/build_round2_data.sbatch
set -euo pipefail
NAME=${NAME:?set NAME, the collection prefix under c2-collect (shards are \$NAME-shard1..N)}
NSHARDS=${NSHARDS:-4}; SH=$(seq -s " " 1 $NSHARDS)
REPO=/gpfs/home/piotrt/OpenWebRL
ZRUN=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction
PY=$ZRUN/venv/bin/python
C=/gpfs/scrubbed/piotrt/owrl/c2-collect
ROOT=/gpfs/scrubbed/piotrt/owrl/joint-data
MERGED=$C/$NAME; FULL=$ROOT/$NAME-full; OUT=$ROOT/$NAME
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd $REPO
[ -e $OUT ] && { echo "$OUT exists; remove it to rebuild"; exit 1; }

# 1. One C2 root over the shards: the builder reads only dataset-audit.json -> training.jsonl,
# whose rows carry absolute turn/attempt paths, so concatenating the shard datasets is enough.
mkdir -p $MERGED
for k in $SH; do
  S=$C/$NAME-shard$k
  [ -f $S/dataset-audit.json ] && [ "$(python3 -c "import json;print(json.load(open('$S/dataset-audit.json'))['complete_collection'])")" = True ] \
    || { echo "FATAL shard $k incomplete or not built: $S"; exit 1; }
done
for k in $SH; do cat $C/$NAME-shard$k/training.jsonl; done > $MERGED/training.jsonl
python3 - <<PYEOF
import hashlib, json
from pathlib import Path
m = Path('$MERGED'); d = m / 'training.jsonl'
shards = [json.loads((Path('$C') / f'$NAME-shard{k}' / 'dataset-audit.json').read_text()) for k in [$(echo $SH | tr ' ' ',')]]
audit = dict(complete_collection=True, merged_from=[s['dataset'] for s in shards],
             scheduled=sum(s['scheduled'] for s in shards), retained_turns=sum(s['retained_turns'] for s in shards),
             successful_tasks_with_usable_turns=sum(s['successful_tasks_with_usable_turns'] for s in shards),
             dataset=str(d), dataset_sha256=hashlib.sha256(d.read_bytes()).hexdigest())
(m / 'dataset-audit.json').write_text(json.dumps(audit, indent=2) + '\n'); print('[round2] merged', json.dumps(audit))
PYEOF

# 2. The joint builder (needs --hf; its Piotr rows are dropped in step 3). Reuse v2's verified
# prompt tokenisations for the Piotr states; round-2 contexts are new, hence the larger CPU cap.
cp $ZRUN/joint-data-v2-20260910/working.sqlite $OPENWEBRL_JOB_TMP/context-cache.sqlite
[ -e $FULL ] || $PY scripts/prepare_arm_joint_data.py --hf $ZRUN/joint-data-inputs-0d83b48 --c2 $MERGED --output $FULL \
  --context-cache $OPENWEBRL_JOB_TMP/context-cache.sqlite --cpu-seconds 10000 \
  --model /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --eval openwebrl/data/eval/online-mind2web.jsonl

# 3. Round-2 C2 train pairs with equal action counts; validation and its panels from joint-data/matched.
python3 - <<PYEOF
import json, shutil
from pathlib import Path
full, out, val = Path('$FULL'), Path('$OUT'), Path('$ROOT/matched')
out.mkdir(parents=True)
pairs = [json.loads(l) for l in (full / 'joint_pairs.train.jsonl').open()]
# The builder's own split holds out different task groups than joint-data/matched, so drop round-2
# train rows from matched's validation groups (no held-out task in training), and rename ids:
# round-2 states reuse the round-1 scheme C2:<task>:<turn> and would collide with validation rows.
held = {json.loads(l)['task_group'] for l in (val / 'joint_pairs.validation.jsonl').open()}
c2 = [r for r in pairs if r['source'] == 'C2']
keep = sorted((r for r in c2 if r['task_group'] not in held and len(r['chosen']['actions']) == len(r['rejected']['actions'])), key=lambda r: r['id'])
dropped = keep.pop()['id'] if len(keep) % 32 % 2 else None   # 2-rank trainer needs an even final batch
ids = {r['id'] for r in keep}
def renamed(r):
    for k in ('id', 'base_id'):
        if isinstance(r.get(k), str) and r[k].startswith('C2:'): r[k] = 'C2r2:' + r[k][3:]
    return json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n'
with (out / 'joint_pairs.train.jsonl').open('w') as f:
    for r in pairs:
        if r['id'] in ids: f.write(renamed(r))
with (full / 'joint_sft.train.jsonl').open() as src, (out / 'joint_sft.train.jsonl').open('w') as f:
    for line in src:
        r = json.loads(line)
        if r['id'] in ids: f.write(renamed(r))
for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl']: shutil.copy2(val / name, out / name)
manifest = {x['path']: x for m in (full, val) for x in json.loads((m / 'image-manifest.json').read_text())}
(out / 'image-manifest.json').write_text(json.dumps(list(manifest.values())) + '\n')
report = dict(source=str(full), collection='$MERGED', c2_train_pairs=len(c2), kept_train_pairs=len(keep),
              dropped_matched_validation_groups=sum(r['task_group'] in held for r in c2),
              filter='source C2 (round-2 collection) and equal action counts; validation copied from joint-data/matched',
              dropped_for_even_final_batch=dropped)
(out / 'filter-report.json').write_text(json.dumps(report, indent=2) + '\n'); print('[round2]', json.dumps(report))
PYEOF

# 4. Verify (disjoint task groups etc.) and write the data-driven train config.
$PY scripts/verify_arm_joint_data.py --data $OUT --cpu-seconds 1800 | tail -3
$PY -c "
import sys; sys.path.insert(0,'scripts')
from train_arm_joint_sft import data_config
c=data_config('$OUT','$OUT/train-config.json'); print('[round2] train-config', c['train_examples'], c['source_counts'], 'updates', c['total_updates'])"
