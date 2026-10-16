#!/usr/bin/env bash
# Requires explicit approval: 1 H200 x 1 hour = 1 GPU-hour, estimated $0.90. No API calls, no browsers.
# Kev-27B (pinned v1.0 weights, audited serving overrides) scores the 124 branching states under the
# controlled selector protocol, in the protocol order plus 4 extra candidate orders; then compares Kev,
# Piotr ARM and the selection heads on Luna agreement and chosen-action continuation success.
#SBATCH --job-name=openwebrl-kev-branch
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-kev-branch-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-kev
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
OUT=$R/kev-branch-$SLURM_JOB_ID
export PYTHONPATH=$SRC PYTHONUNBUFFERED=1
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/score_kev_branch.py --output $OUT --port $((19300 + SLURM_JOB_ID % 600))
picks=(--picks $R/piotr_arm.jsonl --picks $OUT/kev_primary.jsonl --picks $OUT/kev_order_averaged.jsonl)
for v in pointwise listwise; do
  $PY -c "import json,sys; m=json.load(open('$R/heads/$v/metrics.json'))['branch_head_picks']; open('$OUT/head_$v.jsonl','w').write(''.join(json.dumps({'state_id':k,'selected_index':i})+'\n' for k,i in m.items()))"
  picks+=(--picks $OUT/head_$v.jsonl)
done
$PY scripts/compare_branch_selectors.py "${picks[@]}" --output $OUT/comparison.json
echo "kev branch complete $(date -Is)"
