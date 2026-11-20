#!/usr/bin/env bash
#SBATCH --job-name=mix-panels
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only (joining turn records with fresh samples); a MIG slice bills 1/7 of an H200.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/mix-panels-%j.out
#
# Mixed-pool Round 2 (2026-11-19): 6-candidate panels (Round-1 candidates 0-2 + 3 SFT samples from
# tillicum/sample_sft_candidates.sbatch) per seed, scripts/build_mix_panels.py, plus a PILOT-state subset of seed 42
# (hash order of ids) for the Sol cost pilot.
#   sbatch tillicum/build_mix_panels.sbatch          ->  c2-collect/sol2-mix/panels-s{42,43}.jsonl, panels-s42-pilot.jsonl
set -euo pipefail
REPO=/gpfs/home/piotrt/OpenWebRL; PY=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/venv/bin/python
C=/gpfs/scrubbed/piotrt/owrl/c2-collect; PREFIX=${PREFIX:-sol2-Br1}; OUTDIR=${OUTDIR:-$C/sol2-mix}; SEEDS=${SEEDS:-42 43}
TASKS=${TASKS:-/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/tasks.jsonl}
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
cd $REPO
for s in $SEEDS; do
  $PY scripts/build_mix_panels.py --dataset $C/$PREFIX-s$s-all-shard1/training.jsonl $C/$PREFIX-s$s-all-shard2/training.jsonl \
    --sft $OUTDIR/sft-s$s${SFT_SUFFIX:-}.jsonl --tasks $TASKS --out $OUTDIR/panels-s$s${SFT_SUFFIX:-}.jsonl
done
python3 - <<PYEOF
import hashlib, json
from pathlib import Path
src = Path('$OUTDIR/panels-s42${SFT_SUFFIX:-}.jsonl')
if src.exists():
    rows = sorted(src.read_text().splitlines(), key=lambda l: hashlib.sha256(('pilot:' + json.loads(l)['id']).encode()).hexdigest())
    pilot = rows[:${PILOT:-200}]
    Path('$OUTDIR/panels-s42${SFT_SUFFIX:-}-pilot.jsonl').write_text(''.join(l + '\n' for l in pilot))
    print('[mix-panels] pilot', len(pilot), 'of', len(rows))
PYEOF
