#!/usr/bin/env bash
#SBATCH --job-name=owrl-gate
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/gate-%j.out
#
# Generation gate before any browser eval: merge a finished joint run's adapter, then sample
# 5 candidates on C2 states with the probe and report how often responses hit the 1024-token
# cap (length_cap_rate) and tool calls per response. Base is ~1% in browser evals; the
# lr-5e-5 DPO/RPO models that broke in 377372 ran away on ~80% of episodes.
#   LABEL=rpo-a1-lr5e-5-b0.5 CKPT=student/update-000174 sbatch tillicum/gate.sbatch
# A run that did not finish: RUN_NAME=<run> LABEL=<new label> CKPT=student/update-N ALLOW_PARTIAL=1
set -euo pipefail
LABEL="${LABEL:?set LABEL to a run name under joint-retrain/runs}"
CKPT="${CKPT:-student/update-000174}"
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
RUN=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs/${RUN_NAME:-$LABEL}
MERGED=/gpfs/scrubbed/piotrt/owrl/eval/merged/$LABEL
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
cd $REPO
[ -f $RUN/student/complete.json ] || [ "${ALLOW_PARTIAL:-0}" = 1 ] || { echo "FATAL $LABEL training not complete"; exit 1; }
[ -f $MERGED/merge-manifest.json ] || $ZRT/arm-reproduction/venv/bin/python scripts/merge_arm_c2_student.py \
  --run-root $RUN --checkpoint $RUN/$CKPT --output $MERGED
ACTORS="$LABEL=$MERGED" SKIP_REPLAY=1 STATES="${STATES:-200}" bash tillicum/probe_round2.sbatch | tee /gpfs/scrubbed/piotrt/owrl/logs/gate-$SLURM_JOB_ID.summary
# Pass/fail for afterok-chained evals: base runs ~0.5%, broken lr-5e-5 DPO/RPO ~27%.
RATE=$(grep -o '"length_cap_rate": [0-9.]*' /gpfs/scrubbed/piotrt/owrl/logs/gate-$SLURM_JOB_ID.summary | tail -1 | awk '{print $2}')
python3 -c "import sys; r=float('$RATE'); m=float('${MAX_CAP:-0.05}'); print(f'[gate] length_cap_rate {r:.3f} vs max {m}:', 'PASS' if r<=m else 'FAIL'); sys.exit(r>m)"
