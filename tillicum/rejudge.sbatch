#!/usr/bin/env bash
#SBATCH --job-name=owrl-rejudge
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Judge API calls only; no GPU work. A MIG slice bills 1/7 of an H200 and CPU-only jobs
# on this cluster queue for days.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/rejudge-%j.out
#
# Re-judge finished Online-Mind2Web evals with a new judge (o4-mini was retired):
#   EVALS="pass1/base/online-mind2web passsweep3/rpo-matched-lr5e-5/online-mind2web" sbatch tillicum/rejudge.sbatch
# LIMIT=10 for a pilot. Paths are relative to eval/. Resubmitting resumes.
set -euo pipefail
REPO=/gpfs/home/piotrt/OpenWebRL
EVAL=/gpfs/scrubbed/piotrt/owrl/eval
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
trap 'rm -rf "$OPENWEBRL_JOB_TMP"' EXIT
set -a; source $REPO/.env; set +a
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL OPENAI_API_KEY did not resolve"; exit 1; }
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
cd $REPO
DIRS=(); for d in ${EVALS:?set EVALS}; do DIRS+=($EVAL/$d); done
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/rejudge_om2w.py \
  --model ${JUDGE:-gpt-5.6-terra} --parallel ${PARALLEL:-8} ${LIMIT:+--limit $LIMIT} "${DIRS[@]}"
