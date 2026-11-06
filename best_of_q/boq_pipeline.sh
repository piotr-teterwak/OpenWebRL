#!/usr/bin/env bash
# Paper schedule (arXiv 2601.22701 App. A.2) as one Slurm dependency chain on account vlm:
#   5 exploration runs (eps-greedy 0.5)  -> q1
#   4 cycles: 2 runs with argmax-Q(q_k) -> q_{k+1} trained on everything so far   (q5 = final, 13 runs)
#   evaluation, 3 seeds each: Best-of-Q(q5), random-of-3 (eps = 1), our N=1 base (T 0.7); plus 1 greedy N=1 run
# Every job id goes to $ROOT/pipeline_jobs.txt ("<name> <jobid>").   bash boq_pipeline.sh  [DRY=1 to print only]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=${BOQ_ROOT:-/gpfs/scrubbed/piotrt/owrl/best_of_q_v2}; mkdir -p $ROOT; LOG=$ROOT/pipeline_jobs.txt
EXCL=--exclude=g011,g013,g015,g021
sub() {  # name dependency env... -- sbatch-file
  local name=$1 dep=$2; shift 2; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  if [ "${DRY:-0}" = 1 ]; then echo "DRY $name dep=${dep:-none} ${envs[*]} $*" >&2; echo "dry-$name"; return; fi
  local id; id=$(env "${envs[@]}" sbatch --parsable -J boq-$name $EXCL ${dep:+--dependency=afterok:$dep} "$@")
  echo "$name $id" >> $LOG; echo $id
}
seed_jobs=(); seed_runs=()
for s in 1 2 3 4 5; do seed_jobs+=($(sub seed-s$s "" POLICY=epsilon SEED=$s RUN=seed-s$s -- boq_run.sbatch)); seed_runs+=(seed-s$s); done
runs="${seed_runs[*]}"; dep=$(IFS=:; echo "${seed_jobs[*]}")
q=$(sub q1 "$dep" NAME=q1 "RUNS=$runs" -- boq_train.sbatch)
for c in 1 2 3 4; do
  Qpath=$ROOT/q/q$c/q_function_best.pt; cj=()
  for k in 1 2; do s=$((10 * c + k)); cj+=($(sub cyc$c-s$s "$q" POLICY=q Q=$Qpath SEED=$s RUN=cyc$c-s$s -- boq_run.sbatch)); runs="$runs cyc$c-s$s"; done
  q=$(sub q$((c + 1)) "$(IFS=:; echo "${cj[*]}")" NAME=q$((c + 1)) "RUNS=$runs" -- boq_train.sbatch)
done
QF=$ROOT/q/q5/q_function_best.pt
for s in 101 102 103; do
  sub eval-boq-s$s "$q" POLICY=q Q=$QF SEED=$s RUN=eval-boq-s$s COLLECT=0 -- boq_run.sbatch >/dev/null
  sub eval-random-s$s "" POLICY=random SEED=$s RUN=eval-random-s$s COLLECT=0 -- boq_run.sbatch >/dev/null
  sub eval-base-s$s "" POLICY=base SEED=$s RUN=eval-base-s$s COLLECT=0 -- boq_run.sbatch >/dev/null
done
sub eval-greedy-s101 "" POLICY=greedy SEED=101 RUN=eval-greedy-s101 COLLECT=0 -- boq_run.sbatch >/dev/null
echo "submitted; ids in $LOG"
