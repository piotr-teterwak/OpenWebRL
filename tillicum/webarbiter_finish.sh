#!/usr/bin/env bash
# Finish the WebArbiter-7B selector baseline (2026-11-19, Piotr: "do the remaining steps with reruns"):
#  1. DeepShop: the first 3 runs ran concurrently and Amazon served its check page in 80-86% of episodes, so all 150
#     tasks are rerun per run, ONE RUN AT A TIME at 6 browsers (passwebarbiter-sel-ds-r{1,2,3}), then a same-day SFT
#     DeepShop control (passwasel-ds-sftctl/base) to measure that day's Amazon conditions. Throttled results are kept in
#     deepshop-throttled/ (not used).
#  2. OM2W + WebVoyager: invalid non-Wolfram episodes rerun with step 600 s / task 3600 s, as for the other baselines.
#  3. DeepShop invalid episodes of the reruns, same rule.
#  4. Paper table with WebArbiter (Wolfram = failed for all; also no-Amazon) + paired CIs.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
E=/gpfs/scrubbed/piotrt/owrl/eval; X="--exclude=g001,g008,g011,g013,g015,g021"; ERR=$E/webarbiter-finish.sbatch.err
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
sub() { local j; j=$(sbatch --parsable "$@" 2>>$ERR); [ -n "$j" ] || { log "FATAL sbatch refused: $* (see $ERR)"; exit 1; }; echo $j; }
plan() { python3 tillicum/redo_baselines.py --only=webarbiter | awk -F'\t' -v b="$1" '$3==b'; }
redo() {  # redo <bench> <parallel>: submit invalid-episode reruns for every run, return job ids
  local ids="" base ps bench pass n idx
  while IFS=$'\t' read base ps bench pass n idx; do
    [ -n "$idx" ] || continue
    j=$(BENCH=$bench PASS=${pass#pass} INDICES=$idx PARALLEL=$2 STEP_TIMEOUT=600 TASK_TIMEOUT=3600 \
        sub -A vlm $X --time=03:00:00 --job-name=wasel-redo-${ps##*-}-${bench:0:4} webarbiter_sel/eval_webarbiter_sel.sbatch)
    log "redo $ps $bench: $n tasks, job $j" >&2; ids="$ids $j"
  done < <(plan $1)
  echo $ids
}
log "start"
# 1. DeepShop reruns, sequential
for k in 1 2 3; do d=$E/passwebarbiter-sel-r$k/base+webarbitersel; [ -d $d/deepshop ] && [ ! -d $d/deepshop-throttled ] && mv $d/deepshop $d/deepshop-throttled && log "moved r$k deepshop -> deepshop-throttled"; done
prev=""
for k in 1 2 3; do
  j=$(BENCH=deepshop PASS=webarbiter-sel-ds-r$k PARALLEL=6 sub -A vlm $X --time=04:00:00 ${prev:+--dependency=afterany:$prev} --job-name=wasel-ds-r$k webarbiter_sel/eval_webarbiter_sel.sbatch)
  log "deepshop rerun r$k job $j${prev:+ (after $prev)}"; prev=$j; DS="${DS:-} $j"
done
c=$(MODELS=base PASS=wasel-ds-sftctl BENCH=deepshop TEMP=1.0 PARALLEL=6 sub -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --dependency=afterany:$prev --job-name=wasel-ds-sftctl tillicum/eval_concurrent.sbatch)
log "same-day SFT DeepShop control job $c (after $prev)"
# 2. OM2W + WebVoyager invalid reruns, once the WebVoyager runs are done
wait_jobs 500552 500555
log "WebVoyager runs done: $(for k in 1 2 3; do echo -n "r$k $(ls $E/passwebarbiter-sel-r$k/base+webarbitersel/webvoyager/results | wc -l)/595 "; done)"
R1=$(redo online-mind2web 12); R2=$(redo webvoyager 12)
# 3. DeepShop invalid reruns after the DeepShop reruns + control
wait_jobs $DS $c
for k in 1 2 3; do log "deepshop rerun r$k: $(tr -d '\n ' < $E/passwebarbiter-sel-ds-r$k/base+webarbitersel/deepshop/summary.json 2>/dev/null | cut -c1-160)"; done
log "SFT control: $(tr -d '\n ' < $E/passwasel-ds-sftctl/base/deepshop/summary.json 2>/dev/null | cut -c1-160)"
R3=$(redo deepshop 6)
wait_jobs $R1 $R2 $R3
log "all reruns done; remaining invalid non-Wolfram per run/bench:"; python3 tillicum/redo_baselines.py --only=webarbiter | cut -f2-5
# 4. tables (MIG job: reads ~10k result files)
cat > $E/redo2/wasel-tables.sbatch <<EOS
#!/bin/bash
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=wasel-tables
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/wasel-tables-%j.out
cd /gpfs/home/piotrt/OpenWebRL
python3 tillicum/paper_table.py --wolfram3 > tillicum/results/paper-table/wolfram3-with-webarbiter.txt 2>&1
python3 tillicum/paper_table.py --wolfram3 --no-amazon > tillicum/results/paper-table/wolfram3-no-amazon-with-webarbiter.txt 2>&1
python3 tillicum/paper_table_ci.py > tillicum/results/paper-table/ci-no-wolfram-with-webarbiter.txt 2>&1
EOS
t=$(sub $E/redo2/wasel-tables.sbatch); log "tables job $t"; wait_jobs $t
log "WEBARBITER_DONE"
