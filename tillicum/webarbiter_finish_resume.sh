#!/usr/bin/env bash
# Continuation of webarbiter_finish.sh after its first run (2026-11-19 21:41): that run submitted the DeepShop rerun
# chain (501633 -> 501634 -> 501635), the SFT DeepShop control (501636) and the OM2W/WebVoyager invalid reruns
# (501738-501743), but captured its log lines into the job-id lists (fixed in webarbiter_finish.sh: logs to stderr),
# so its later waits would not have waited. This picks up: wait, DeepShop invalid reruns, tables.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
E=/gpfs/scrubbed/piotrt/owrl/eval; X="--exclude=g001,g008,g011,g013,g015,g021"; ERR=$E/webarbiter-finish.sbatch.err
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
sub() { local j; j=$(sbatch --parsable "$@" 2>>$ERR); [ -n "$j" ] || { log "FATAL sbatch refused: $* (see $ERR)" >&2; exit 1; }; echo $j; }
plan() { python3 tillicum/redo_baselines.py --only=webarbiter | awk -F'\t' -v b="$1" '$3==b'; }
redo() {  # redo <bench> <parallel>: submit invalid-episode reruns for every run; prints only job ids
  local ids="" base ps bench pass n idx j
  while IFS=$'\t' read base ps bench pass n idx; do
    [ -n "$idx" ] || continue
    j=$(BENCH=$bench PASS=${pass#pass} INDICES=$idx PARALLEL=$2 STEP_TIMEOUT=600 TASK_TIMEOUT=3600 \
        sub -A vlm $X --time=03:00:00 --job-name=wasel-redo-${ps##*-}-${bench:0:4} webarbiter_sel/eval_webarbiter_sel.sbatch)
    log "redo $ps $bench: $n tasks, job $j" >&2; ids="$ids $j"
  done < <(plan $1)
  echo $ids
}
log "resume: waiting for DeepShop chain + control (501633-501636) and OM2W/WebVoyager reruns (501738-501743)"
wait_jobs 501633 501634 501635 501636
for k in 1 2 3; do log "deepshop rerun r$k: $(tr -d '\n ' < $E/passwebarbiter-sel-ds-r$k/base+webarbitersel/deepshop/summary.json 2>/dev/null | cut -c1-160)"; done
log "SFT control: $(tr -d '\n ' < $E/passwasel-ds-sftctl/base/deepshop/summary.json 2>/dev/null | cut -c1-160)"
R3=$(redo deepshop 6)
log "deepshop redo jobs:$R3"
wait_jobs 501738 501739 501740 501741 501742 501743 $R3
log "all reruns done; remaining invalid non-Wolfram per run/bench:"; python3 tillicum/redo_baselines.py --only=webarbiter | cut -f2-5
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
