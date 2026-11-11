#!/usr/bin/env bash
# Option C2: add the half-A states of *failed* episodes (Sol labels are pre-execution, so still valid labels).
# Failed-episode pairs get the DPO term only (SFT_SUCCESS_ONLY=1: RPO's NLL term only on pairs from successful episodes).
#   r2-sol-r1-all-m0.3   sol-A-m0.3 (C1's successful-episode pairs) + failed-episode pairs, margin 0.3, same recipe
# Evaluated with base and r2-sol-r1-m0.3 (C1) in one pass (sol-relabel-c2), OM2W + DeepShop, T 1.0, so C2 - C1 is the
# effect of adding failed-episode states. Relabel jobs: sol-relabel/jobs-fail.txt; C1's data from sol_relabel_pipeline.sh.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
O=/gpfs/scrubbed/piotrt/owrl/c2-collect/sol-relabel; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
X="--exclude=g001,g008,g011,g013,g015,g021"
log() { echo "$(date '+%F %T') $*"; }

wait_job() {
  local j=$1 st state elapsed q w
  while st=$(squeue -h -j $j -o "%T %M %q" 2>/dev/null) && [ -n "$st" ]; do
    read state elapsed q <<<"$st"
    if [ "$state" = PENDING ] && [ "$q" != urgent ]; then
      w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
      [ $w -gt 1800 ] && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
    fi
    sleep 120
  done
}

build() {  # build <out> <collection> <prefix> <labels>: failed-episode states, margin 0.3
  [ -f $D/$1/train-config.json ] && return 0
  local j=$(NAME=$2 OUTNAME=$1 IDPREFIX=$3 NSHARDS=2 RELABEL=$4 RELABEL_NAME=gpt-6-sol MARGIN=0.3 ALLOW_FAILED=1 \
    sbatch --parsable -A vlm $X --job-name=bld-$3 tillicum/build_round2_data.sbatch 2>/dev/null)
  log "build $1 ($2) job $j"; wait_job $j
  [ -f $D/$1/train-config.json ] && { log "built $1: $(tr -d '\n ' < $D/$1/filter-report.json | head -c 300)"; return 0; }
  log "FATAL build $1 failed (logs/round2-data-$j.out)"; return 1
}

combine() {
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 sbatch --parsable -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch 2>/dev/null)
  log "combine $1 + $2 -> $3 job $j"; wait_job $j
  [ -f $D/$3/train-config.json ] && { log "combined $3"; return 0; }
  log "FATAL combine $3 failed (logs/combined-data-$j.out)"; return 1
}

log "sol-relabel-c2 start"
for j in $(awk '{print $2}' $O/jobs-fail.txt); do wait_job $j; done
for s in 42 43; do log "relabel Afail-s$s: $(wc -l < $O/Afail-s$s/labels.jsonl) labels of $(wc -l < $O/Afail-s$s-distinct.jsonl); $(cat $O/Afail-s$s/sol-api/usage.json)"; done
( build sol-Afail-s42-m0.3 luna-Afail-s42 SF42m $O/Afail-s42/labels.jsonl ) & p=$!
( build sol-Afail-s43-m0.3 luna-Afail-s43 SF43m $O/Afail-s43/labels.jsonl ) & q=$!
wait $p && wait $q || { log "FATAL failed-episode data"; exit 1; }
combine sol-Afail-s42-m0.3 sol-Afail-s43-m0.3 sol-Afail-m0.3 || exit 1
until [ -f $D/sol-A-m0.3/train-config.json ]; do sleep 300; done   # C1's pairs, built by sol_relabel_pipeline.sh
combine sol-A-m0.3 sol-Afail-m0.3 sol-Aall-m0.3 || exit 1
python3 -c "
import json,collections
c=collections.Counter(json.loads(l).get('terminal_success') for l in open('$D/sol-Aall-m0.3/joint_pairs.train.jsonl'))
print('terminal_success counts', dict(c))"
run=r2-sol-r1-all-m0.3
for attempt in 1 2 3; do
  [ -f $RUNS/$run/student/complete.json ] && break
  live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
  if [ -n "$live" ]; then wait_job $live; continue; fi
  j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$SFT OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/sol-Aall-m0.3 RUN=$run \
    sbatch --parsable -A vlm $X --time=14:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
  log "train $run attempt $attempt job $j"; wait_job $j
done
[ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; exit 1; }
python3 -c "import json;assert json.load(open('$RUNS/$run/frozen-config.json')).get('sft_successful_only') is True" || { log "FATAL sft_successful_only not set in $run"; exit 1; }
u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
g=$(LABEL=$run CKPT=student/update-$u sbatch --parsable -A vlm $X --job-name=gt-$run tillicum/gate.sbatch 2>/dev/null)
log "gate $run job $g"; wait_job $g; log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
[ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $run"; exit 1; }
until [ -f $M/r2-sol-r1-m0.3/merge-manifest.json ]; do sleep 300; done
EJ=""
for B in online-mind2web deepshop; do
  j=$(MODELS=base,r2-sol-r1-m0.3,$run PASS=sol-relabel-c2 BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:3 --cpus-per-task=24 --mem=192G --time=06:00:00 --job-name=ev-solc2-$B tillicum/eval_concurrent.sbatch 2>/dev/null)
  EJ="$EJ $j"; log "eval $B job $j"
done
for j in $EJ; do wait_job $j; done
log "SOL_C2_DONE"
