#!/usr/bin/env bash
# Option C: does a stronger labeller help? gpt-6-sol re-picks the preferred candidate (with a probability per candidate)
# at the stored half-A states of the Luna round-1 collections (successful episodes, >= 2 distinct candidate actions),
# then round 1 is retrained with exactly r2-luna-r1's recipe (RPO beta 0.1, SFT weight 1, lr 5e-5, 2 epochs, seed 42,
# reference SFT) on
#   r2-sol-r1        Sol pick vs any other candidate (same pairing as r2-luna-r1)          -> labeller effect
#   r2-sol-r1-m0.3   rejected must be >= 0.3 below the chosen in Sol's probability          -> false-negative fix
# and both are evaluated with base and r2-luna-r1 in one pass (sol-relabel) on OM2W and DeepShop, T 1.0.
# Relabel jobs are submitted by hand first (sol-relabel/jobs.txt).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
O=/gpfs/scrubbed/piotrt/owrl/c2-collect/sol-relabel; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
X="--exclude=g001,g008,g011,g013,g015,g021"
log() { echo "$(date '+%F %T') $*"; }

wait_job() {  # wait for one job; urgent after 30 min pending
  local j=${1:-} st state elapsed q w; [ -n "$j" ] || { log "wait_job: empty job id (sbatch failed?)"; return 1; }
  while st=$(squeue -h -j $j -o "%T %M %q" 2>/dev/null) && [ -n "$st" ]; do
    read state elapsed q <<<"$st"
    if [ "$state" = PENDING ] && [ "$q" != urgent ]; then
      w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
      [ $w -gt 1800 ] && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
    fi
    sleep 120
  done
}

build() {  # build <out name> <collection> <id prefix> <labels> [margin]
  [ -f $D/$1/train-config.json ] && return 0
  local j=$(NAME=$2 OUTNAME=$1 IDPREFIX=$3 NSHARDS=2 RELABEL=$4 RELABEL_NAME=gpt-6-sol MARGIN=${5:-} \
    sbatch --parsable -A vlm $X --job-name=bld-$3 tillicum/build_round2_data.sbatch 2>/dev/null)
  log "build $1 ($2, labels $(basename $(dirname $4)), margin ${5:-none}) job $j"; wait_job $j
  [ -f $D/$1/train-config.json ] && { log "built $1: $(tr -d '\n ' < $D/$1/filter-report.json | head -c 300)"; return 0; }
  log "FATAL build $1 failed (logs/round2-data-$j.out)"; return 1
}

combine() {  # combine <A> <B> <OUT>
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 sbatch --parsable -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch 2>/dev/null)
  log "combine $1 + $2 -> $3 job $j"; wait_job $j
  [ -f $D/$3/train-config.json ] && { log "combined $3"; return 0; }
  log "FATAL combine $3 failed (logs/combined-data-$j.out)"; return 1
}

train() {  # train <run> <data>: r2-luna-r1's recipe from SFT, resume up to 2x, then gate
  local run=$1 data=$2 j live
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then log "train $run: adopting job $live"; wait_job $live; continue; fi
    j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 OWRL_ACTOR=$SFT OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/$data RUN=$run \
      sbatch --parsable -A vlm $X --time=10:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  local g=$(LABEL=$run CKPT=student/update-$u sbatch --parsable -A vlm $X --job-name=gt-$run tillicum/gate.sbatch 2>/dev/null)
  log "gate $run job $g"; wait_job $g
  log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $run"; return 2; }
}

log "sol-relabel start"
for j in $(awk '{print $2}' $O/jobs.txt); do wait_job $j; done
for s in 42 43; do log "relabel A-s$s: $(wc -l < $O/A-s$s/labels.jsonl) labels of $(wc -l < $O/A-s$s-distinct.jsonl); $(cat $O/A-s$s/sol-api/usage.json)"; done
# Builds of one collection run one after the other: step 1 of build_round2_data rewrites the collection's merged
# training.jsonl, so two builds of luna-A-s42 at once could read a half-written file.
for m in "" 0.3; do sfx=${m:+-m$m}
  ( build sol-A-s42$sfx luna-A-s42 SA42${m:+m} $O/A-s42/labels.jsonl $m ) & p=$!
  ( build sol-A-s43$sfx luna-A-s43 SA43${m:+m} $O/A-s43/labels.jsonl $m ) & q=$!
  wait $p && wait $q || { log "FATAL data (margin ${m:-none})"; exit 1; }
  combine sol-A-s42$sfx sol-A-s43$sfx sol-A$sfx || exit 1
done
train r2-sol-r1 sol-A & a=$!
train r2-sol-r1-m0.3 sol-A-m0.3 & b=$!
wait $a; wait $b
MODELS=base,r2-luna-r1; for r in r2-sol-r1 r2-sol-r1-m0.3; do [ -f $M/$r/merge-manifest.json ] && MODELS="$MODELS,$r"; done
n=$(tr ',' '\n' <<<"$MODELS" | wc -l); EJ=""
for B in online-mind2web deepshop; do
  j=$(MODELS=$MODELS PASS=sol-relabel BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-sol-$B tillicum/eval_concurrent.sbatch 2>/dev/null)
  EJ="$EJ $j"; log "eval $B ($MODELS) job $j"
done
for j in $EJ; do wait_job $j; done
log "SOL_RELABEL_DONE"
