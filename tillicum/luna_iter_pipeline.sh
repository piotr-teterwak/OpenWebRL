#!/usr/bin/env bash
# Iterative Luna RPO, end to end (David's recipe: RPO beta 0.1 + SFT weight 1, lr 5e-5, 2 epochs, LoRA r16, reference SFT).
#   round 1   : SFT + Luna pre-execution selection on half A (seeds 42, 43) -> r2-luna-r1 (RPO from SFT)
#   round 2   : r2-luna-r1 + Luna on half B (seeds 42, 43) -> r2-luna-joint (from SFT, A + B-on-policy pairs)
#                                                       -> r2-luna-cont  (from r2-luna-r1, B pairs, reference SFT)
#   control   : SFT + Luna on half B too -> r2-luna-ctrl (one round-1-style run from SFT on A + B-from-SFT)
#   eval      : base + the 4 models, one job per benchmark, T 1.0, pass luna-iter
# Collection jobs for A and B-from-SFT are submitted by hand first (round1-jobs.txt); this script resubmits
# incomplete shards. Any job pending > 30 min is moved to urgent QoS (2x billing; window to Nov 16).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
L=/gpfs/scrubbed/piotrt/owrl/c2-collect/luna-iter; C=/gpfs/scrubbed/piotrt/owrl/c2-collect
D=/gpfs/scrubbed/piotrt/owrl/joint-data; RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged
# Working mirror of Zixian's scrubbed runtime (hers is past the 60-day purge). Jobs resuming frozen state started
# on her path keep it, since their frozen configs record it: r2-luna-r1 training and the luna-* round-1 shards.
SFT_Z=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
X="--exclude=g001,g008,g011,g013,g015,g021"
log() { echo "$(date '+%F %T') $*"; }

wait_job() {  # wait for one job; urgent after 30 min pending
  # all locals: an unscoped `read s ...` here once overwrote collect()'s seed (resubmitted with SEED=RUNNING)
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
complete_shard() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$C/$1/dataset-audit.json'))['complete_collection'] else 1)" 2>/dev/null; }

collect() {  # collect <name> <half A|B> <seed> <actor> [first job ids...]: both shards complete, resubmitting up to 2x
  local name=$1 h=$2 s=$3 actor=$4; shift 4; local jobs="$*"
  for attempt in 1 2 3; do
    for j in $jobs; do wait_job $j; done
    local todo=""; for k in 1 2; do complete_shard $name-shard$k || todo="$todo $k"; done
    [ -z "$todo" ] && { log "collect $name complete"; return 0; }
    [ $attempt = 3 ] && break
    jobs=""; for k in $todo; do
      jobs="$jobs $(ACTOR=$actor TEACHER=luna SEED=$s NAME=$name-shard$k PARALLEL=16 TASKS=$L/tasks-$h-shard$k.jsonl LUNA_MAX_USD=25 \
        sbatch --parsable -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-${name#luna-}-$k tillicum/c2_collect.sbatch)"
    done; log "collect $name attempt $((attempt+1)): resubmitted shards$todo as$jobs"
  done
  log "FATAL collect $name incomplete"; return 1
}

build() {  # build <collection name> <id prefix>
  [ -f $D/$1/train-config.json ] && return 0
  local j=$(NAME=$1 OUTNAME=$1 IDPREFIX=$2 NSHARDS=2 sbatch --parsable -A vlm $X --job-name=bld-$2 tillicum/build_round2_data.sbatch)
  log "build $1 ($2) job $j"; wait_job $j
  [ -f $D/$1/train-config.json ] && { log "built $1: $(cat $D/$1/filter-report.json | tr -d '\n ' | head -c 300)"; return 0; }
  log "FATAL build $1 failed (logs/round2-data-$j.out)"; return 1
}

combine() {  # combine <A> <B> <OUT>
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 sbatch --parsable -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch)
  log "combine $1 + $2 -> $3 job $j"; wait_job $j
  [ -f $D/$3/train-config.json ] && { log "combined $3"; return 0; }
  log "FATAL combine $3 failed (logs/combined-data-$j.out)"; return 1
}

train() {  # train <run> <data> <actor> -> RPO, resubmitting (resume) up to 2x; then gate (merge + length-cap check)
  local run=$1 data=$2 actor=$3
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    local j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 OWRL_ACTOR=$actor OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/$data RUN=$run \
      sbatch --parsable -A vlm $X --time=10:00:00 --job-name=tr-$run tillicum/train_joint.sbatch)
    log "train $run (data $data, actor $(basename $actor)) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  local g=$(LABEL=$run CKPT=student/update-$u sbatch --parsable -A vlm $X --job-name=gt-$run tillicum/gate.sbatch)
  log "gate $run job $g"; wait_job $g
  log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ -f $M/$run/merge-manifest.json ] || { log "FATAL merge $run missing"; return 1; }
  # a failed gate (runaway generations) stops whatever would build on this model; the merge is kept for inspection
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $run"; return 2; }
}

read_jobs() { awk -v n="$1" '$1 ~ "^"n"-shard" {print $2}' $L/round1-jobs.txt | tr '\n' ' '; }

round1_and_2() {
  ( collect luna-A-s42 A 42 $SFT_Z $(read_jobs luna-A-s42) && build luna-A-s42 LA42 ) & p1=$!
  ( collect luna-A-s43 A 43 $SFT_Z $(read_jobs luna-A-s43) && build luna-A-s43 LA43 ) & p2=$!
  wait $p1 && wait $p2 || { log "FATAL round-1 data"; return 1; }
  combine luna-A-s42 luna-A-s43 luna-A || return 1
  train r2-luna-r1 luna-A $SFT_Z || return 1
  local R1=$M/r2-luna-r1 jobs
  for s in 42 43; do jobs=""
    for k in 1 2; do jobs="$jobs $(ACTOR=$R1 TEACHER=luna SEED=$s NAME=luna-Br1-s$s-shard$k PARALLEL=16 TASKS=$L/tasks-B-shard$k.jsonl LUNA_MAX_USD=25 \
      sbatch --parsable -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-Br1$s-$k tillicum/c2_collect.sbatch)"; done
    echo "$s $jobs" >> $L/round2-jobs.txt; log "round-2 collection seed $s jobs$jobs"
  done
  ( collect luna-Br1-s42 B 42 $R1 $(awk '$1==42{$1="";print}' $L/round2-jobs.txt) && build luna-Br1-s42 LBr42 ) & p1=$!
  ( collect luna-Br1-s43 B 43 $R1 $(awk '$1==43{$1="";print}' $L/round2-jobs.txt) && build luna-Br1-s43 LBr43 ) & p2=$!
  wait $p1 && wait $p2 || { log "FATAL round-2 data"; return 1; }
  combine luna-Br1-s42 luna-Br1-s43 luna-Br1 || return 1
  combine luna-A luna-Br1 luna-joint || return 1
  train r2-luna-joint luna-joint $SFT & p1=$!
  train r2-luna-cont luna-Br1 $R1 & p2=$!
  wait $p1; a=$?; wait $p2; b=$?; return $((a + b))
}

control() {
  ( collect luna-Bsft-s42 B 42 $SFT_Z $(read_jobs luna-Bsft-s42) && build luna-Bsft-s42 LBs42 ) & p1=$!
  ( collect luna-Bsft-s43 B 43 $SFT_Z $(read_jobs luna-Bsft-s43) && build luna-Bsft-s43 LBs43 ) & p2=$!
  wait $p1 && wait $p2 || { log "FATAL control data"; return 1; }
  combine luna-Bsft-s42 luna-Bsft-s43 luna-Bsft || return 1
  # luna-A is built by the round-1 branch; wait for it
  until [ -f $D/luna-A/train-config.json ]; do sleep 300; done
  combine luna-A luna-Bsft luna-ctrl || return 1
  train r2-luna-ctrl luna-ctrl $SFT
}

log "pipeline start"
round1_and_2 > >(sed -u 's/^/[iter] /') 2>&1 & P=$!
control > >(sed -u 's/^/[ctrl] /') 2>&1 & Q=$!
wait $P; a=$?; wait $Q; b=$?
log "training branches done (iter rc $a, ctrl rc $b)"
MODELS=base; for r in r2-luna-r1 r2-luna-joint r2-luna-cont r2-luna-ctrl; do [ -f $M/$r/merge-manifest.json ] && MODELS="$MODELS,$r"; done
n=$(tr ',' '\n' <<<"$MODELS" | wc -l); EJ=""
for B in online-mind2web webvoyager deepshop; do
  t=06:00:00; [ $B = webvoyager ] && t=14:00:00
  j=$(MODELS=$MODELS PASS=luna-iter BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=$t --job-name=ev-li-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $j"; log "eval $B ($MODELS) job $j"
done
for j in $EJ; do wait_job $j; done
log "PIPELINE_DONE"
