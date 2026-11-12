#!/usr/bin/env bash
# DAgger round 2 + data-matched control (follows overnight_dagger_fullft.sh's r2-dagger-r1). All collections are
# DAgger-style (EXECUTE=first: the actor executes its own sample, gpt-6-luna labels every state); pairs = Luna pick vs
# another candidate at all valid states; RPO's SFT term only on successful-episode pairs (SFT_SUCCESS_ONLY=1).
#   r2-dagger-ctrl  one round from SFT on half A + half B, both SFT-executed (dagger-A + dagger-Bsft)
#   r2-dagger-r2    from r2-dagger-r1 on half B executed by r2-dagger-r1, reference = r2-dagger-r1 (standard iterative)
# Evaluated with base and r2-dagger-r1 in one pass (night-dagger2), OM2W + DeepShop, T 1.0.
# The half-B SFT collection (jobs-Bsft.txt) is submitted by hand; nothing is submitted once vlm usage reaches $CAP.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
C=/gpfs/scrubbed/piotrt/owrl/c2-collect; G=$C/dagger; L=$C/luna-iter; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
PY=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/venv/bin/python
X="--exclude=g001,g008,g011,g013,g015,g021"; CAP=${CAP:-950}
log() { echo "$(date '+%F %T') $*"; }

budget_ok() {
  local used=$(hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); print a[1]; exit}')
  if [ -z "$used" ]; then log "budget: could not read hyakusage; proceeding"; return 0; fi
  if python3 -c "import sys; sys.exit(0 if float('$used') < $CAP else 1)"; then return 0; fi
  log "BUDGET GUARD: piotrt vlm usage \$$used >= \$$CAP, not submitting"; return 1
}
submit() { budget_ok || return 1; sbatch --parsable "$@" 2>/dev/null; }

wait_job() {
  local j=${1:-} st state elapsed q w; [ -n "$j" ] || { log "wait_job: empty job id"; return 1; }
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

collect() {  # collect <prefix e.g. dagger-Bsft> <actor> <jobs file> <job tag>: half B, seeds 42/43 x shards 1/2
  local pre=$1 actor=$2 jf=$3 tag=$4
  for attempt in 1 2 3; do
    for j in $(awk '{print $2}' $jf 2>/dev/null); do wait_job $j; done
    local todo=""; for s in 42 43; do for k in 1 2; do complete_shard $pre-s$s-shard$k || todo="$todo $s:$k"; done; done
    [ -z "$todo" ] && { log "collect $pre complete"; return 0; }
    [ $attempt = 3 ] && { log "FATAL collect $pre incomplete:$todo"; return 1; }
    : > $jf
    for sk in $todo; do s=${sk%:*}; k=${sk#*:}
      j=$(ACTOR=$actor TEACHER=luna EXECUTE=first SEED=$s NAME=$pre-s$s-shard$k PARALLEL=16 TASKS=$L/tasks-B-shard$k.jsonl LUNA_MAX_USD=25 \
          submit -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-$tag$s-$k tillicum/c2_collect.sbatch) || return 1
      echo "$pre-s$s-shard$k $j" >> $jf; log "collect $pre shard s$s-$k job $j (attempt $((attempt+1)))"
    done
  done
}

data() {  # data <prefix> <id tag>: export all valid episodes, teacher labels, build per seed, combine -> $D/<prefix>
  local pre=$1 tag=$2 j
  [ -f $D/$pre/train-config.json ] && return 0
  for s in 42 43; do
    for k in 1 2; do [ -f $C/$pre-s$s-all-shard$k/dataset-audit.json ] || OMP_NUM_THREADS=1 PYTHONPATH=$PWD $PY scripts/export_failed_turns.py \
        --include-successful --collection $C/$pre-s$s-shard$k --out $C/$pre-s$s-all-shard$k 2>&1 | grep -v Warn | tail -1 | cut -c1-200; done
    python3 scripts/teacher_labels_from_selections.py --collection $C/$pre-s$s-shard1 $C/$pre-s$s-shard2 --out $G/labels-$pre-s$s.jsonl
  done
  for s in 42 43; do
    ( [ -f $D/$pre-s$s/train-config.json ] && exit 0
      j=$(NAME=$pre-s$s-all OUTNAME=$pre-s$s IDPREFIX=$tag$s NSHARDS=2 RELABEL=$G/labels-$pre-s$s.jsonl RELABEL_NAME=gpt-6-luna ALLOW_FAILED=1 \
          submit -A vlm $X --job-name=bld-$tag$s tillicum/build_round2_data.sbatch) || exit 1
      log "build $pre-s$s job $j"; wait_job $j
      [ -f $D/$pre-s$s/train-config.json ] || { log "FATAL build $pre-s$s (logs/round2-data-$j.out)"; exit 1; } ) &
  done
  wait
  [ -f $D/$pre-s42/train-config.json ] && [ -f $D/$pre-s43/train-config.json ] || { log "FATAL $pre data"; return 1; }
  combine $pre-s42 $pre-s43 $pre
}

combine() {
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 submit -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch) || return 1
  log "combine $1 + $2 -> $3 job $j"; wait_job $j
  [ -f $D/$3/train-config.json ] && { log "combined $3: $(python3 -c "import json;print(json.load(open('$D/$3/train-config.json'))['train_examples'])") pairs"; return 0; }
  log "FATAL combine $3"; return 1
}

train() {  # train <run> <data> <actor> <reference>: same RPO recipe, resume up to 2x, then gate
  local run=$1 data=$2 actor=$3 ref=$4 j live
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then log "train $run: adopting job $live"; wait_job $live; continue; fi
    budget_ok || return 1
    j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$actor OWRL_REFERENCE=$ref OWRL_JOINT_DATA=$D/$data RUN=$run \
        sbatch --parsable -A vlm $X --time=12:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data, actor $(basename $actor), ref $(basename $ref)) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  local g=$(LABEL=$run CKPT=student/update-$u submit -A vlm $X --job-name=gt-$run tillicum/gate.sbatch) || return 1
  log "gate $run job $g"; wait_job $g; log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $run"; return 2; }
}

control() {
  collect dagger-Bsft $SFT $G/jobs-Bsft.txt dgBs || return 1
  data dagger-Bsft DBs || return 1
  until [ -f $D/dagger-A/train-config.json ]; do sleep 300; done        # built by overnight_dagger_fullft.sh
  combine dagger-A dagger-Bsft dagger-ctrl || return 1
  train r2-dagger-ctrl dagger-ctrl $SFT $SFT
}

round2() {
  until [ -f $M/r2-dagger-r1/merge-manifest.json ]; do                  # gated + merged by overnight_dagger_fullft.sh
    grep -qE "\[dagger\].*(GATE FAIL|FATAL)" $G/overnight.log 2>/dev/null && { log "r2-dagger-r1 failed upstream; no round 2"; return 1; }
    sleep 300; done
  local R1=$M/r2-dagger-r1
  [ -s $G/jobs-Br1.txt ] || : > $G/jobs-Br1.txt
  collect dagger-Br1 $R1 $G/jobs-Br1.txt dgBr || return 1
  data dagger-Br1 DBr || return 1
  train r2-dagger-r2 dagger-Br1 $R1 $R1
}

log "dagger round 2 start (cap \$$CAP)"
control > >(sed -u 's/^/[ctrl] /') 2>&1 & A=$!
round2 > >(sed -u 's/^/[r2] /') 2>&1 & B=$!
wait $A; wait $B
MODELS=base; for r in r2-dagger-r1 r2-dagger-r2 r2-dagger-ctrl; do [ -f $M/$r/merge-manifest.json ] && MODELS="$MODELS,$r"; done
n=$(tr ',' '\n' <<<"$MODELS" | wc -l); EJ=""
if [ $n -ge 2 ]; then
  for B in online-mind2web deepshop; do
    j=$(MODELS=$MODELS PASS=night-dagger2 BENCH=$B TEMP=1.0 submit -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-dagger2-$B tillicum/eval_concurrent.sbatch) || break
    EJ="$EJ $j"; log "eval night-dagger2 $B ($MODELS) job $j"
  done
  for j in $EJ; do wait_job $j; done
fi
log "DAGGER2_DONE"
