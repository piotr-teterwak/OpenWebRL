#!/usr/bin/env bash
# Overnight 2026-11-11/12, two independent branches, each evaluated in its own pass (OM2W + DeepShop, T 1.0) with
# base and r2-luna-r1 (LoRA, Luna-steered round 1) as references:
#   fullft  r2-luna-r1-fullft-lr{1e-6,5e-6}: r2-luna-r1's RPO recipe and data (luna-A) with all language-layer weights
#           trained instead of a LoRA (FULL_FT=1; reference scores seeded from r2-luna-r1).        -> pass night-fullft
#   dagger  r2-dagger-r1: SFT executes its own sample (EXECUTE=first) on half A, gpt-6-luna labels every state; pairs
#           = Luna pick vs another candidate at all valid states, RPO's SFT term only on successful-episode pairs
#           (SFT_SUCCESS_ONLY=1), same recipe otherwise.                                            -> pass night-dagger
# Budget guard: nothing is submitted once piotrt's vlm usage (hyakusage) reaches $850.
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

gate() {  # gate <run>
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$1/frozen-config.json'))['total_updates'])")
  local g=$(LABEL=$1 CKPT=student/update-$u submit -A vlm $X --job-name=gt-$1 tillicum/gate.sbatch) || return 1
  log "gate $1 job $g"; wait_job $g
  log "gate $1: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $1"; return 2; }
}

train() {  # train <run> <data> <extra env...>: resume up to 2x (adopts a running job of the same name), then gate
  local run=$1 data=$2; shift 2; local j live
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then log "train $run: adopting job $live"; wait_job $live; continue; fi
    budget_ok || return 1
    j=$(env OBJECTIVE=dpo SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 OWRL_ACTOR=$SFT OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/$data RUN=$run "$@" \
        sbatch --parsable -A vlm $X --time=12:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data; $*) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete: $(cat $RUNS/$run/training-status.json 2>/dev/null | tr -d '\n' | head -c 300)"; return 1; }
  gate $run
}

evalpass() {  # evalpass <pass> <models,...>
  local n=$(tr ',' '\n' <<<"$2" | wc -l) j EJ=""
  for B in online-mind2web deepshop; do
    j=$(MODELS=$2 PASS=$1 BENCH=$B TEMP=1.0 submit -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-$1-$B tillicum/eval_concurrent.sbatch) || return 1
    EJ="$EJ $j"; log "eval $1 $B ($2) job $j"
  done
  for j in $EJ; do wait_job $j; done
  log "eval $1 done"
}

fullft() {
  local ok=""
  ( train r2-luna-r1-fullft-lr1e-6 luna-A FULL_FT=1 LR=1e-6 ) &
  ( train r2-luna-r1-fullft-lr5e-6 luna-A FULL_FT=1 LR=5e-6 ) &
  wait
  for lr in 1e-6 5e-6; do [ -f $M/r2-luna-r1-fullft-lr$lr/merge-manifest.json ] && ok="$ok,r2-luna-r1-fullft-lr$lr"; done
  [ -n "$ok" ] || { log "FATAL fullft: no model passed"; return 1; }
  evalpass night-fullft base,r2-luna-r1$ok
}

dagger() {
  # collection (resubmit incomplete shards up to 2x)
  for attempt in 1 2 3; do
    for j in $(awk '{print $2}' $G/jobs-A.txt); do wait_job $j; done
    local todo=""; for s in 42 43; do for k in 1 2; do complete_shard dagger-A-s$s-shard$k || todo="$todo $s:$k"; done; done
    [ -z "$todo" ] && break
    [ $attempt = 3 ] && { log "FATAL dagger collection incomplete:$todo"; return 1; }
    : > $G/jobs-A.txt
    for sk in $todo; do s=${sk%:*}; k=${sk#*:}
      j=$(ACTOR=$SFT TEACHER=luna EXECUTE=first SEED=$s NAME=dagger-A-s$s-shard$k PARALLEL=16 TASKS=$L/tasks-A-shard$k.jsonl LUNA_MAX_USD=25 \
          submit -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-dgA$s-$k tillicum/c2_collect.sbatch) || return 1
      echo "dagger-A-s$s-shard$k $j" >> $G/jobs-A.txt; log "dagger resubmit shard s$s-$k job $j"
    done
  done
  log "dagger collection complete: $(for s in 42 43; do for k in 1 2; do python3 -c "import json;a=json.load(open('$C/dagger-A-s$s-shard$k/dataset-audit.json'));print('s$s-$k', a['successful_tasks_with_usable_turns'], end=' ')"; done; done)"
  # export all valid episodes, teacher labels, build per seed
  for s in 42 43; do
    for k in 1 2; do [ -f $C/dagger-A-s$s-all-shard$k/dataset-audit.json ] || OMP_NUM_THREADS=1 PYTHONPATH=$PWD $PY scripts/export_failed_turns.py --include-successful \
        --collection $C/dagger-A-s$s-shard$k --out $C/dagger-A-s$s-all-shard$k 2>&1 | grep -v Warn | tail -1 | cut -c1-200; done
    python3 scripts/teacher_labels_from_selections.py --collection $C/dagger-A-s$s-shard1 $C/dagger-A-s$s-shard2 --out $G/labels-A-s$s.jsonl
  done

  for s in 42 43; do
    ( [ -f $D/dagger-A-s$s/train-config.json ] && exit 0
      j=$(NAME=dagger-A-s$s-all OUTNAME=dagger-A-s$s IDPREFIX=DA$s NSHARDS=2 RELABEL=$G/labels-A-s$s.jsonl RELABEL_NAME=gpt-6-luna ALLOW_FAILED=1 \
          submit -A vlm $X --job-name=bld-DA$s tillicum/build_round2_data.sbatch) || exit 1
      log "build dagger-A-s$s job $j"; wait_job $j
      [ -f $D/dagger-A-s$s/train-config.json ] || { log "FATAL build dagger-A-s$s (logs/round2-data-$j.out)"; exit 1; }
      log "built dagger-A-s$s: $(tr -d '\n ' < $D/dagger-A-s$s/filter-report.json | head -c 250)" ) &
  done
  wait
  [ -f $D/dagger-A-s42/train-config.json ] && [ -f $D/dagger-A-s43/train-config.json ] || { log "FATAL dagger data"; return 1; }
  if [ ! -f $D/dagger-A/train-config.json ]; then
    j=$(A=dagger-A-s42 B=dagger-A-s43 OUT=dagger-A submit -A vlm $X --job-name=cmb-dagger-A tillicum/build_combined_data.sbatch) || return 1
    log "combine dagger-A job $j"; wait_job $j
    [ -f $D/dagger-A/train-config.json ] || { log "FATAL combine dagger-A"; return 1; }
  fi
  python3 -c "
import json,collections
c=collections.Counter(json.loads(l).get('terminal_success') for l in open('$D/dagger-A/joint_pairs.train.jsonl'))
print('dagger-A pairs by terminal_success', dict(c))"
  train r2-dagger-r1 dagger-A SFT_SUCCESS_ONLY=1 LR=5e-5 || return 1
  evalpass night-dagger base,r2-luna-r1,r2-dagger-r1
}

log "overnight start"
fullft > >(sed -u 's/^/[fullft] /') 2>&1 & A=$!
dagger > >(sed -u 's/^/[dagger] /') 2>&1 & B=$!
wait $A; wait $B
log "OVERNIGHT_DONE"
