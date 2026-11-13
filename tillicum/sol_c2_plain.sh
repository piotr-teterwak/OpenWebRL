#!/usr/bin/env bash
# C2 with Sol re-prompted plainly and no filter (2026-11-12). C2 (r2-sol-r1-all-m0.3) used Sol picks from the
# probability-prompt variant (serve_sol_selector --probabilities) and the 0.3 margin; Luna's labels came from the plain
# prompt. Here Sol relabels the same half-A states (successful + failed episodes, >= 2 distinct candidate actions) with
# the plain pinned ARM prompt, pairs are Sol pick vs any other candidate (no margin), DPO-only on failed-episode pairs
# (SFT_SUCCESS_ONLY=1), recipe otherwise C2's:
#   r2-solp-r1-all   evaluated in 2 passes (c2p-a, c2p-b) with base and C2, OM2W + DeepShop, T 1.0
# Budget: nothing is submitted once piotrt's vlm usage reaches $CAP (950); urgent upgrades stop above $URGENT_CAP (850).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
C=/gpfs/scrubbed/piotrt/owrl/c2-collect; S=$C/sol-relabel; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged; E=/gpfs/scrubbed/piotrt/owrl/eval
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
X="--exclude=g001,g008,g011,g013,g015,g021"; CAP=${CAP:-950}; URGENT_CAP=${URGENT_CAP:-850}
TASKS=$C/luna-iter/tasks-A.jsonl
log() { echo "$(date '+%F %T') $*"; }
usage() { hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); print a[1]; exit}'; }
below() { local u=$(usage); [ -z "$u" ] && return 0; python3 -c "import sys; sys.exit(0 if float('$u') < $1 else 1)"; }
budget_ok() { below $CAP && return 0; log "BUDGET GUARD: vlm usage \$$(usage) >= \$$CAP, not submitting"; return 1; }
submit() { budget_ok || return 1; sbatch --parsable "$@" 2>/dev/null; }
wait_job() {
  local j=${1:-} st state elapsed q w; [ -n "$j" ] || { log "wait_job: empty job id"; return 1; }
  while st=$(squeue -h -j $j -o "%T %M %q" 2>/dev/null) && [ -n "$st" ]; do
    read state elapsed q <<<"$st"
    if [ "$state" = PENDING ] && [ "$q" != urgent ]; then
      w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
      [ $w -gt 1800 ] && below $URGENT_CAP && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
    fi
    sleep 120
  done
}
wait_all() { local j; for j in "$@"; do wait_job $j; done; }   # each job is upgraded when its turn comes; see upgrade_all
upgrade_all() {  # all listed pending jobs past 30 min (wait_job only watches one at a time)
  local j w
  for j in "$@"; do
    [ "$(squeue -h -j $j -o '%T %q' 2>/dev/null)" = "PENDING normal" ] || continue
    w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
    [ $w -gt 1800 ] && below $URGENT_CAP && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
  done
}
wait_many() { while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do upgrade_all "$@"; sleep 120; done; }

build() {  # build <out> <collection> <prefix> <labels> [allow_failed]
  [ -f $D/$1/train-config.json ] && return 0
  local live=$(squeue -h -u $USER -n bld-$3 -o %i | head -1) j
  if [ -n "$live" ]; then wait_job $live; else
    j=$(NAME=$2 OUTNAME=$1 IDPREFIX=$3 NSHARDS=2 RELABEL=$4 RELABEL_NAME=gpt-6-sol-plain ALLOW_FAILED=${5:-} \
        submit -A vlm $X --job-name=bld-$3 tillicum/build_round2_data.sbatch) || return 1
    log "build $1 ($2) job $j"; wait_job $j; fi
  [ -f $D/$1/train-config.json ] || { log "FATAL build $1"; return 1; }
}
combine() {
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 submit -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch) || return 1
  log "combine $1 + $2 -> $3 job $j"; wait_job $j
  [ -f $D/$3/train-config.json ] || { log "FATAL combine $3"; return 1; }
  log "combined $3: $(python3 -c "
import json,collections
c=json.load(open('$D/$3/train-config.json'));t=collections.Counter(json.loads(l).get('terminal_success') for l in open('$D/$3/joint_pairs.train.jsonl'))
print(c['train_examples'],'pairs, terminal_success',dict(t))")"
}

log "sol C2-plain start (cap \$$CAP, urgent below \$$URGENT_CAP)"
# 1. plain-prompt Sol relabels (no PROBS): 4 MIG jobs, API only
J=""
for d in A-s42 A-s43 Afail-s42 Afail-s43; do
  out=$S/plain-$d; mkdir -p $out
  [ "$(wc -l < $out/labels.jsonl 2>/dev/null || echo 0)" -ge "$(wc -l < $S/$d-distinct.jsonl)" ] && continue
  live=$(squeue -h -u $USER -n solp-$d -o %i | head -1)
  if [ -n "$live" ]; then J="$J $live"; continue; fi
  cap=$([[ $d == Afail* ]] && echo 80 || echo 60)
  j=$(MODEL=gpt-6-sol LUNA_MAX_USD=$cap DATASET=$S/$d-distinct.jsonl OUT=$out/labels.jsonl TASKS=$TASKS \
      submit -A vlm --time=08:00:00 --job-name=solp-$d tillicum/relabel_luna.sbatch) || exit 1
  J="$J $j"; log "relabel plain $d job $j"
done
[ -n "$J" ] && wait_many $J
for d in A-s42 A-s43 Afail-s42 Afail-s43; do
  log "relabel plain $d: $(wc -l < $S/plain-$d/labels.jsonl) labels of $(wc -l < $S/$d-distinct.jsonl); $(cat $S/plain-$d/sol-api/usage.json 2>/dev/null)"
  python3 - "$S/$d/labels.jsonl" "$S/plain-$d/labels.jsonl" <<'EOF'
import json,sys
a={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}; b={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[2])}
k=set(a)&set(b); same=sum(a[i]['selected_index']==b[i]['selected_index'] for i in k)
luna=sum(b[i]['selected_index']==b[i]['collection_index'] for i in k)
print(f"  plain vs probability-prompt Sol pick agree {100*same/len(k):.1f}% (n={len(k)}); plain Sol agrees with Luna's executed pick {100*luna/len(k):.1f}%")
EOF
done
# 2. pairs: Sol plain pick vs any other candidate, no margin
pids=""
for s in 42 43; do
  ( build solp-A-s$s luna-A-s$s SPA$s $S/plain-A-s$s/labels.jsonl ) & pids="$pids $!"
  ( build solp-Afail-s$s luna-Afail-s$s SPF$s $S/plain-Afail-s$s/labels.jsonl 1 ) & pids="$pids $!"
done
for p in $pids; do wait $p; done
for x in solp-A-s42 solp-A-s43 solp-Afail-s42 solp-Afail-s43; do [ -f $D/$x/train-config.json ] || { log "FATAL data $x"; exit 1; }; done
combine solp-A-s42 solp-A-s43 solp-A || exit 1
combine solp-Afail-s42 solp-Afail-s43 solp-Afail || exit 1
combine solp-A solp-Afail solp-Aall || exit 1
# 3. train (C2's recipe), gate
run=r2-solp-r1-all
for attempt in 1 2 3; do
  [ -f $RUNS/$run/student/complete.json ] && break
  live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
  if [ -n "$live" ]; then wait_job $live; continue; fi
  budget_ok || exit 1
  j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$SFT OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/solp-Aall RUN=$run \
      sbatch --parsable -A vlm $X --time=14:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
  log "train $run attempt $attempt job $j"; wait_job $j
done
[ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; exit 1; }
if [ ! -f $RUNS/$run/gate-passed.json ]; then
  u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  for attempt in 1 2; do
    g=$(LABEL=$run CKPT=student/update-$u submit -A vlm $X --mem=160G --job-name=gt-$run tillicum/gate.sbatch) || exit 1
    log "gate $run job $g"; wait_job $g; line=$(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1); log "gate $run: $line"
    [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && { echo "{\"job\": $g}" > $RUNS/$run/gate-passed.json; break; }
    [ -n "$line" ] && { log "GATE FAIL $run"; exit 1; }
  done
  [ -f $RUNS/$run/gate-passed.json ] || { log "GATE FAIL $run"; exit 1; }
fi
# 4. eval: 2 passes with base and C2
EJ=""
for ps in c2p-a c2p-b; do for B in online-mind2web deepshop; do
  [ -f $E/pass$ps/.done-$B ] && continue
  j=$(MODELS=base,r2-sol-r1-all-m0.3,$run PASS=$ps BENCH=$B TEMP=1.0 submit -A vlm $X --gpus=h200:3 --cpus-per-task=24 --mem=192G --time=06:00:00 --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch) || exit 1
  EJ="$EJ $j"; log "eval $ps $B job $j"; mkdir -p $E/pass$ps
done; done
[ -n "$EJ" ] && wait_many $EJ
for ps in c2p-a c2p-b; do for B in online-mind2web deepshop; do touch $E/pass$ps/.done-$B; done; done
log "SOL_C2_PLAIN_DONE"
