#!/usr/bin/env bash
# Iterative RPO v2 (2026-11-12): the C2 lesson applied to every round. Rollouts are Luna-steered (as in luna-iter); pairs
# come from every valid state with differing candidates, successful AND failed episodes, chosen = Luna's executed
# (pre-execution) pick, no margin filter; RPO's SFT term only on successful-episode pairs (SFT_SUCCESS_ONLY=1).
# Recipe otherwise r2-luna-r1's (beta 0.1, SFT weight 1, lr 5e-5, 2 epochs, LoRA r16).
#   step 0  r2-luna-v2-r1    from SFT on the existing half-A Luna collections (luna-A-s{42,43})
#           eval: 2 passes (v2-r1a, v2-r1b) of base, C2 (r2-sol-r1-all-m0.3), r2-luna-v2-r1; OM2W + DeepShop
#   rule    step 1 runs only if r2-luna-v2-r1 - SFT, pooled over both passes and both benchmarks (paired per task),
#           has a 95% bootstrap lower bound > 0 (fixed before any result)
#   step 1  r2-luna-v2-r2    from r2-luna-v2-r1 (reference r2-luna-v2-r1) on half B collected by r2-luna-v2-r1 + Luna
#           r2-luna-v2-ctrl  one round from SFT on half A + half B (existing luna-Bsft collections), same pair recipe
#           eval: 2 passes (v2-r2a, v2-r2b) of base, r2-luna-v2-r1, r2-luna-v2-r2, r2-luna-v2-ctrl
# Budget: nothing is submitted once piotrt's vlm usage reaches $CAP (950); urgent upgrades stop above $URGENT_CAP (850).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
C=/gpfs/scrubbed/piotrt/owrl/c2-collect; V=$C/iter-v2; L=$C/luna-iter; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged; E=/gpfs/scrubbed/piotrt/owrl/eval
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
PY=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/venv/bin/python
X="--exclude=g001,g008,g011,g013,g015,g021"; CAP=${CAP:-950}; URGENT_CAP=${URGENT_CAP:-850}
mkdir -p $V
log() { echo "$(date '+%F %T') $*"; }

usage() { hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); print a[1]; exit}'; }
below() { local u=$(usage); [ -z "$u" ] && return 0; python3 -c "import sys; sys.exit(0 if float('$u') < $1 else 1)"; }
budget_ok() { below $CAP && return 0; log "BUDGET GUARD: vlm usage \$$(usage) >= \$$CAP, not submitting"; return 1; }
submit() { budget_ok || return 1; sbatch --parsable "$@" 2>/dev/null; }

wait_job() {  # urgent after 30 min pending, while usage < URGENT_CAP
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
complete_shard() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$C/$1/dataset-audit.json'))['complete_collection'] else 1)" 2>/dev/null; }

alldata() {  # alldata <collection prefix e.g. luna-A> <out e.g. luna-Aall> <id tag>: all valid states, executed Luna pick, no margin
  local pre=$1 out=$2 tag=$3 pids="" s k
  [ -f $D/$out/train-config.json ] && return 0
  for s in 42 43; do
    for k in 1 2; do [ -f $C/$pre-s$s-all-shard$k/dataset-audit.json ] || OMP_NUM_THREADS=1 PYTHONPATH=$PWD $PY scripts/export_failed_turns.py \
        --include-successful --collection $C/$pre-s$s-shard$k --out $C/$pre-s$s-all-shard$k 2>&1 | grep -v Warn | tail -1 | cut -c1-200; done
    python3 scripts/teacher_labels_from_selections.py --executed --collection $C/$pre-s$s-shard1 $C/$pre-s$s-shard2 --out $V/labels-$pre-s$s.jsonl
  done
  for s in 42 43; do
    ( [ -f $D/$out-s$s/train-config.json ] && exit 0
      live=$(squeue -h -u $USER -n bld-$tag$s -o %i | head -1)
      if [ -n "$live" ]; then log "build $out-s$s: adopting job $live"; wait_job $live; [ -f $D/$out-s$s/train-config.json ] && exit 0; fi
      j=$(NAME=$pre-s$s-all OUTNAME=$out-s$s IDPREFIX=$tag$s NSHARDS=2 RELABEL=$V/labels-$pre-s$s.jsonl RELABEL_NAME=gpt-6-luna ALLOW_FAILED=1 \
          submit -A vlm $X --job-name=bld-$tag$s tillicum/build_round2_data.sbatch) || exit 1
      log "build $out-s$s job $j"; wait_job $j
      [ -f $D/$out-s$s/train-config.json ] || { log "FATAL build $out-s$s (logs/round2-data-$j.out)"; exit 1; } ) & pids="$pids $!"
  done
  for p in $pids; do wait $p; done   # explicit pids: a bare wait also waits for the branch's own sed log prefixer
  [ -f $D/$out-s42/train-config.json ] && [ -f $D/$out-s43/train-config.json ] || { log "FATAL $out data"; return 1; }
  combine $out-s42 $out-s43 $out
}

combine() {
  [ -f $D/$3/train-config.json ] && return 0
  local live=$(squeue -h -u $USER -n cmb-$3 -o %i | head -1) j
  if [ -n "$live" ]; then wait_job $live; else
    j=$(A=$1 B=$2 OUT=$3 submit -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch) || return 1
    log "combine $1 + $2 -> $3 job $j"; wait_job $j; fi
  [ -f $D/$3/train-config.json ] || { log "FATAL combine $3"; return 1; }
  log "combined $3: $(python3 -c "
import json,collections
c=json.load(open('$D/$3/train-config.json'));t=collections.Counter(json.loads(l).get('terminal_success') for l in open('$D/$3/joint_pairs.train.jsonl'))
print(c['train_examples'],'pairs, terminal_success',dict(t))")"
}

train() {  # train <run> <data> <actor> <reference>: resume up to 2x (adopts a running job), then gate (recorded)
  local run=$1 data=$2 actor=$3 ref=$4 j live
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then log "train $run: adopting job $live"; wait_job $live; continue; fi
    budget_ok || return 1
    j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$actor OWRL_REFERENCE=$ref OWRL_JOINT_DATA=$D/$data RUN=$run \
        sbatch --parsable -A vlm $X --time=14:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data, actor $(basename $actor), ref $(basename $ref)) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  [ -f $RUNS/$run/gate-passed.json ] && return 0
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])") g
  for attempt in 1 2; do
    g=$(LABEL=$run CKPT=student/update-$u submit -A vlm $X --mem=160G --job-name=gt-$run tillicum/gate.sbatch) || return 1
    log "gate $run job $g"; wait_job $g; log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
    if [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ]; then echo "{\"job\": $g}" > $RUNS/$run/gate-passed.json; return 0; fi
    grep -q '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out && break   # measured fail: do not retry
    log "gate $run crashed before measuring; retrying"
  done
  log "GATE FAIL $run"; return 2
}

evalpasses() {  # evalpasses <models,...> <pass>...: all passes submitted together
  local models=$1; shift; local n=$(tr ',' '\n' <<<"$models" | wc -l) j EJ="" ps B
  for ps in "$@"; do for B in online-mind2web deepshop; do
    [ -f $E/pass$ps/.done-$B ] && continue
    j=$(MODELS=$models PASS=$ps BENCH=$B TEMP=1.0 submit -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch) || return 1
    EJ="$EJ $ps:$B:$j"; log "eval $ps $B ($models) job $j"
  done; done
  for x in $EJ; do ps=${x%%:*}; j=${x##*:}; B=${x#*:}; B=${B%%:*}; wait_job $j; mkdir -p $E/pass$ps; touch $E/pass$ps/.done-$B; done
}

pooled() {  # pooled <model> <ref> <pass>...: prints "mean lo hi n" (paired per task within pass, pooled, 4000 bootstrap)
  local m=$1 r=$2; shift 2
  python3 - "$m" "$r" "$@" <<'EOF'
import json,glob,random,sys
m,r,passes=sys.argv[1],sys.argv[2],sys.argv[3:]
E='/gpfs/scrubbed/piotrt/owrl/eval'
def load(ps,x,b):
    out={}
    for f in glob.glob(f'{E}/pass{ps}/{x}/{b}/results/*.json'):
        j=json.load(open(f))
        if j.get('valid'): out[j['task_id']]=float(j.get('reward') or 0)>=0.5
    return out
d=[]
for ps in passes:
    for b in ['online-mind2web','deepshop']:
        a=load(ps,m,b); s=load(ps,r,b); d+=[a[k]-s[k] for k in set(a)&set(s)]
rng=random.Random(0); bs=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(4000))
print(f"{100*sum(d)/len(d):.2f} {100*bs[100]:.2f} {100*bs[3899]:.2f} {len(d)}")
EOF
}

log "iter v2 start (cap \$$CAP, urgent below \$$URGENT_CAP)"
# ---- step 0
alldata luna-A luna-Aall LAA || exit 1
train r2-luna-v2-r1 luna-Aall $SFT $SFT || exit 1
evalpasses base,r2-sol-r1-all-m0.3,r2-luna-v2-r1 v2-r1a v2-r1b || exit 1
read mean lo hi n <<<"$(pooled r2-luna-v2-r1 base v2-r1a v2-r1b)"
read cm clo chi cn <<<"$(pooled r2-sol-r1-all-m0.3 base v2-r1a v2-r1b)"
read vc vlo vhi vn <<<"$(pooled r2-luna-v2-r1 r2-sol-r1-all-m0.3 v2-r1a v2-r1b)"
log "STEP0 r2-luna-v2-r1 - SFT pooled (2 passes x 2 benchmarks): $mean [$lo, $hi] n=$n | C2 - SFT: $cm [$clo, $chi] | v2-r1 - C2: $vc [$vlo, $vhi]"
python3 -c "import sys; sys.exit(0 if float('$lo') > 0 else 1)" || { log "RULE: lower bound $lo <= 0 -> step 1 not run"; log "ITER_V2_DONE"; exit 0; }
log "RULE: lower bound $lo > 0 -> running step 1"
# ---- step 1
R1=$M/r2-luna-v2-r1
round2() {
  local jf=$V/jobs-v2Br1.txt s k j todo
  [ -f $jf ] || : > $jf
  for attempt in 1 2 3; do
    for j in $(awk '{print $2}' $jf); do wait_job $j; done
    todo=""; for s in 42 43; do for k in 1 2; do complete_shard luna-v2Br1-s$s-shard$k || todo="$todo $s:$k"; done; done
    [ -z "$todo" ] && break
    [ $attempt = 3 ] && { log "FATAL collect luna-v2Br1 incomplete:$todo"; return 1; }
    : > $jf
    for sk in $todo; do s=${sk%:*}; k=${sk#*:}
      j=$(ACTOR=$R1 TEACHER=luna SEED=$s NAME=luna-v2Br1-s$s-shard$k PARALLEL=16 TASKS=$L/tasks-B-shard$k.jsonl LUNA_MAX_USD=25 \
          submit -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-v2B$s-$k tillicum/c2_collect.sbatch) || return 1
      echo "luna-v2Br1-s$s-shard$k $j" >> $jf; log "collect luna-v2Br1 s$s-$k job $j"
    done
  done
  alldata luna-v2Br1 luna-v2Br1all LVB || return 1
  train r2-luna-v2-r2 luna-v2Br1all $R1 $R1
}
control() {
  alldata luna-Bsft luna-Bsftall LBA || return 1
  combine luna-Aall luna-Bsftall luna-v2ctrl || return 1
  train r2-luna-v2-ctrl luna-v2ctrl $SFT $SFT
}
round2 > >(sed -u 's/^/[r2] /') 2>&1 & A=$!
control > >(sed -u 's/^/[ctrl] /') 2>&1 & B=$!
wait $A; wait $B
MODELS=base,r2-luna-v2-r1; for r in r2-luna-v2-r2 r2-luna-v2-ctrl; do [ -f $RUNS/$r/gate-passed.json ] && MODELS="$MODELS,$r"; done
evalpasses $MODELS v2-r2a v2-r2b
for x in r2-luna-v2-r2 r2-luna-v2-ctrl; do [ -f $RUNS/$x/gate-passed.json ] || continue
  for y in base r2-luna-v2-r1 r2-luna-v2-ctrl; do [ $x = $y ] && continue; [ -f $RUNS/$y/gate-passed.json ] || [ $y = base ] || continue
    read a b c n <<<"$(pooled $x $y v2-r2a v2-r2b)"; log "STEP1 $x - $y pooled (2 passes x 2 benchmarks): $a [$b, $c] n=$n"; done; done
log "ITER_V2_DONE"
