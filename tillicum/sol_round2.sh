#!/usr/bin/env bash
# Round 2 with Sol steering + data-matched control (2026-11-12, Piotr). Waits for sol_c2_plain.sh's evals, then picks the
# round-1 recipe by a rule fixed here before the result:
#   plain  r2-solp-r1-all       (Sol plain prompt, no margin)              -- default
#   prob   r2-sol-r1-all-m0.3   (C2: Sol probability prompt, margin 0.3)   -- only if C2 - plain, pooled over passes c2p-a/b
#                                                                             and both benchmarks, has 95% lower bound > 0
# Collections on half B, 2 seeds x 2 shards, Sol steering (its executed pre-execution pick is the label, so nothing is
# relabelled), skip_identical (no Sol call when all 5 candidates share one action), TEACHER_PROBS for the prob recipe:
#   sol2-Br1   actor = the round-1 model      -> r2-sol2-r2   (from round 1, reference round 1, on half B)
#   sol2-Bsft  actor = SFT                    -> r2-sol2-ctrl (one round from SFT on round 1's half-A pairs + half B)
# Pairs: Sol pick vs another candidate, all valid states, DPO-only on failed-episode pairs, margin only for prob.
# Eval: 2 passes (s2-a, s2-b) of base, round 1, round 2, control; OM2W + DeepShop, T 1.0.
# Budget follows hyakusage: nothing submitted within $50 of the cap, no urgent upgrades within $150 of it.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
C=/gpfs/scrubbed/piotrt/owrl/c2-collect; V=$C/sol2; L=$C/luna-iter; S=$C/sol-relabel; D=/gpfs/scrubbed/piotrt/owrl/joint-data
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged; E=/gpfs/scrubbed/piotrt/owrl/eval
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT
PY=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/openwebrl-runtime/arm-reproduction/venv/bin/python
X="--exclude=g001,g008,g011,g013,g015,g021"
mkdir -p $V
log() { echo "$(date '+%F %T') $*"; }
budget() { hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); gsub(/ /,"",a[2]); print a[1], a[2]; exit}'; }
room() {  # room <margin $>: usage < cap - margin
  local u c; read u c <<<"$(budget)"; [ -z "$u" ] && return 0
  python3 -c "import sys; sys.exit(0 if float('$u') < float('$c') - $1 else 1)"
}
budget_ok() { room 50 && return 0; log "BUDGET GUARD: vlm $(budget | tr ' ' '/'), not submitting"; return 1; }
submit() { budget_ok || return 1; sbatch --parsable "$@" 2>/dev/null; }
upgrade() {  # upgrade <jobs...>: pending > 30 min -> urgent while there is room
  local j w
  for j in "$@"; do
    [ "$(squeue -h -j $j -o '%T %q' 2>/dev/null)" = "PENDING normal" ] || continue
    w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
    [ $w -gt 1800 ] && room 150 && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
  done
}
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do upgrade "$@"; sleep 120; done; }
complete_shard() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$C/$1/dataset-audit.json'))['complete_collection'] else 1)" 2>/dev/null; }

pooled() {  # pooled <model> <ref> <pass>...: mean lo hi n
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

collect() {  # collect <prefix> <actor> <tag>: half B, Sol steering
  local pre=$1 actor=$2 tag=$3 jf=$V/jobs-$1.txt s k j todo
  [ -f $jf ] || : > $jf
  for attempt in 1 2 3; do
    wait_jobs $(awk '{print $2}' $jf)
    todo=""; for s in 42 43; do for k in 1 2; do complete_shard $pre-s$s-shard$k || todo="$todo $s:$k"; done; done
    [ -z "$todo" ] && { log "collect $pre complete"; return 0; }
    [ $attempt = 3 ] && { log "FATAL collect $pre incomplete:$todo"; return 1; }
    : > $jf
    for sk in $todo; do s=${sk%:*}; k=${sk#*:}
      j=$(ACTOR=$actor TEACHER=sol TEACHER_PROBS=$PROBS SKIP_IDENTICAL=1 SEED=$s NAME=$pre-s$s-shard$k PARALLEL=16 TASKS=$L/tasks-B-shard$k.jsonl LUNA_MAX_USD=70 \
          submit -A vlm $X --gpus=h200:1 --cpus-per-task=8 --mem=240G --time=12:00:00 --job-name=lu-$tag$s-$k tillicum/c2_collect.sbatch) || return 1
      echo "$pre-s$s-shard$k $j" >> $jf; log "collect $pre s$s-$k job $j"
    done
  done
}

data() {  # data <prefix> <id tag>: all valid states, Sol executed pick (+ scores), margin for prob -> $D/<prefix>
  local pre=$1 tag=$2 s k j pids=""
  [ -f $D/$pre/train-config.json ] && return 0
  for s in 42 43; do
    for k in 1 2; do [ -f $C/$pre-s$s-all-shard$k/dataset-audit.json ] || OMP_NUM_THREADS=1 PYTHONPATH=$PWD $PY scripts/export_failed_turns.py \
        --include-successful --collection $C/$pre-s$s-shard$k --out $C/$pre-s$s-all-shard$k 2>&1 | grep -v Warn | tail -1 | cut -c1-160; done
    python3 scripts/teacher_labels_from_selections.py --executed --collection $C/$pre-s$s-shard1 $C/$pre-s$s-shard2 --out $V/labels-$pre-s$s.jsonl
  done
  for s in 42 43; do
    ( [ -f $D/$pre-s$s/train-config.json ] && exit 0
      j=$(NAME=$pre-s$s-all OUTNAME=$pre-s$s IDPREFIX=$tag$s NSHARDS=2 RELABEL=$V/labels-$pre-s$s.jsonl RELABEL_NAME=gpt-6-sol ALLOW_FAILED=1 MARGIN=$MARGIN \
          submit -A vlm $X --job-name=bld-$tag$s tillicum/build_round2_data.sbatch) || exit 1
      log "build $pre-s$s job $j"; wait_jobs $j
      [ -f $D/$pre-s$s/train-config.json ] || { log "FATAL build $pre-s$s (logs/round2-data-$j.out)"; exit 1; } ) & pids="$pids $!"
  done
  for p in $pids; do wait $p; done
  [ -f $D/$pre-s42/train-config.json ] && [ -f $D/$pre-s43/train-config.json ] || { log "FATAL $pre data"; return 1; }
  combine $pre-s42 $pre-s43 $pre
}

combine() {
  [ -f $D/$3/train-config.json ] && return 0
  local j=$(A=$1 B=$2 OUT=$3 submit -A vlm $X --job-name=cmb-$3 tillicum/build_combined_data.sbatch) || return 1
  log "combine $1 + $2 -> $3 job $j"; wait_jobs $j
  [ -f $D/$3/train-config.json ] || { log "FATAL combine $3"; return 1; }
  log "combined $3: $(python3 -c "
import json,collections
c=json.load(open('$D/$3/train-config.json'));t=collections.Counter(json.loads(l).get('terminal_success') for l in open('$D/$3/joint_pairs.train.jsonl'))
print(c['train_examples'],'pairs, terminal_success',dict(t))")"
}

train() {  # train <run> <data> <actor> <reference>, then gate
  local run=$1 data=$2 actor=$3 ref=$4 j live u g line
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then wait_jobs $live; continue; fi
    budget_ok || return 1
    j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$actor OWRL_REFERENCE=$ref OWRL_JOINT_DATA=$D/$data RUN=$run \
        sbatch --parsable -A vlm $X --time=14:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data, actor $(basename $actor), ref $(basename $ref)) attempt $attempt job $j"; wait_jobs $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  [ -f $RUNS/$run/gate-passed.json ] && return 0
  u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  for attempt in 1 2; do
    g=$(LABEL=$run CKPT=student/update-$u submit -A vlm $X --mem=160G --job-name=gt-$run tillicum/gate.sbatch) || return 1
    log "gate $run job $g"; wait_jobs $g; line=$(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1); log "gate $run: $line"
    [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && { echo "{\"job\": $g}" > $RUNS/$run/gate-passed.json; return 0; }
    [ -n "$line" ] && break
  done
  log "GATE FAIL $run"; return 2
}

log "sol round 2 start; budget $(budget | tr ' ' '/')"
# ---- recipe decision (waits for the plain-prompt C2 evals)
until grep -q SOL_C2_PLAIN_DONE $S/c2-plain.log 2>/dev/null; do
  grep -qE "FATAL|GATE FAIL" $S/c2-plain.log && { log "FATAL sol_c2_plain failed; no round 2"; exit 1; }
  sleep 300; done
read cm clo chi cn <<<"$(pooled r2-sol-r1-all-m0.3 r2-solp-r1-all c2p-a c2p-b)"
read pm plo phi pn <<<"$(pooled r2-solp-r1-all base c2p-a c2p-b)"
log "C2 - plain pooled: $cm [$clo, $chi] n=$cn | plain - SFT: $pm [$plo, $phi] n=$pn"
if python3 -c "import sys; sys.exit(0 if float('$clo') > 0 else 1)"; then
  R1RUN=r2-sol-r1-all-m0.3; PROBS=1; MARGIN=0.3; ADATA=sol-Aall-m0.3; log "RULE: C2 significantly better -> probability prompt + margin 0.3"
else
  R1RUN=r2-solp-r1-all; PROBS=; MARGIN=; ADATA=solp-Aall; log "RULE: plain recipe (C2 not significantly better)"
fi
R1=$M/$R1RUN
# ---- collections (round 2 first in submission order, so the budget guard favours it)
( collect sol2-Br1 $R1 s2B && data sol2-Br1 S2B && train r2-sol2-r2 sol2-Br1 $R1 $R1 ) > >(sed -u 's/^/[r2] /') 2>&1 & A=$!
sleep 30
( collect sol2-Bsft $SFT s2S && data sol2-Bsft S2S && combine $ADATA sol2-Bsft sol2-ctrl && train r2-sol2-ctrl sol2-ctrl $SFT $SFT ) > >(sed -u 's/^/[ctrl] /') 2>&1 & B=$!
wait $A; wait $B
MODELS=base,$R1RUN; for r in r2-sol2-r2 r2-sol2-ctrl; do [ -f $RUNS/$r/gate-passed.json ] && MODELS="$MODELS,$r"; done
n=$(tr ',' '\n' <<<"$MODELS" | wc -l); EJ=""
for ps in s2-a s2-b; do for B in online-mind2web deepshop; do
  [ -f $E/pass$ps/.done-$B ] && continue
  j=$(MODELS=$MODELS PASS=$ps BENCH=$B TEMP=1.0 submit -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch) || break
  EJ="$EJ $j"; log "eval $ps $B ($MODELS) job $j"; mkdir -p $E/pass$ps
done; done
wait_jobs $EJ
for ps in s2-a s2-b; do for B in online-mind2web deepshop; do touch $E/pass$ps/.done-$B; done; done
for x in r2-sol2-r2 r2-sol2-ctrl; do [ -f $RUNS/$x/gate-passed.json ] || continue
  for y in base $R1RUN r2-sol2-ctrl; do [ $x = $y ] && continue; [ $y = base ] || [ -f $RUNS/$y/gate-passed.json ] || [ $y = $R1RUN ] || continue
    read a b c n <<<"$(pooled $x $y s2-a s2-b)"; log "RESULT $x - $y pooled (2 passes x 2 benchmarks): $a [$b, $c] n=$n"; done; done
log "SOL_ROUND2_DONE"
