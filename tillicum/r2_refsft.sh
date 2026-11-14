#!/usr/bin/env bash
# Round 2 with the SFT model as DPO reference (2026-11-14, Piotr: "round 2 used round 1 as reference policy? That seems
# worse than sft as reference" -> "do both, don't evaluate only if val improves"). Identical to r2-sol2-r2 (start from
# C2 = r2-sol-r1-all-m0.3, data sol2-Br1, DPO b 0.1 + SFT on successful episodes, lr 5e-5, 2 epochs, seed 42) except
# OWRL_REFERENCE = SFT. Then the gate (length-cap sanity check only) and, whatever validation shows, 2 eval passes
# (rs-a, rs-b) x OM2W + DeepShop at T 1.0 with C2, r2 and this run in the same window. Nothing submitted at >= $CAP.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; D=/gpfs/scrubbed/piotrt/owrl/joint-data; E=/gpfs/scrubbed/piotrt/owrl/eval
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT; C2=/gpfs/scrubbed/piotrt/owrl/eval/merged/r2-sol-r1-all-m0.3
X="--exclude=g001,g008,g011,g013,g015,g021"; CAP=${CAP:-975}; run=r2-sol2-r2-refsft; LOGERR=/gpfs/scrubbed/piotrt/owrl/c2-collect/refsft-sbatch.err
log() { echo "$(date '+%F %T') $*"; }
usage() { hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); print a[1]; exit}'; }
budget_ok() { local u=$(usage); [ -z "$u" ] && return 0; python3 -c "import sys; sys.exit(0 if float('$u') < $CAP else 1)" && return 0; log "BUDGET GUARD: \$$u >= \$$CAP"; return 1; }
wait_jobs() { while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 120; done; }
[ -d $C2 ] || { log "FATAL no merged C2 at $C2"; exit 1; }
log "refsft start; vlm usage \$$(usage), cap \$$CAP"
for attempt in 1 2 3; do
  [ -f $RUNS/$run/student/complete.json ] && break
  budget_ok || exit 1
  j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=2 SEED=42 SFT_SUCCESS_ONLY=1 OWRL_ACTOR=$C2 OWRL_REFERENCE=$SFT OWRL_JOINT_DATA=$D/sol2-Br1 RUN=$run \
      sbatch --parsable -A vlm $X --time=11:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>>$LOGERR)
  [ -n "$j" ] || { log "FATAL sbatch refused (see $LOGERR)"; exit 1; }
  log "train $run attempt $attempt job $j"; wait_jobs $j
done
[ -f $RUNS/$run/student/complete.json ] || { log "FATAL train incomplete"; exit 1; }
python3 - $RUNS/$run $RUNS/r2-sol2-r2 <<'PYEOF'
import json, sys
for r in sys.argv[1:]:
    c = json.load(open(f'{r}/frozen-config.json')); vals = [json.loads(l) for l in open(f'{r}/student/metrics.jsonl') if '"val/' in l]
    f, l = vals[0], vals[-1]; print(r.split('/')[-1], 'reference', c.get('reference_actor'))
    for k in ['val/C2/winner_ce', 'val/C2/action_ranking', 'val/Piotr/winner_ce', 'val/Piotr/action_ranking']:
        print(f'   {k:26s} {f.get(k, float("nan")):.3f} -> {l.get(k, float("nan")):.3f}')
PYEOF
if [ ! -f $RUNS/$run/gate-passed.json ]; then
  u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  for attempt in 1 2; do
    budget_ok || exit 1
    g=$(LABEL=$run CKPT=student/update-$u sbatch --parsable -A vlm $X --mem=160G --job-name=gt-$run tillicum/gate.sbatch 2>/dev/null)
    log "gate job $g"; wait_jobs $g; line=$(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1); log "gate: $line"
    [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && { echo "{\"job\": $g}" > $RUNS/$run/gate-passed.json; break; }
    [ -n "$line" ] && { log "GATE FAIL (checkpoint broken: length cap)"; exit 1; }
  done
  [ -f $RUNS/$run/gate-passed.json ] || { log "GATE FAIL"; exit 1; }
fi
EJ=""
for ps in rs-a rs-b; do for B in online-mind2web deepshop; do
  budget_ok || break 2
  j=$(MODELS=r2-sol-r1-all-m0.3,r2-sol2-r2,$run PASS=$ps BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:3 --cpus-per-task=24 --mem=192G --time=$([ $B = deepshop ] && echo 02:00:00 || echo 02:30:00) --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch 2>>$LOGERR)
  EJ="$EJ $j"; mkdir -p $E/pass$ps; log "eval $ps $B job $j"
done; done
[ -n "$EJ" ] && wait_jobs $EJ
PASSES=rs-a,rs-b MODELS=C2=r2-sol-r1-all-m0.3,r2=r2-sol2-r2,refsft=$run PAIRS=refsft-r2,refsft-C2,r2-C2 python3 tillicum/round2_result.py | tee tillicum/results/luna-iter/r2-refsft.txt
log "REFSFT_DONE"
