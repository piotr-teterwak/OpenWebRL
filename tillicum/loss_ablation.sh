#!/usr/bin/env bash
# Round 2 loss ablation (2026-11-19, Piotr: "launch a and b"): which loss term causes Round 2's drop?
#   (a) r2-sol2-r2-dpoonly  Round 2 exactly, SFT weight 0 (job 501790)
#   (b) r2-sol2-r2-sftonly  SFT only on Round 2's successful-episode chosen responses (sol2-Br1-succ, 2,254; job 501791)
#   (c) r2-sol2-r2-action (action-only DPO scoring) was launched as job 501851 and paused (cancelled) per Piotr
# After training: gate each, then 2 eval passes (la-a, la-b) of Round 1, Round 2, (a), (b) in one job per pass x bench
# (OM2W 12 browsers, DeepShop 6), then the paired comparison -> tillicum/results/iterative-rounds/loss-ablation.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; X="--exclude=g001,g008,g011,g013,g015,g021"; ERR=/gpfs/scrubbed/piotrt/owrl/c2-collect/loss-ablation.sbatch.err
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
sub() { local j; j=$(sbatch --parsable "$@" 2>>$ERR); [ -n "$j" ] || { log "FATAL sbatch refused: $* (see $ERR)" >&2; exit 1; }; echo $j; }
wait_jobs 501790 501791
for r in r2-sol2-r2-dpoonly r2-sol2-r2-sftonly; do
  [ -f $RUNS/$r/student/complete.json ] || { log "FATAL $r did not complete"; exit 1; }
  python3 - $RUNS/$r <<'PY'
import json, sys
r = sys.argv[1]; v = [json.loads(l) for l in open(f'{r}/student/metrics.jsonl') if '"val/' in l]
f, l = v[0], v[-1]
print(r.split('/')[-1], 'updates', l['updates'], ' '.join(f"{k[4:]} {f.get(k, float('nan')):.3f}->{l[k]:.3f}" for k in
      ['val/C2/winner_ce', 'val/Piotr/winner_ce', 'val/C2/action_ranking', 'val/Piotr/action_ranking'] if k in l))
PY
done
G=""
for r in r2-sol2-r2-dpoonly r2-sol2-r2-sftonly; do
  [ -f $RUNS/$r/gate-passed.json ] && continue
  u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$r/frozen-config.json'))['total_updates'])")
  g=$(LABEL=$r CKPT=student/update-$u sub -A vlm $X --mem=160G --job-name=gt-$r tillicum/gate.sbatch); log "gate $r job $g"; G="$G $r:$g"
done
for rg in $G; do r=${rg%:*}; g=${rg#*:}; wait_jobs $g; line=$(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1); log "gate $r: $line"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && echo "{\"job\": $g}" > $RUNS/$r/gate-passed.json; done
M=r2-sol-r1-all-m0.3,r2-sol2-r2
for r in r2-sol2-r2-dpoonly r2-sol2-r2-sftonly; do [ -f $RUNS/$r/gate-passed.json ] && M="$M,$r" || log "NOT EVALUATED: $r failed its gate"; done
n=$(tr ',' '\n' <<<"$M" | wc -l); EJ=""
for ps in la-a la-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  j=$(MODELS=$M PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sub -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $j"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B ($M) job $j"
done; done
wait_jobs $EJ
r=$(sub -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=la-result \
      --output=/gpfs/scrubbed/piotrt/owrl/logs/la-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; \
      PASSES=la-a,la-b MODELS=R1=r2-sol-r1-all-m0.3,R2=r2-sol2-r2,dpo_only=r2-sol2-r2-dpoonly,sft_only=r2-sol2-r2-sftonly \
      PAIRS=R2-R1,dpo_only-R1,sft_only-R1,dpo_only-R2,sft_only-R2,dpo_only-sft_only \
      python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/loss-ablation.txt")
log "result job $r"; wait_jobs $r; cat tillicum/results/iterative-rounds/loss-ablation.txt
log "LOSS_ABLATION_DONE"
