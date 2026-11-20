#!/usr/bin/env bash
# Mixed-pool Round 2 (r2c-mix; 2026-11-20): Round 2's loss on sol2-Bmix (Sol picks among 3 Round-1 + 3 SFT candidates
# at Round-1 states; 7,314 pairs), as a continuation of Round 1 (adapter + Adam, flat 2.5e-5, reference Round 1).
# Training job 503297, then gate, then 2 eval passes (mx-a, mx-b) of Round 1, r2c-rpo (same recipe, Round-1-only candidates)
# and r2c-mix (OM2W 12 browsers, DeepShop 6) -> tillicum/results/iterative-rounds/r2c-mix.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; X="--exclude=g001,g008,g011,g013,g015,g021"
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
wait_jobs 503297 501958
for r in r2c-mix r2c-rpo; do [ -f $RUNS/$r/student/complete.json ] || { log "FATAL $r incomplete"; exit 1; }; done
until [ -f $RUNS/r2c-rpo/gate-passed.json ] || ! ps -eo args | grep -q "^bash tillicum/loss_ablation.sh"; do sleep 180; done   # r2c-rpo is gated by loss_ablation.sh
r=r2c-mix; u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$r/frozen-config.json'))['total_updates'])")
g=$(LABEL=$r CKPT=student/update-$u sbatch --parsable -A vlm $X --mem=160G --job-name=gt-$r tillicum/gate.sbatch); log "gate $r job $g"; wait_jobs $g
log "gate: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
[ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL"; exit 1; }; echo "{\"job\": $g}" > $RUNS/$r/gate-passed.json
[ -f $RUNS/r2c-rpo/gate-passed.json ] || { log "r2c-rpo not gated; evaluating Round 1 + r2c-mix only"; }
M=r2-sol-r1-all-m0.3,r2c-mix; [ -f $RUNS/r2c-rpo/gate-passed.json ] && M=r2-sol-r1-all-m0.3,r2c-rpo,r2c-mix
n=$(tr ',' '\n' <<<"$M" | wc -l); EJ=""
for ps in mx-a mx-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  e=$(MODELS=$M PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sbatch --parsable -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $e"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B ($M) job $e"; done; done
wait_jobs $EJ
res=$(sbatch --parsable -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=mx-result \
  --output=/gpfs/scrubbed/piotrt/owrl/logs/mx-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; PASSES=mx-a,mx-b \
  MODELS=R1=r2-sol-r1-all-m0.3,R2c=r2c-rpo,mix=r2c-mix PAIRS=mix-R1,mix-R2c,R2c-R1 \
  python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/r2c-mix.txt"); wait_jobs $res
cat tillicum/results/iterative-rounds/r2c-mix.txt; log "R2C_MIX_DONE"
