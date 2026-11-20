#!/usr/bin/env bash
# Round 1 with action-only DPO scoring (objective dpo_action; 2026-11-19, Piotr: "lets try round 1 with dpo action").
# Identical to r2-sol-r1-all-m0.3 (Round 1) except the DPO score sums only the tool-call tokens (reasoning masked);
# beta auto-calibrated by the trainer. Training job 501849, then gate, then 2 eval passes (ra-a, ra-b) of SFT, Round 1 and
# this run (OM2W 12 browsers, DeepShop 6), then the paired comparison -> tillicum/results/iterative-rounds/r1-action.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; X="--exclude=g001,g008,g011,g013,g015,g021"; r=r2-sol-r1-all-m0.3-action
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
wait_jobs 501849
[ -f $RUNS/$r/student/complete.json ] || { log "FATAL training incomplete"; exit 1; }
log "resolved objective: $(cat $RUNS/$r/student/resolved-objective.json 2>/dev/null | tr -d '\n' | cut -c1-200)"
u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$r/frozen-config.json'))['total_updates'])")
g=$(LABEL=$r CKPT=student/update-$u sbatch --parsable -A vlm $X --mem=160G --job-name=gt-$r tillicum/gate.sbatch); log "gate job $g"; wait_jobs $g
log "gate: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
[ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL"; exit 1; }; echo "{\"job\": $g}" > $RUNS/$r/gate-passed.json
EJ=""
for ps in ra-a ra-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  e=$(MODELS=base,r2-sol-r1-all-m0.3,$r PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sbatch --parsable -A vlm $X --gpus=h200:3 --cpus-per-task=24 --mem=192G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $e"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B job $e"; done; done
wait_jobs $EJ
res=$(sbatch --parsable -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=ra-result \
  --output=/gpfs/scrubbed/piotrt/owrl/logs/ra-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; PASSES=ra-a,ra-b \
  MODELS=SFT=base,R1=r2-sol-r1-all-m0.3,R1_action=r2-sol-r1-all-m0.3-action PAIRS=R1-SFT,R1_action-SFT,R1_action-R1 \
  python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/r1-action.txt"); wait_jobs $res
cat tillicum/results/iterative-rounds/r1-action.txt; log "R1_ACTION_DONE"
