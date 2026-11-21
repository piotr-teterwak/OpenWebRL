#!/usr/bin/env bash
# r2c-rpo (Round 2 as a continuation of Round 1: adapter + Adam, flat 2.5e-5, reference Round 1) evaluated on its own,
# ahead of the other continuations (2026-11-20, Piotr): its validation broke down in epoch 2 (winner CE 0.237 -> 0.332
# at update 293, like the fresh-adapter Round 2), so both the epoch-1 checkpoint (update 196) and the final (391) are
# evaluated next to Round 1. Waits for training job 501958, gates the final checkpoint, 2 eval passes (rp-a, rp-b),
# OM2W 12 browsers + DeepShop 6, then the paired comparison -> tillicum/results/iterative-rounds/r2c-rpo.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; X="--exclude=g001,g008,g011,g013,g015,g021"; r=r2c-rpo
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
wait_jobs 501958
[ -f $RUNS/$r/student/complete.json ] || { log "FATAL $r incomplete"; exit 1; }
if [ ! -f $RUNS/$r/gate-passed.json ]; then
  g=$(LABEL=$r CKPT=student/update-000391 sbatch --parsable -A vlm $X --mem=160G --job-name=gt-$r tillicum/gate.sbatch); log "gate job $g"; wait_jobs $g
  log "gate: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && echo "{\"job\": $g}" > $RUNS/$r/gate-passed.json || log "GATE FAIL (evaluating anyway, flagged)"
fi
M=r2-sol-r1-all-m0.3,r2c-rpo@196,r2c-rpo; EJ=""
for ps in rp-a rp-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  e=$(MODELS=$M PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sbatch --parsable -A vlm $X --gpus=h200:3 --cpus-per-task=24 --mem=192G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $e"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B ($M) job $e"; done; done
wait_jobs $EJ
res=$(sbatch --parsable -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=rp-result \
  --output=/gpfs/scrubbed/piotrt/owrl/logs/rp-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; PASSES=rp-a,rp-b \
  MODELS=R1=r2-sol-r1-all-m0.3,R2c_e1=r2c-rpo@196,R2c=r2c-rpo PAIRS=R2c_e1-R1,R2c-R1,R2c_e1-R2c \
  python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/r2c-rpo.txt"); wait_jobs $res
cat tillicum/results/iterative-rounds/r2c-rpo.txt; log "R2C_RPO_DONE"
