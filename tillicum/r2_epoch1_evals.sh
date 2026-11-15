#!/usr/bin/env bash
# Epoch-1 checkpoints (update 196 of 391) of round 2 with SFT reference and with C2 reference (2026-11-14, Piotr: "will
# you be able to eval after epoch 1?"). Waits until r2_refsft.sh submits its evals (passes rs-a / rs-b: C2, r2, refsft),
# then submits r2-sol2-r2-refsft@196 and r2-sol2-r2@196 into the same passes, same window, T 1.0.
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
E=/gpfs/scrubbed/piotrt/owrl/eval; X="--exclude=g001,g008,g011,g013,g015,g021"; LOG=/gpfs/scrubbed/piotrt/owrl/c2-collect/refsft.log
log() { echo "$(date '+%F %T') [epoch1] $*"; }
until grep -q "eval rs-b deepshop job" $LOG; do grep -qE "FATAL|GATE FAIL|BUDGET GUARD" $LOG && { log "driver stopped before evals; not submitting"; exit 1; }; sleep 120; done
J=""
for ps in rs-a rs-b; do for B in online-mind2web deepshop; do
  t=$([ $B = deepshop ] && echo 02:00:00 || echo 02:30:00)
  j=$(MODELS=r2-sol2-r2-refsft@196,r2-sol2-r2@196 PASS=$ps BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:2 --cpus-per-task=16 \
      --mem=128G --time=$t --job-name=ev1-$ps-$B tillicum/eval_concurrent.sbatch 2>>/gpfs/scrubbed/piotrt/owrl/c2-collect/refsft-sbatch.err)
  [ -n "$j" ] || { log "sbatch refused for $ps $B (see refsft-sbatch.err)"; continue; }
  J="$J $j"; log "eval $ps $B (refsft@196, r2@196) job $j"
done; done
while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$J") 2>/dev/null)" ]; do sleep 120; done
until grep -q REFSFT_DONE $LOG; do sleep 120; done
PASSES=rs-a,rs-b MODELS=C2=r2-sol-r1-all-m0.3,r2=r2-sol2-r2,refsft=r2-sol2-r2-refsft,refsft_e1=r2-sol2-r2-refsft@196,r2_e1=r2-sol2-r2@196 \
  PAIRS=refsft_e1-C2,refsft_e1-refsft,refsft_e1-r2_e1,r2_e1-C2,r2_e1-r2,refsft-C2,refsft-r2,r2-C2 python3 tillicum/round2_result.py \
  | tee tillicum/results/luna-iter/r2-refsft-epoch1.txt
log "EPOCH1_DONE"
