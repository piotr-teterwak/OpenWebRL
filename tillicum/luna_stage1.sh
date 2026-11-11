#!/usr/bin/env bash
# Stage 1 diagnostics for the iterative Luna RPO experiment (luna_iter_pipeline.sh), RPO as before (beta 0.1, SFT
# weight 1, lr 5e-5) but 1 epoch:
#   r2-luna-r1-1ep-s1      round 1 again, seed 1            (seed variance; 1 epoch vs 2)
#   r2-luna-ctrl-1ep-s1    control again, seed 1            (same)
#   r2-luna-cont-refr1-1ep continued from r2-luna-r1 on the round-2 pairs, reference = r2-luna-r1 (standard iterative
#                          DPO/RPO) instead of SFT
# then gate each and evaluate them with base on OM2W and DeepShop (pass luna-iter-s1, T 1.0).
# The end-of-epoch-1 checkpoints of the 2-epoch runs are evaluated separately (pass luna-iter-e1).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
D=/gpfs/scrubbed/piotrt/owrl/joint-data; RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; M=/gpfs/scrubbed/piotrt/owrl/eval/merged
SFT=/gpfs/scrubbed/piotrt/owrl/zixian-mirror/checkpoints/web/OpenWebRL-4B-SFT; R1=$M/r2-luna-r1
X="--exclude=g001,g008,g011,g013,g015,g021"
log() { echo "$(date '+%F %T') $*"; }

wait_job() {  # wait for one job; urgent after 30 min pending
  local j=$1 st state elapsed q w
  while st=$(squeue -h -j $j -o "%T %M %q" 2>/dev/null) && [ -n "$st" ]; do
    read state elapsed q <<<"$st"
    if [ "$state" = PENDING ] && [ "$q" != urgent ]; then
      w=$(( $(date +%s) - $(date -d "$(squeue -h -j $j -o %V)" +%s) ))
      [ $w -gt 1800 ] && scontrol update JobId=$j QOS=urgent 2>/dev/null && log "job $j pending ${w}s -> urgent"
    fi
    sleep 120
  done
}

train() {  # train <run> <data> <actor> <reference> <seed>: 1-epoch RPO (resume up to 2x), then gate
  local run=$1 data=$2 actor=$3 ref=$4 seed=$5 j live
  for attempt in 1 2 3; do
    [ -f $RUNS/$run/student/complete.json ] && break
    live=$(squeue -h -u $USER -n tr-$run -o %i | head -1)
    if [ -n "$live" ]; then log "train $run: adopting job $live"; wait_job $live; continue; fi
    j=$(OBJECTIVE=dpo LR=5e-5 SFT_WEIGHT=1 BETA=0.1 EPOCHS=1 SEED=$seed OWRL_ACTOR=$actor OWRL_REFERENCE=$ref OWRL_JOINT_DATA=$D/$data RUN=$run \
      sbatch --parsable -A vlm $X --time=06:00:00 --job-name=tr-$run tillicum/train_joint.sbatch 2>/dev/null)
    log "train $run (data $data, actor $(basename $actor), ref $(basename $ref), seed $seed) attempt $attempt job $j"; wait_job $j
  done
  [ -f $RUNS/$run/student/complete.json ] || { log "FATAL train $run incomplete"; return 1; }
  local u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$run/frozen-config.json'))['total_updates'])")
  local g=$(LABEL=$run CKPT=student/update-$u sbatch --parsable -A vlm $X --job-name=gt-$run tillicum/gate.sbatch 2>/dev/null)
  log "gate $run job $g"; wait_job $g
  log "gate $run: $(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1)"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] || { log "GATE FAIL $run"; return 2; }
}

log "stage1 start"
train r2-luna-r1-1ep-s1 luna-A $SFT $SFT 1 & a=$!
train r2-luna-ctrl-1ep-s1 luna-ctrl $SFT $SFT 1 & b=$!
train r2-luna-cont-refr1-1ep luna-Br1 $R1 $R1 42 & c=$!
wait $a; wait $b; wait $c
MODELS=base; for r in r2-luna-r1-1ep-s1 r2-luna-ctrl-1ep-s1 r2-luna-cont-refr1-1ep; do [ -f $M/$r/merge-manifest.json ] && MODELS="$MODELS,$r"; done
n=$(tr ',' '\n' <<<"$MODELS" | wc -l); EJ=""
for B in online-mind2web deepshop; do
  j=$(MODELS=$MODELS PASS=luna-iter-s1 BENCH=$B TEMP=1.0 sbatch --parsable -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=06:00:00 --job-name=ev-s1-$B tillicum/eval_concurrent.sbatch 2>/dev/null)
  EJ="$EJ $j"; log "eval $B ($MODELS) job $j"
done
for j in $EJ; do wait_job $j; done
log "STAGE1_DONE"
