#!/usr/bin/env bash
# Submit on-policy-pair runs: train -> gate -> 1-GPU eval, plus one same-session base eval.
#   OBJ=dpo|rpo BETAS="1.0" SEEDS="42 1 2" NICE=0 PASS=onpol-best bash tillicum/submit_onpolicy.sh
set -euo pipefail
cd /gpfs/home/piotrt/OpenWebRL
OBJ=${OBJ:?dpo or rpo}; BETAS=${BETAS:?}; SEEDS=${SEEDS:-42 1 2}; NICE=${NICE:-0}; PASS=${PASS:?}
D=/gpfs/scrubbed/piotrt/owrl/joint-data/onpolicy; X="--exclude=g013,g021 --nice=$NICE"
E1="--gpus=h200:1 --cpus-per-task=8 --mem=120G --time=03:00:00 $X"
SFT=0; [ $OBJ = rpo ] && SFT=1
GATES=""
for b in $BETAS; do for sd in $SEEDS; do
  RUN=onpol-$OBJ-b$b-s$sd
  [ -e /gpfs/scrubbed/piotrt/owrl/joint-retrain/runs/$RUN ] && { echo "skip $RUN: run dir exists"; continue; }
  T=$(OBJECTIVE=dpo LR=5e-5 BETA=$b SEED=$sd SFT_WEIGHT=$SFT OWRL_JOINT_DATA=$D RUN=$RUN sbatch --parsable $X --time=06:00:00 --job-name=op-$OBJ-$b-$sd tillicum/train_joint.sbatch)
  G=$(LABEL=$RUN CKPT=student/update-000163 sbatch --parsable $X --job-name=gop-$OBJ-$b-$sd --dependency=afterok:$T tillicum/gate.sbatch)
  E=$(MODELS=$RUN PASS=$PASS sbatch --parsable $E1 --job-name=eop-$OBJ-$b-$sd --dependency=afterok:$G tillicum/eval_concurrent.sbatch)
  GATES="$GATES:$G"; echo "$RUN train=$T gate=$G eval=$E"
done; done
[ -n "$GATES" ] && echo "base eval=$(MODELS=base PASS=$PASS sbatch --parsable $E1 --job-name=eop-base --dependency=afterany$GATES tillicum/eval_concurrent.sbatch)"
