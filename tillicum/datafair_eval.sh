#!/usr/bin/env bash
# Data-fair comparison (2026-11-22, Piotr): do the best two-round recipes beat spending the same data in one round?
# Same-window eval of Round 1, the data-matched single round (r2-sol2-ctrl: half A + half B, SFT candidates, one round
# from SFT), r2c-sftonly and r2c-replay. 2 passes (df-a, df-b), OM2W 12 browsers + DeepShop 6, 1 GPU per model,
# then the paired comparison -> tillicum/results/iterative-rounds/data-fair.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
X="--exclude=g001,g008,g011,g013,g015,g021"; ERR=/gpfs/scrubbed/piotrt/owrl/c2-collect/datafair.sbatch.err
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
sub() { local j; j=$(sbatch --parsable "$@" 2>>$ERR); [ -n "$j" ] || { log "FATAL sbatch refused: $* (see $ERR)" >&2; exit 1; }; echo $j; }
M=r2-sol-r1-all-m0.3,r2-sol2-ctrl,r2c-sftonly,r2c-replay; n=4; EJ=""
for ps in df-a df-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  j=$(MODELS=$M PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sub -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $j"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B ($M) job $j"
done; done
wait_jobs $EJ
r=$(sub -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=df-result \
  --output=/gpfs/scrubbed/piotrt/owrl/logs/df-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; PASSES=df-a,df-b \
  MODELS=R1=r2-sol-r1-all-m0.3,single_round=r2-sol2-ctrl,sft_only=r2c-sftonly,replay=r2c-replay \
  PAIRS=single_round-R1,sft_only-R1,replay-R1,sft_only-single_round,replay-single_round,replay-sft_only \
  python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/data-fair.txt"); wait_jobs $r
cat tillicum/results/iterative-rounds/data-fair.txt; log "DATAFAIR_DONE"
