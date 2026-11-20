#!/usr/bin/env bash
# Round 2 loss ablation, as continuations of Round 1 (2026-11-19, Piotr: no fresh adapter in any loss-ablation or
# mixed-pool run). Every run starts from SFT + Round 1's own LoRA adapter AND Adam state (INIT_FROM = Round 1's
# update-000332), flat LR 2.5e-5 (where Round 1 ended; CONSTANT_LR=1, no re-warmup), DPO reference = Round 1, 2 epochs:
#   r2c-rpo      Round 2's loss (DPO b0.1 + SFT term on successful pairs) on sol2-Br1   (the matched control; job 501958)
#   r2c-dpoonly  DPO only (SFT weight 0) on sol2-Br1                                     (job 501959)
#   r2c-sftonly  SFT only on sol2-Br1's 2,254 successful-episode chosen responses       (job 501960)
#   r2c-replay   Round 2's loss on sol2-Br1 + a 50% replay of Round 1's half-A pairs     (job 502175; data sol2-Br1-replayA)
# (The earlier fresh-adapter versions 501790/501791 and the action-only run 501851 were cancelled.)
# After training: gate each, then 2 eval passes (la-a, la-b) of Round 1 and the four continuations (5 GPUs per job; the
# fresh-adapter Round 2 is not re-evaluated, it has 4 earlier passes) (OM2W 12 browsers, DeepShop 6), then the paired comparison ->
# tillicum/results/iterative-rounds/loss-ablation.txt
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
RUNS=/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs; X="--exclude=g001,g008,g011,g013,g015,g021"; ERR=/gpfs/scrubbed/piotrt/owrl/c2-collect/loss-ablation.sbatch.err
log() { echo "$(date '+%F %T') $*"; }
wait_jobs() { [ $# -gt 0 ] || return 0; while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 180; done; }
sub() { local j; j=$(sbatch --parsable "$@" 2>>$ERR); [ -n "$j" ] || { log "FATAL sbatch refused: $* (see $ERR)" >&2; exit 1; }; echo $j; }
wait_jobs 501958 501959 501960 502175
for r in r2c-rpo r2c-dpoonly r2c-sftonly r2c-replay; do
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
for r in r2c-rpo r2c-dpoonly r2c-sftonly r2c-replay; do
  [ -f $RUNS/$r/gate-passed.json ] && continue
  u=$(python3 -c "import json;print('%06d'%json.load(open('$RUNS/$r/frozen-config.json'))['total_updates'])")
  g=$(LABEL=$r CKPT=student/update-$u sub -A vlm $X --mem=160G --job-name=gt-$r tillicum/gate.sbatch); log "gate $r job $g"; G="$G $r:$g"
done
for rg in $G; do r=${rg%:*}; g=${rg#*:}; wait_jobs $g; line=$(grep '\[gate\]' /gpfs/scrubbed/piotrt/owrl/logs/gate-$g.out | tail -1); log "gate $r: $line"
  [ "$(sacct -j $g -X -n -o State | tr -d ' ')" = COMPLETED ] && echo "{\"job\": $g}" > $RUNS/$r/gate-passed.json; done
M=r2-sol-r1-all-m0.3
for r in r2c-rpo r2c-dpoonly r2c-sftonly r2c-replay; do [ -f $RUNS/$r/gate-passed.json ] && M="$M,$r" || log "NOT EVALUATED: $r failed its gate"; done
n=$(tr ',' '\n' <<<"$M" | wc -l); EJ=""
for ps in la-a la-b; do for B in online-mind2web deepshop; do
  if [ $B = deepshop ]; then P=6; T=03:00:00; else P=12; T=02:30:00; fi
  j=$(MODELS=$M PASS=$ps BENCH=$B TEMP=1.0 PARALLEL=$P sub -A vlm $X --gpus=h200:$n --cpus-per-task=$((8*n)) --mem=$((64*n))G --time=$T --job-name=ev-$ps-$B tillicum/eval_concurrent.sbatch)
  EJ="$EJ $j"; mkdir -p /gpfs/scrubbed/piotrt/owrl/eval/pass$ps; log "eval $ps $B ($M) job $j"
done; done
wait_jobs $EJ
r=$(sub -A vlm -p gpu-h200-mig --qos=debug --gpus=1 --cpus-per-task=1 --mem=8G --time=00:45:00 --job-name=la-result \
      --output=/gpfs/scrubbed/piotrt/owrl/logs/la-result-%j.out --wrap "cd /gpfs/home/piotrt/OpenWebRL; \
      PASSES=la-a,la-b MODELS=R1=r2-sol-r1-all-m0.3,R2c=r2c-rpo,dpo_only=r2c-dpoonly,sft_only=r2c-sftonly,replay=r2c-replay \
      PAIRS=R2c-R1,dpo_only-R1,sft_only-R1,replay-R1,dpo_only-R2c,sft_only-R2c,replay-R2c,dpo_only-sft_only \
      python3 tillicum/round2_result.py > tillicum/results/iterative-rounds/loss-ablation.txt")
log "result job $r"; wait_jobs $r; cat tillicum/results/iterative-rounds/loss-ablation.txt
log "LOSS_ABLATION_DONE"
