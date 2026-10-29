#!/usr/bin/env bash
#SBATCH --job-name=arm-compose-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=05:00:00
#SBATCH --exclude=g013,g021
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/arm-compose-eval-%j.out
#
# Do inference-time ARM selection and ARM-pair fine-tuning compose? Online-Mind2Web, one session,
# 2x2: {base, fine-tuned actor} x {n=1, best-of-5 SelectionARM (canonical prompt)}. The ARM is the
# same served model in both selection arms (its own --base is OpenWebRL-4B-SFT, whatever the actor).
# Copy of eval_arm_prompt.sbatch with a per-arm actor. One GPU per arm; the ARM shares the GPU.
# TUNED lists fine-tuned actors as label=path; each gets an n=1 arm and a +arm arm. ARMS picks the
# arms for this job (one GPU each). An arm name may end in @SEED to set arm_eval's sampling seed
# (default 42), e.g. base+arm@1. Example, base plus two seeds = 5 GPUs:
#   M=/gpfs/scrubbed/piotrt/owrl/eval/merged
#   TUNED="s1=$M/onpol-dpo-b0.3-s1 s2=$M/onpol-dpo-b0.3-s2" ARMS="base s1 s1+arm s2 s2+arm" PASS=compose1 \
#     sbatch --gpus=h200:5 --cpus-per-task=40 --mem=600G tillicum/eval_arm_compose.sbatch      (SMOKE=1: tasks 0-1)
set -euo pipefail
PASS=${PASS:?set PASS}
ARMS_SEL=${ARMS:?set ARMS, e.g. "base base+arm s1 s1+arm"}
TUNED=${TUNED:-}
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python
SERVER_PY=$ZRT/venv/bin/python
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
ARM=/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89
OUT=/gpfs/scrubbed/piotrt/owrl/eval/pass$PASS
export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=4
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=$OPENWEBRL_JOB_TMP/port-locks
set -a; source $REPO/.env; set +a
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL OPENAI_API_KEY did not resolve"; exit 1; }
cd $REPO; mkdir -p $OUT
TASKS=(); [ "${SMOKE:-0}" = 1 ] && TASKS=(--task-indices 0,1)
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
PIDS=(); cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || true; done; rm -rf "$OPENWEBRL_JOB_TMP"; }; trap cleanup EXIT

# arm label | actor | ARM prompt format ("" = no ARM, n=1)
ALL=("base|$BASE||42" "base+arm|$BASE|canonical|42")
for t in $TUNED; do l=${t%%=*} m=${t#*=}
  [ -f $m/config.json ] || { echo "FATAL no model at $m"; exit 1; }
  ALL+=("$l|$m||42" "$l+arm|$m|canonical|42"); done
ARMS=()
for want in $ARMS_SEL; do name=${want%@*}; sd=42; [[ $want == *@* ]] && sd=${want#*@}
  for spec in "${ALL[@]}"; do IFS='|' read l m f _ <<<"$spec"
    [ "$l" = "$name" ] && { [ $sd = 42 ] && lab=$l || lab=$l-seed$sd; ARMS+=("$lab|$m|$f|$sd"); }; done; done
[ ${#ARMS[@]} -eq $(wc -w <<<"$ARMS_SEL") ] || { echo "FATAL unknown arm in ARMS=$ARMS_SEL"; exit 1; }
[ ${#GPUS[@]} -eq ${#ARMS[@]} ] || { echo "FATAL ${#ARMS[@]} arms need ${#ARMS[@]} GPUs, got ${#GPUS[@]}"; exit 1; }
# Port block per job: 1-GPU jobs (this script's and eval_concurrent's) can share a node, so take the
# first 1,000-port block above 21000 whose server ports are free here (browser ports are probed by arm_eval).
PORT0=$(python3 - <<PYEOF
import socket, sys
def free(p):
    with socket.socket() as s:
        try: s.bind(('127.0.0.1', p)); return True
        except OSError: return False
start = $SLURM_JOB_ID % 40
for k in list(range(start, 40)) + list(range(start)):
    base = 21000 + 1000 * k
    if all(free(base + 10 + 300 * i) and free(base + 11 + 300 * i) for i in range(${#ARMS[@]})):
        print(base); sys.exit()
sys.exit('no free port block')
PYEOF
)
echo "[arm-compose] port block $PORT0"
i=0
for spec in "${ARMS[@]}"; do
  IFS='|' read label actor fmt sd <<<"$spec"; mkdir -p $OUT/$label
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $SERVER_PY -m sglang.launch_server --model-path $actor --host 127.0.0.1 \
    --port $((PORT0 + 10 + 300*i)) --dtype bfloat16 --tp 1 --mem-fraction-static 0.4 --context-length 32768 \
    --max-running-requests 48 --chunked-prefill-size 4096 --disable-cuda-graph > $OUT/$label/actor-server.log 2>&1 &
  PIDS+=($!)
  if [ -n "$fmt" ]; then
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $ARM_PY scripts/serve_arm.py --mode selection --model $ARM --base $BASE \
      --port $((PORT0 + 11 + 300*i)) --prompt-format $fmt > $OUT/$label/arm-server.log 2>&1 &
    PIDS+=($!)
  fi
  i=$((i+1))
done
wait_ok() { local url=$1 t0=$(date +%s)
  until curl -sf $url >/dev/null; do
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL $url not healthy"; tail -20 $OUT/*/*-server.log; exit 1; }
    sleep 5; done; }
for i in $(seq 0 $((${#ARMS[@]}-1))); do wait_ok http://127.0.0.1:$((PORT0 + 10 + 300*i))/health_generate; done
i=0; for spec in "${ARMS[@]}"; do IFS='|' read _ _ fmt _ <<<"$spec"; [ -n "$fmt" ] && { wait_ok http://127.0.0.1:$((PORT0 + 11 + 300*i))/health; curl -s http://127.0.0.1:$((PORT0 + 11 + 300*i))/health; echo; }; i=$((i+1)); done

END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
i=0; EPIDS=()
for spec in "${ARMS[@]}"; do
  IFS='|' read label actor fmt sd <<<"$spec"; b=$((PORT0 + 300*i))
  mode=selection; [ -z "$fmt" ] && mode=baseline
  timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode $mode --actor $actor \
    --actor-port $((b + 10)) --selector-endpoint http://127.0.0.1:$((b + 11)) --output $OUT/$label/online-mind2web \
    --parallel 12 --task-file openwebrl/data/eval/online-mind2web.jsonl --seed $sd --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file .env \
    --browser-port-start $((b + 100)) --browser-port-end $((b + 299)) "${TASKS[@]}" > $OUT/$label/eval.log 2>&1 &
  EPIDS+=($!); i=$((i+1))
done
rc=0; for p in "${EPIDS[@]}"; do wait $p || rc=$?; done
for spec in "${ARMS[@]}"; do l=${spec%%|*}; s=$OUT/$l/online-mind2web/summary.json
  [ -f $s ] && echo "[arm-compose] $l: $(tr -d '\n ' < $s)" || echo "[arm-compose] $l: incomplete"; done
exit $rc
