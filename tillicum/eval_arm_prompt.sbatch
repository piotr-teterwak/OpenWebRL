#!/usr/bin/env bash
#SBATCH --job-name=arm-prompt-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=05:00:00
#SBATCH --exclude=g013,g021
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/arm-prompt-eval-%j.out
#
# Does the SelectionARM's prompt format matter? Online-Mind2Web best-of-5 with the base actor and
# Piotr's released SelectionARM, served (scripts/serve_arm.py) with
#   canonical: selection_prompt.py v2 (what every tillicum ARM run has used), or
#   frontier:  action-reward-models' frontier_arbiter.py (the ARM's training prompt),
# plus the base actor alone, all in one session. One GPU per arm; the ARM shares its arm's GPU
# with the actor server, as in C2 collection. Same decoding (greedy, JSON schema) for both prompts.
#   PASS=armprompt1 sbatch tillicum/eval_arm_prompt.sbatch     (SMOKE=1: tasks 0-1)
# One arm per 1-GPU job (schedules much sooner):
#   ARMS=frontier PASS=armprompt1 sbatch --gpus=h200:1 --cpus-per-task=8 --mem=120G tillicum/eval_arm_prompt.sbatch
set -euo pipefail
PASS=${PASS:?set PASS}
ARMS_SEL=${ARMS:-canonical frontier base}
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

# arm label | prompt format ("" = no ARM)
ALL=("arm-canonical|canonical" "arm-frontier|frontier" "base|")
ARMS=()
for spec in "${ALL[@]}"; do f=${spec#*|}; [[ " ${ARMS_SEL:-canonical frontier base} " == *" ${f:-base} "* ]] && ARMS+=("$spec"); done
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
echo "[arm-prompt] port block $PORT0"
i=0
for spec in "${ARMS[@]}"; do
  IFS='|' read label fmt <<<"$spec"; mkdir -p $OUT/$label
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $SERVER_PY -m sglang.launch_server --model-path $BASE --host 127.0.0.1 \
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
i=0; for spec in "${ARMS[@]}"; do [ -n "${spec#*|}" ] && { wait_ok http://127.0.0.1:$((PORT0 + 11 + 300*i))/health; curl -s http://127.0.0.1:$((PORT0 + 11 + 300*i))/health; echo; }; i=$((i+1)); done

END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); REMAIN=$(( END - $(date +%s) - 300 ))
i=0; EPIDS=()
for spec in "${ARMS[@]}"; do
  IFS='|' read label fmt <<<"$spec"; b=$((PORT0 + 300*i))
  mode=selection; [ -z "$fmt" ] && mode=baseline
  timeout --signal=TERM --kill-after=120 $REMAIN $ARM_PY -m openwebrl.arm_eval --mode $mode --actor $BASE \
    --actor-port $((b + 10)) --selector-endpoint http://127.0.0.1:$((b + 11)) --output $OUT/$label/online-mind2web \
    --parallel 12 --task-file openwebrl/data/eval/online-mind2web.jsonl --seed 42 --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file .env \
    --browser-port-start $((b + 100)) --browser-port-end $((b + 299)) "${TASKS[@]}" > $OUT/$label/eval.log 2>&1 &
  EPIDS+=($!); i=$((i+1))
done
rc=0; for p in "${EPIDS[@]}"; do wait $p || rc=$?; done
for spec in "${ARMS[@]}"; do l=${spec%%|*}; s=$OUT/$l/online-mind2web/summary.json
  [ -f $s ] && echo "[arm-prompt] $l: $(tr -d '\n ' < $s)" || echo "[arm-prompt] $l: incomplete"; done
exit $rc
