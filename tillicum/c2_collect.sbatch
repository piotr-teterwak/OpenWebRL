#!/usr/bin/env bash
#SBATCH --job-name=owrl-c2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
# Zixian's 2-GPU, 32-browser collection (283899) averaged 4.2 busy cores on 8.
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
# Her full 2,091-task pool took ~29 GPU-h across several allocations. Collection
# resumes by resubmitting with the same C2_ROOT (finished tasks are skipped).
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/piotrt/owrl/logs/c2-%j.out
#
# tillicum port of klone/c2_collect.sbatch: roll out ACTOR on the C2 task pool
# with best-of-5 ARM selection, judge outcomes with o4-mini, and export the
# executed (ARM-picked) turns of successful trajectories as training.jsonl.
# Each GPU co-hosts one actor sglang server and one ARM selector, as in her
# c2-full-282782 run; arm_c2.collect spreads browsers across the replicas.
#
# Round 2: ACTOR is a merged round-1 model, e.g. the 1A retrain merged by the eval:
#   ACTOR=/gpfs/scrubbed/piotrt/owrl/eval/merged/1a NAME=round2-1a sbatch tillicum/c2_collect.sbatch
# Defaults reuse her deduplicated 2,091-task pool so rounds are comparable.
#
# No-ARM baseline: TEACHER=baseline executes the actor's single sample (no selector is
# launched); SEED changes the sampling seeds, so repeated attempts use distinct C2_ROOTs:
#   ACTOR=$BASE TEACHER=baseline SEED=1 NAME=baseline-a1 sbatch --gpus=h200:1 tillicum/c2_collect.sbatch
set -euo pipefail
ACTOR="${ACTOR:?set ACTOR to a model directory (base or merged round-N actor)}"
NAME="${NAME:?set NAME, e.g. round2-1a}"
REPO=/gpfs/home/piotrt/OpenWebRL
ZRT=/gpfs/scrubbed/zixianma/openwebrl-runtime
ARM_PY=$ZRT/arm-reproduction/venv/bin/python
SERVER_PY=$ZRT/venv/bin/python
ZC2=$ZRT/arm-reproduction/runs/c2-full-282782-20260908T075414Z
TASKS="${TASKS:-$ZC2/tasks.jsonl}"
# The selection ARM she collected C2 with (its /health must match on resume).
ARM_MODEL="${ARM_MODEL:-/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89}"
BASE=/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT
C2_ROOT="${C2_ROOT:-/gpfs/scrubbed/piotrt/owrl/c2-collect/$NAME}"
PARALLEL="${PARALLEL:-32}"
TEACHER="${TEACHER:-selection}"
# o4-mini (used for Zixian's C2 outcomes) was retired in 2026-10; outcome filtering now uses JUDGE.
SEED="${SEED:-42}"
case "$TEACHER" in selection|baseline) ;; *) echo "bad TEACHER=$TEACHER"; exit 1;; esac

export OPENWEBRL_CACHE_ROOT=/gpfs/scrubbed/piotrt/owrl/cache
source $REPO/scripts/h200_env.sh
export TORCHINDUCTOR_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/torchinductor TRITON_CACHE_DIR=$OPENWEBRL_CACHE_ROOT/triton
export OMP_NUM_THREADS=4
# Loopback must bypass the inherited Squid proxy (see eval_concurrent.sbatch).
export no_proxy="127.0.0.1,${no_proxy:-localhost}" NO_PROXY="127.0.0.1,${no_proxy:-localhost}"
# Browser env servers reserve ports via lock files in a node-shared /tmp directory by
# default; files another user left there are unwritable to us, every port then looks
# taken, and all tasks fail in seconds (376803 on g014). Use a job-private lock dir;
# the code still probes each port for actual use, so jobs sharing a node cannot clash.
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=$OPENWEBRL_JOB_TMP/port-locks
set -a; source $REPO/.env; set +a
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL OPENAI_API_KEY did not resolve"; exit 1; }
cd $REPO
mkdir -p $C2_ROOT

mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
N=${#GPUS[@]}
[ $((PARALLEL % N)) -eq 0 ] || { echo "FATAL PARALLEL=$PARALLEL must divide across $N GPUs"; exit 1; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || kill $p 2>/dev/null || true; done; rm -rf "$OPENWEBRL_JOB_TMP"; }
trap cleanup EXIT

# arm_c2 wants actor/selector ports below its browser range (19200-19399); stay
# clear of the 19100-19103 her runs use.
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $SERVER_PY -m sglang.launch_server --model-path $ACTOR \
    --host 127.0.0.1 --port $((18100 + 10*i)) --dtype bfloat16 --tp 1 --mem-fraction-static 0.4 \
    --context-length 32768 --max-running-requests 48 --chunked-prefill-size 4096 --cuda-graph-max-bs 48 \
    > $C2_ROOT/actor-replica-$i.log 2>&1 &
  PIDS+=($!)
  [ "$TEACHER" = baseline ] && { PIDS+=(${PIDS[-1]}); continue; }  # keep 2 slots per GPU for wait_ok
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $ARM_PY scripts/serve_arm.py --mode selection --model $ARM_MODEL \
    --base $BASE --port $((18101 + 10*i)) > $C2_ROOT/teacher-replica-$i.log 2>&1 &
  PIDS+=($!)
done
wait_ok() { local url=$1 pid=$2 t0=$(date +%s)
  until curl -sf $url >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "FATAL server for $url died"; tail -20 $C2_ROOT/*replica*.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL $url not healthy after 15 min"; exit 1; }
    sleep 5
  done; }
for i in $(seq 0 $((N-1))); do
  wait_ok http://127.0.0.1:$((18100 + 10*i))/health_generate ${PIDS[$((2*i))]}
  [ "$TEACHER" = baseline ] || wait_ok http://127.0.0.1:$((18101 + 10*i))/health ${PIDS[$((2*i+1))]}
done
echo "[c2] $N replicas healthy; teacher=$TEACHER seed=$SEED actor=$ACTOR"

# The config is frozen on first submission; arm_c2.build reads it from C2_ROOT.
# Only the allocation-bound fields (stop time, ports) are refreshed on resume.
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)   # %e is local time
STOP=$(( END - 900 ))
[ $(( STOP - $(date +%s) )) -gt 1800 ] || { echo "FATAL stop time is <30 min away; check the deadline math"; exit 1; }
SELECTOR_HEALTH=null
[ "$TEACHER" = baseline ] || SELECTOR_HEALTH=$(curl -sf http://127.0.0.1:18101/health)
$ARM_PY - <<PYEOF
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
root = Path('$C2_ROOT'); path = root / 'frozen-config.json'
tasks = Path('$TASKS')
fresh = dict(output=str(root), task_file=str(tasks), task_file_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
             task_count=len(tasks.read_text().splitlines()), parallel=$PARALLEL, seed=$SEED, teacher='$TEACHER', judge='${JUDGE:-gpt-5.6-terra}',
             teacher_manifest=dict(actor=str(Path('$ACTOR').resolve()), selector_health=json.loads('''$SELECTOR_HEALTH''')),
             training=dict(max_tokens=32768), round_source='$NAME')
if path.exists():
    prior = json.loads(path.read_text())
    for key in fresh:
        if prior[key] != fresh[key]: raise SystemExit(f'FATAL resume differs from frozen config in {key!r}; use a new C2_ROOT')
    config = prior
else:
    config = fresh
config.update(stop_utc=datetime.fromtimestamp($STOP, timezone.utc).isoformat(), allocation='$SLURM_JOB_ID',
              collection_replicas=[dict(actor_port=18100 + 10*i, selector_port=18101 + 10*i) for i in range($N)])
path.write_text(json.dumps(config, indent=1) + '\n')
print(f"[c2] config {path}: {config['task_count']} tasks, parallel {config['parallel']}")
PYEOF

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 300 > $C2_ROOT/gpu-util-$SLURM_JOB_ID.csv &
PIDS+=($!)
echo "[c2] collect (resumable: resubmit with the same NAME/C2_ROOT)"
$ARM_PY -m openwebrl.arm_c2 collect --config $C2_ROOT/frozen-config.json >> $C2_ROOT/collection.log 2>&1 || {
  echo "[c2] collect exited non-zero; see $C2_ROOT/collection.log"; exit 1; }
cat $C2_ROOT/collection-summary.json; echo
DONE_N=$(ls $C2_ROOT/results | wc -l); TOTAL=$($ARM_PY -c "import json;print(json.load(open('$C2_ROOT/frozen-config.json'))['task_count'])")
if [ "$DONE_N" -eq "$TOTAL" ]; then
  $ARM_PY -m openwebrl.arm_c2 build --output $C2_ROOT && echo "[c2] DONE -> $C2_ROOT/training.jsonl"
else
  echo "[c2] paused at $DONE_N/$TOTAL tasks; resubmit with NAME=$NAME to continue"
fi
