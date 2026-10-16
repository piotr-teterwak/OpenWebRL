#!/usr/bin/env bash
# Requires explicit approval: 1 H200, 2 CPUs, 120 GiB, 1.5 hours (~$1.35). No API calls, no browsers.
# Scores the saved selectors on the new branching states (pilot32, fresh64, fresh64 successor) with the same
# scripts and checkpoints that produced the 124-state picks, after a reproduction control on the 124 states:
# re-extracted features, Piotr SelectionARM and the Kev-style best checkpoint must reproduce the saved picks.
# Selectors: Piotr SelectionARM; frozen heads (pointwise, listwise, sweep L27, L18-36 action); top-8 adapter
# (adapter-353247 best); Kev-style (kev-353512 best, step 150). Branch outcomes are never used for selection.
#SBATCH --job-name=openwebrl-selection-score-new
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-score-new-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
SRC=$R/source-score-new   # frozen copy made at submission; the job never reads the live checkout
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
O=$R/score-new-$SLURM_JOB_ID
mkdir -p $O
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv

score() {   # score <name> <panels dir> [<luna reference>]
    local name=$1 dir=$2 ref=${3:-} out=$O/$1 refargs=()
    [[ -n $ref ]] && refargs=(--reference-picks $ref)
    mkdir -p $out
    $PY scripts/extract_selection_head_features.py --panels $dir/panels.jsonl --states $dir/states.jsonl \
        --output $out/features --chunk-states 200
    $PY scripts/score_selection_arm_offline.py --panels $dir/panels.jsonl --output $out/piotr_arm.jsonl \
        > $out/piotr.log 2>&1
    for head in heads/pointwise heads/listwise sweep/L27 sweep/L18-36_action; do
        $PY scripts/train_selection_head.py --panels $R/panels/panels.jsonl --features $R/features-piotr \
            --load-head $R/$head/head.pt --branch-panels $dir/panels.jsonl --branch-features $out/features \
            --external-picks $out/piotr_arm.jsonl "${refargs[@]}" --output $out/$head >> $out/head.log 2>&1
    done
    common=(--panels $R/panels/panels.jsonl --branch-panels $dir/panels.jsonl --branch-states $dir/states.jsonl
            --external-picks $out/piotr_arm.jsonl "${refargs[@]}" --init-head $R/heads/pointwise/head.pt
            --eval-only --eval-checkpoint best --final-val-states 0)
    $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 36 --kev --resume $R/kev-353512 \
        --output $out/kev-353512 > $out/kev.log 2>&1
    $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 8 --resume $R/adapter-353247 \
        --output $out/adapter-353247 > $out/adapter.log 2>&1
    echo "$name scored $(date -Is)"
}

# 1. Reproduction control on the original 124 states; stop before new states if picks differ.
score control124 $R/branch124 $R/branch124/luna_before.jsonl
$PY - <<EOF
import json
R, O = "$R", "$O/control124"
saved = {"piotr": {r["state_id"]: r["selected_index"] for r in map(json.loads, open(f"{R}/piotr_arm.jsonl"))},
         **{k: json.load(open(f"{R}/{p}/metrics.json"))["branch_head_picks"] for k, p in [
             ("pointwise", "heads/pointwise"), ("listwise", "heads/listwise"), ("L27", "sweep/L27"),
             ("L18-36_action", "sweep/L18-36_action"), ("kev", "kev-353512"), ("adapter", "adapter-353247")]}}
new = {"piotr": {r["state_id"]: r["selected_index"] for r in map(json.loads, open(f"{O}/piotr_arm.jsonl"))},
       **{k: json.load(open(f"{O}/{p}/metrics.json"))["branch_head_picks"] for k, p in [
           ("pointwise", "heads/pointwise"), ("listwise", "heads/listwise"), ("L27", "sweep/L27"),
           ("L18-36_action", "sweep/L18-36_action"), ("kev", "kev-353512"), ("adapter", "adapter-353247")]}}
report = {k: {"same": sum(new[k].get(s) == v for s, v in saved[k].items()), "states": len(saved[k])} for k in saved}
json.dump(report, open(f"{O}/reproduction.json", "w"), indent=1)
print("reproduction", json.dumps(report))
# GPU kernels can flip near-ties; allow at most 2 of 124 changed picks per selector.
assert all(r["states"] - r["same"] <= 2 for r in report.values()), "control picks differ from the saved picks"
EOF

# 2. New branching states.
for lineage in pilot32 fresh64 fresh64succ; do
    score $lineage $R/branch-new/$lineage
done
echo "score-new complete $(date -Is)" | tee $O/complete.txt
