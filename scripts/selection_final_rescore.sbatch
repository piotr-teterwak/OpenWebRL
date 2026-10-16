#!/usr/bin/env bash
# Requires explicit approval before submission (planned: 1 H200, 4 CPUs, 120 GiB, 1.5 h). No API calls, no browsers.
# Run once branching collection has finished. For every new collection: rebuild panels from the final
# accepted states, then score them with the saved selectors exactly as selection_score_new_branch.sbatch
# does (Piotr SelectionARM, four frozen heads, Kev-style steps 150 and 600, top-8 adapter). States already
# scored by job 354319 must reproduce their picks. Afterwards the CPU analyses run on every scored state:
#   SELECTION_SOURCES=$O/sources.json scripts/adapt_selection_head_outcomes.py / selection_outcome_experiments.py
#SBATCH --job-name=openwebrl-selection-final
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-selection-final-%j.out
set -euo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
B=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008
SRC=${SRC:-$R/source-final}   # frozen working-tree copy made at submission (as for source-score-new)
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
export PYTHONPATH=$SRC HF_HOME=$R/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
# name:campaign-root pairs; extend when further collections finish
LINEAGES=${LINEAGES:-"pilot32:$B/pilot32-v1 fresh64:$B/expansion64-20261010-v1 fresh64succ:$B/next-fresh-batch-preparation-20261010/staged-campaign-v1"}
O=$R/score-final-$SLURM_JOB_ID
mkdir -p $O
cd $SRC
nvidia-smi --query-gpu=name,memory.total --format=csv

score() {   # score <name> <panels dir>
    local name=$1 dir=$2 out=$O/$1
    mkdir -p $out
    $PY scripts/extract_selection_head_features.py --panels $dir/panels.jsonl --states $dir/states.jsonl \
        --output $out/features --chunk-states 200
    $PY scripts/score_selection_arm_offline.py --panels $dir/panels.jsonl --output $out/piotr_arm.jsonl > $out/piotr.log 2>&1
    for head in heads/pointwise heads/listwise sweep/L27 sweep/L18-36_action; do
        $PY scripts/train_selection_head.py --panels $R/panels/panels.jsonl --features $R/features-piotr \
            --load-head $R/$head/head.pt --branch-panels $dir/panels.jsonl --branch-features $out/features \
            --external-picks $out/piotr_arm.jsonl --output $out/$head >> $out/head.log 2>&1
    done
    common=(--panels $R/panels/panels.jsonl --branch-panels $dir/panels.jsonl --branch-states $dir/states.jsonl
            --external-picks $out/piotr_arm.jsonl --init-head $R/heads/pointwise/head.pt
            --eval-only --eval-checkpoint best --final-val-states 0)
    $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 36 --kev --resume $R/kev-353512 --output $out/kev-353512 > $out/kev.log 2>&1
    $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 36 --kev --resume $R/kev2-353914 --output $out/kev2-353914 > $out/kev2.log 2>&1
    $PY scripts/train_selection_adapter.py "${common[@]}" --top-layers 8 --resume $R/adapter-353247 --output $out/adapter-353247 > $out/adapter.log 2>&1
    echo "$name scored $(date -Is)"
}

entries=()
for pair in $LINEAGES; do
    name=${pair%%:*}; root=${pair#*:}
    $PY scripts/build_selection_head_branch_eval.py --run $root --output $R/branch-final/$name --split $name --expect-trials 0 \
        > $O/build-$name.json
    score $name $R/branch-final/$name
    entries+=("{\"name\": \"$name\", \"panels\": \"$R/branch-final/$name\", \"features\": \"$O/$name/features\", \"picks\": \"$O/$name\"}")
done
( IFS=,; echo "[${entries[*]}]" ) > $O/sources.json

# Reproduction: states scored earlier (job 354319 / 354320) must keep their picks (allow 2 GPU near-ties per selector).
$PY - <<EOF
import json, pathlib
R, O = pathlib.Path("$R"), pathlib.Path("$O")
def read(path):
    if path.suffix == ".jsonl":
        return {r["state_id"]: r["selected_index"] for r in map(json.loads, open(path))}
    return json.loads(path.read_text())["branch_head_picks"]
report = {}
for name in ("pilot32", "fresh64", "fresh64succ"):
    old, new = R / "score-new-354319" / name, O / name
    if not new.exists():
        continue
    pairs = {"piotr": (old / "piotr_arm.jsonl", new / "piotr_arm.jsonl"),
             "kev600": (R / "score-new-kev2-354320" / name / "metrics.json", new / "kev2-353914" / "metrics.json")}
    for k, sub in [("pointwise", "heads/pointwise"), ("listwise", "heads/listwise"), ("L27", "sweep/L27"),
                   ("L18-36_action", "sweep/L18-36_action"), ("kev150", "kev-353512"), ("adapter", "adapter-353247")]:
        pairs[k] = (old / sub / "metrics.json", new / sub / "metrics.json")
    for k, (a, b) in pairs.items():
        before, after = read(a), read(b)
        common = [s for s in before if s in after]
        report[f"{name}/{k}"] = {"states": len(common), "same": sum(before[s] == after[s] for s in common)}
(O / "reproduction.json").write_text(json.dumps(report, indent=1))
print("reproduction", json.dumps(report))
assert all(r["states"] - r["same"] <= 2 for r in report.values()), "previously scored picks changed"
EOF
echo "final rescore complete $(date -Is)" | tee $O/complete.txt
