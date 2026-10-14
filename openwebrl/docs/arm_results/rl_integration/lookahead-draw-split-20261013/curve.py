"""Draw-split curve for the consistent-format lookahead labels.

The outcome reference picks on draws 0..n-1; every condition is scored on draws n..7 (n = 2..6). Two outcome
conventions: invalid draws count as failures, or valid-only (per-action rates over valid draws; a state is kept at
split n when every action has a valid draw in both halves). Task-clustered 95% intervals on absolute rates."""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from adapt_selection_head_outcomes import LEDGERS  # noqa: E402
from selection_outcome_experiments import R, cluster_boot, read  # noqa: E402

OUT = Path(__file__).resolve().parent
BASES = [R / "lookahead-v2-20261013", R / "lookahead-fast1k-20261013"]
CONDS = {"before": "Before execution", "immediate": "Immediate result", "k1": "Lookahead K=1", "k2": "Lookahead K=2",
         "k3": "Lookahead K=3"}

slot = defaultdict(dict)                              # state -> (candidate, repeat) -> (valid, success)
plans = {b.name: read(b / "k3" / "plan.json") for b in BASES}
for root in set(map(str, LEDGERS)) | {e["root"] for p in plans.values() for e in p["states"]}:
    f = Path(root) / "state.sqlite3"
    if f.exists():
        db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") == "branch" and r.get("released"):
                v = bool(r["valid"])
                slot[r["state"]].setdefault((r["candidate"], r["repeat"]), (v, v and (r.get("reward") or 0) >= 1))
slot["3cec061d636f899f"].setdefault((4, 0), (False, False))   # as outcomes(): unreleased committed invalid draw

data = {}                                             # collection -> (states, picks, task); conditions run there only
for b in BASES:
    picks = {c: defaultdict(list) for c in CONDS if (b / c / "judgments").is_dir()}
    for c in picks:
        for f in (b / c / "judgments").glob("*.json"):
            for j in json.loads(f.read_text()):
                if j["valid"]:
                    picks[c][j["state"]].append(j["candidate"])
    task = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])["task_id"]
            for e in plans[b.name]["states"]}
    states = sorted(s for s in set.intersection(*(set(v) for v in picks.values()))
                    if all((i, k) in slot[s] for i in range(5) for k in range(8)))
    data[b.name] = (states, picks, task)
# SUBSET=piotr_valid40: Piotr-matched tasks (normalized instruction in the pinned training set, as
# piotr-lineage/audit_training_tasks.py), every one of the 40 draws valid, unique model input (pre-action input hash)
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import unicodedata  # noqa: E402
SUBSET = os.environ.get("SUBSET")
if SUBSET in ("piotr", "piotr_valid40"):     # piotr: Piotr-matched and unique, any validity
    LINEAGE = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
                   "parallel-fresh64-preparation-20261010/piotr-lineage")
    piotr = set(read(LINEAGE / "task-membership.private.json")["piotr_training_task_counts"])
    norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))
    seen, audit = set(), {}
    for b in BASES:
        states, picks, task = data[b.name]
        anchors = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
                   for e in plans[b.name]["states"] if e["state"] in set(states)}
        text = lambda a: a["task"] if isinstance(a["task"], str) else (a["task"].get("instruction") or a["task"].get("task") or json.dumps(a["task"]))
        matched = [s for s in states if norm(text(anchors[s])) in piotr]
        valid = [s for s in matched if SUBSET == "piotr" or all(slot[s][i, k][0] for i in range(5) for k in range(8))]
        keep = []
        for s in valid:
            an = anchors[s]                            # older anchors lack the pre-action input hash
            h = an.get("pre_action_input_sha256") or an.get("prompt_input_sha256") or hashlib.sha256(
                json.dumps([an["task_id"], an.get("before_observation")], sort_keys=True, default=str).encode()).hexdigest()
            if h not in seen:
                seen.add(h); keep.append(s)
        audit[b.name] = dict(labelled=len(states), piotr_matched=len(matched), all_40_valid=len(valid), unique=len(keep),
                             task_groups=len({task[s] for s in keep}))
        data[b.name] = (keep, picks, task)
    print(json.dumps(audit), flush=True)
# earlier states restricted, like the new ones, to states where some candidate's outcome differs (draws 0-7)
st, pk, tk = data["lookahead-v2-20261013"]
data["earlier-informative"] = ([s for s in st if len({slot[s][i, k][1] for i in range(5) for k in range(8)}) > 1], pk, tk)


def rate(s, i, draws, valid_only):
    cells = [slot[s][i, k] for k in draws]
    if valid_only:
        cells = [c for c in cells if c[0]]
        return float(np.mean([c[1] for c in cells])) if cells else None
    return float(np.mean([c[1] for c in cells]))


result = {}
for mode, valid_only in ((("all_valid", False),) if SUBSET == "piotr_valid40" else (("invalid_as_failure", False), ("valid_only", True))):
    result[mode] = {}
    for n in range(2, 7):
        rows = {}                                     # collection -> {state: {name: score}}, task
        for name, (states, picks, task) in data.items():
            per = {}
            for s in states:
                fit = [rate(s, i, range(n), valid_only) for i in range(5)]
                hold = [rate(s, i, range(n, 8), valid_only) for i in range(5)]
                if None in fit or None in hold:
                    continue
                fit, hold = np.array(fit), np.array(hold)
                per[s] = {CONDS[c]: float(np.mean([hold[x] for x in picks[c][s]])) for c in picks}
                per[s]["Actor first sample"] = float(hold[0])
                per[s]["Uniform"] = float(hold.mean())
                per[s]["Best on observed draws"] = float(hold[fit == fit.max()].mean())
            rows[name] = (per, task)
        base_rows = [rows[b.name] for b in BASES]
        shared = set.intersection(*(set(next(iter(p.values()))) for p, _ in base_rows))
        rows["pooled"] = ({s: {k: v[k] for k in shared} for p, _ in base_rows for s, v in p.items()},
                          {s: t for _, tk in base_rows for s, t in tk.items()})
        result[mode][n] = {}
        for name, (per, task) in rows.items():
            st = sorted(per)
            result[mode][n][name] = {"states": len(st), "task_groups": len({task[s] for s in st}),
                                     **{k: cluster_boot([per[s][k] for s in st], [task[s] for s in st])
                                        for k in per[st[0]]}}
            # paired differences on the same states: each condition minus uniform, and K = 3 minus observed outcomes
            result[mode][n][name]["minus_uniform"] = {
                k: cluster_boot([per[s][k] - per[s]["Uniform"] for s in st], [task[s] for s in st]) for k in per[st[0]] if k != "Uniform"}
            result[mode][n][name]["minus_actor"] = {
                k: cluster_boot([per[s][k] - per[s]["Actor first sample"] for s in st], [task[s] for s in st])
                for k in per[st[0]] if k != "Actor first sample"}
            result[mode][n][name]["k3_minus_observed"] = cluster_boot(
                [per[s]["Lookahead K=3"] - per[s]["Best on observed draws"] for s in st], [task[s] for s in st])
        print(mode, n, result[mode][n]["pooled"]["k3_minus_observed"], flush=True)
if SUBSET:
    result["subset"] = dict(name=SUBSET, audit=audit)
(OUT / (f"curve_{SUBSET}.json" if SUBSET else "curve.json")).write_text(json.dumps(result, indent=1))
