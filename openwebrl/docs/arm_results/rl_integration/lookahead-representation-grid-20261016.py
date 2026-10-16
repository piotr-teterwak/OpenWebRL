"""Representation x depth grid without history (S+T, T = no images, S = no page text; pre / post / K=1 / K=3), all on
the same 736 states with 2 randomized-order calls per state. Held-out draws 3-7. Sets: Piotr-task states with valid
draws only (reporting set) and all states with invalid = failure. Paired, task-clustered comparisons; history effect
from the randomized with-history runs (lookahead-random-20261015)."""
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from adapt_selection_head_outcomes import LEDGERS  # noqa: E402
from selection_outcome_experiments import cluster_boot, read  # noqa: E402

W = Path(__file__).resolve().parent; B = W.parent; R = B / "lookahead-random-20261015"
F, E = B / "lookahead-fast1k-20261013", B / "lookahead-v2-20261013"
LIN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-lineage")
piotr = set(read(LIN / "task-membership.private.json")["piotr_training_task_counts"])
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))
REPS, DEPTHS = {"st": "S+T", "t": "T", "s": "S"}, {"pre": "Pre-execution", "post": "Post-execution", "k1": "K=1", "k3": "K=3"}
RUNS = {f"{r}-{d}": W / f"{r}-{d}" for r in REPS for d in DEPTHS}
RUNS |= {"hist-pre": R / "before", "hist-post": R / "immediate", "hist-k3": R / "k3"}
plans = [read(F / "k3" / "plan.json"), read(E / "k3" / "plan.json")]
slot = defaultdict(dict)
for root in {str(p) for p in LEDGERS} | {e["root"] for p in plans for e in p["states"]}:
    f = Path(root) / "state.sqlite3"
    if f.exists():
        db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            x = json.loads(raw)
            if x.get("mode") == "branch" and x.get("released"):
                v = bool(x["valid"])
                slot[x["state"]].setdefault((x["candidate"], x["repeat"]), (v, v and (x.get("reward") or 0) >= 1))
slot["3cec061d636f899f"].setdefault((4, 0), (False, False))
meta = {}
for p in plans:
    for e in p["states"]:
        a = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
        meta[e["state"]] = dict(task=a["task_id"], piotr=norm(a["task"] if isinstance(a["task"], str) else json.dumps(a["task"])) in piotr)
picks = {}
for k, root in RUNS.items():
    picks[k] = defaultdict(list)
    for f in (root / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[k][j["state"]].append(j["candidate"])
cost = {k: round(read(root / "ledger.json")["charged_usd"], 2) for k, root in RUNS.items()}
common = set.intersection(*(set(v) for v in picks.values()))
out = {"cost_usd": cost, "states_with_all_runs": len(common)}
for setname, valid_only, keep in (("Piotr tasks, valid draws only", True, lambda s: meta[s]["piotr"]),
                                  ("all states, invalid = failure", False, lambda s: True)):
    per = {}
    for s in sorted(common):
        if not keep(s):
            continue
        h = []
        for i in range(5):
            cells = [slot[s][i, k] for k in range(3, 8) if (i, k) in slot[s]]
            cells = [x for x in cells if x[0]] if valid_only else cells
            h.append(np.mean([x[1] for x in cells]) if cells else None)
        if None in h:
            continue
        h = np.array(h)
        per[s] = {k: float(np.mean([h[c] for c in picks[k][s]])) for k in RUNS} | {"uniform": float(h.mean()), "actor": float(h[0])}
    st = sorted(per); g = [meta[s]["task"] for s in st]
    diff = lambda a, b: cluster_boot([per[s][a] - per[s][b] for s in st], g)
    res = {"states": len(st), "tasks": len(set(g)),
           "success": {k: round(100 * np.mean([per[s][k] for s in st]), 2) for k in list(RUNS) + ["uniform", "actor"]},
           "rep_vs_st": {f"{REPS[r]} - S+T, {DEPTHS[d]}": diff(f"{r}-{d}", f"st-{d}") for r in ("t", "s") for d in DEPTHS},
           "depth_vs_post": {f"{REPS[r]}: {DEPTHS[d]} - post": diff(f"{r}-{d}", f"{r}-post") for r in REPS for d in ("pre", "k1", "k3")},
           "vs_actor": {f"{REPS[r]}, {DEPTHS[d]}": diff(f"{r}-{d}", "actor") for r in REPS for d in DEPTHS},
           "history_minus_none": {DEPTHS[d]: diff(f"hist-{d}", f"st-{d}") for d in ("pre", "post", "k3")}}
    out[setname] = res
(W / "grid-results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "cost_usd"}, indent=1)[:300])
