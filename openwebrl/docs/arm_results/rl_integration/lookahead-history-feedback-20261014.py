"""History and lookahead-feedback effects (pre-execution, K = 1, K = 3), on the same fast1k states; held-out draws 3-7.
Two outcome conventions (invalid = failure; valid only). Paired differences with task-clustered bootstrap."""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from selection_outcome_experiments import cluster_boot, read  # noqa: E402

V = Path(__file__).resolve().parent
CONDS = {"before": "Pre-execution, 3 past steps", "before-h0": "Pre-execution, no history",
         "k1": "K=1, 3 past steps", "k1-h0": "K=1, no history", "immediate": "Post-execution, 3 past steps",
         "k3": "K=3, 3 past steps", "k3-h0": "K=3, no history", "k3-nofb": "K=3, no tool feedback in lookahead"}
PAIRS = {"history - none: pre-execution": ("before", "before-h0"), "history - none: K=1": ("k1", "k1-h0"),
         "history - none: K=3": ("k3", "k3-h0"), "feedback - none: K=3": ("k3", "k3-nofb"),
         "K=3 no feedback - post-execution": ("k3-nofb", "immediate"), "K=3 - post-execution": ("k3", "immediate"),
         "K=1 - pre-execution (3 past steps)": ("k1", "before"), "K=1 - pre-execution (no history)": ("k1-h0", "before-h0")}
plan = read(V / "k3" / "plan.json")
slot = defaultdict(dict)
for root in {e["root"] for e in plan["states"]}:
    db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
    for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
        r = json.loads(raw)
        if r.get("mode") == "branch" and r.get("released"):
            v = bool(r["valid"])
            slot[r["state"]].setdefault((r["candidate"], r["repeat"]), (v, v and (r.get("reward") or 0) >= 1))
task = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])["task_id"] for e in plan["states"]}
picks = {}
for c in CONDS:
    picks[c] = defaultdict(list)
    for f in (V / c / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[c][j["state"]].append(j["candidate"])
states = sorted(set.intersection(*(set(v) for v in picks.values())))
out = {"states": len(states)}
for mode in ("invalid_as_failure", "valid_only"):
    per = {}
    for s in states:
        hold = []
        for i in range(5):
            cells = [slot[s][i, k] for k in range(3, 8) if (i, k) in slot[s]]
            cells = [c for c in cells if c[0]] if mode == "valid_only" else cells
            hold.append(np.mean([c[1] for c in cells]) if cells else None)
        if None in hold:
            continue
        per[s] = {n: float(np.mean([hold[x] for x in picks[c][s]])) for c, n in CONDS.items()}
    st = sorted(per); g = [task[s] for s in st]
    res = {"states": len(st), "tasks": len(set(g)),
           "success": {n: cluster_boot([per[s][n] for s in st], g) for n in CONDS.values()}}
    res["paired"] = {name: cluster_boot([per[s][CONDS[a]] - per[s][CONDS[b]] for s in st], g) for name, (a, b) in PAIRS.items()}
    out[mode] = res
(V / "history-effect.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
