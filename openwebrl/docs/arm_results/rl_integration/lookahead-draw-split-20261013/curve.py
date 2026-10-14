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
CONDS = {"before": "Before execution", "immediate": "Immediate result", "k1": "Lookahead K=1", "k3": "Lookahead K=3"}

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

data = {}                                             # collection -> (states, picks, task)
for b in BASES:
    picks = {c: defaultdict(list) for c in CONDS}
    for c in CONDS:
        for f in (b / c / "judgments").glob("*.json"):
            for j in json.loads(f.read_text()):
                if j["valid"]:
                    picks[c][j["state"]].append(j["candidate"])
    task = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])["task_id"]
            for e in plans[b.name]["states"]}
    states = sorted(s for s in set.intersection(*(set(v) for v in picks.values()))
                    if all((i, k) in slot[s] for i in range(5) for k in range(8)))
    data[b.name] = (states, picks, task)


def rate(s, i, draws, valid_only):
    cells = [slot[s][i, k] for k in draws]
    if valid_only:
        cells = [c for c in cells if c[0]]
        return float(np.mean([c[1] for c in cells])) if cells else None
    return float(np.mean([c[1] for c in cells]))


result = {}
for mode, valid_only in (("invalid_as_failure", False), ("valid_only", True)):
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
                per[s] = {CONDS[c]: float(np.mean([hold[x] for x in picks[c][s]])) for c in CONDS}
                per[s]["Uniform"] = float(hold.mean())
                per[s]["Best on observed draws"] = float(hold[fit == fit.max()].mean())
            rows[name] = (per, task)
        rows["pooled"] = ({s: v for p, _ in rows.values() for s, v in p.items()},
                          {s: t for _, tk in list(rows.values()) for s, t in tk.items()})
        result[mode][n] = {}
        for name, (per, task) in rows.items():
            st = sorted(per)
            result[mode][n][name] = {"states": len(st), "task_groups": len({task[s] for s in st}),
                                     **{k: cluster_boot([per[s][k] for s in st], [task[s] for s in st])
                                        for k in per[st[0]]}}
        print(mode, n, {k: (round(v[0], 2) if isinstance(v, list) else v) for k, v in result[mode][n]["pooled"].items()}, flush=True)
(OUT / "curve.json").write_text(json.dumps(result, indent=1))
