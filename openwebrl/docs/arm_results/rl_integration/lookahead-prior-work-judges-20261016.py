"""WebShepherd / WebArbiter / PRO-CUA (formulation-faithful, GPT-6 Luna) vs our lookahead labels, on Piotr-task states,
valid held-out draws 3-7, under the o4-mini judge and the GPT-6 Luna re-judge. Ties count as an even split."""
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL"); sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from adapt_selection_head_outcomes import LEDGERS  # noqa: E402
from selection_outcome_experiments import cluster_boot, read  # noqa: E402
from openwebrl.eval.reward_online_mind2web import _parse_verdict  # noqa: E402

R = Path(__file__).resolve().parent; B = R.parent; G = B / "lookahead-repr-20261015"; JL = B / "rejudge-luna-all-20261016"
LIN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-lineage")
piotr = set(read(LIN / "task-membership.private.json")["piotr_training_task_counts"])
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))
plan = read(R / "plan.json")
picks = {}
for m in plan["methods"]:
    picks[m] = {read(f)["state"]: read(f)["picks"] for f in (R / "results" / m).glob("*.json")}
for name, root in {"pre-execution (ours)": "s-pre", "post-execution (ours)": "s-post", "lookahead K=3 (ours)": "s-k3"}.items():
    picks[name] = defaultdict(list)
    for f in (G / root / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[name][j["state"]].append(j["candidate"])
luna = {}
for f in (JL / "verdicts").glob("*.json"):
    v = read(f)
    luna[(v["state"], v["candidate"], v["repeat"])] = _parse_verdict(v["second_text"]) if v["second_text"] else None
slot, task, keep = defaultdict(dict), {}, set()
for root in {str(p) for p in LEDGERS} | {e["root"] for e in plan["states"]}:
    f = Path(root) / "state.sqlite3"
    if f.exists():
        db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            x = json.loads(raw)
            if x.get("mode") == "branch" and x.get("released"):
                v = bool(x["valid"])
                slot[x["state"]].setdefault((x["candidate"], x["repeat"]), (v, v and (x.get("reward") or 0) >= 1))
for e in plan["states"]:
    a = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
    task[e["state"]] = a["task_id"]
    if norm(a["task"] if isinstance(a["task"], str) else json.dumps(a["task"])) in piotr:
        keep.add(e["state"])
states = sorted(keep & set.intersection(*(set(v) for v in picks.values())))
out = {"states_with_all_methods": len(states), "pick_set_size": {m: round(float(np.mean([len(set(picks[m][s])) for s in states])), 2) for m in plan["methods"]}}
for judge in ("o4-mini", "luna"):
    per = {}
    for s in states:
        hold = []
        for i in range(5):
            cells = []
            for k in range(3, 8):
                if (i, k) not in slot[s] or not slot[s][i, k][0]:
                    continue
                succ = slot[s][i, k][1]
                if judge == "luna" and luna.get((s, i, k)) is not None:
                    succ = bool(luna[(s, i, k)])
                cells.append(succ)
            hold.append(float(np.mean(cells)) if cells else None)
        if None in hold:
            continue
        hold = np.array(hold)
        per[s] = {m: float(np.mean([hold[c] for c in picks[m][s]])) for m in picks} | {"uniform": float(hold.mean()), "actor": float(hold[0])}
    st = sorted(per); g = [task[s] for s in st]
    out[judge] = dict(states=len(st), tasks=len(set(g)), success={m: round(100 * np.mean([per[s][m] for s in st]), 2) for m in per[st[0]]},
                      minus_k3={m: cluster_boot([per[s][m] - per[s]["lookahead K=3 (ours)"] for s in st], g) for m in per[st[0]] if m != "lookahead K=3 (ours)"},
                      minus_pre={m: cluster_boot([per[s][m] - per[s]["pre-execution (ours)"] for s in st], g) for m in plan["methods"]})
(R / "method-comparison.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
