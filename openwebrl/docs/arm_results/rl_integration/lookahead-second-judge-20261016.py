"""Does a second outcome judge (GPT-6 Luna, medium, same logged input) change the findings? 100-state subset.

Agreement on the 1,312 re-judged held-out continuations; then every screenshot-only no-history condition (pre, post,
K = 1, K = 3) plus uniform / actor / oracle (picks on draws 0-2, o4-mini) scored on held-out draws 3-7 under each judge.
Continuations with no answer (score 0 without a judge) and invalid ones are unchanged. Valid draws only."""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from selection_outcome_experiments import cluster_boot, read  # noqa: E402

J = Path(__file__).resolve().parent; B = J.parent; G = B / "lookahead-repr-20261015"
plan = read(J / "plan.json")
second = {}
sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL")
from openwebrl.eval.reward_online_mind2web import _parse_verdict  # noqa: E402  Online-Mind2Web "Status:" parser
for f in (J / "verdicts").glob("*.json"):
    v = read(f)                                       # re-parse both saved texts with the judge's own parser
    second[(v["state"], v["candidate"], v["repeat"])] = (_parse_verdict(v["original_text"]), _parse_verdict(v["second_text"]) if v["second_text"] else None)
pairs = [(a, b) for a, b in second.values() if a is not None and b is not None]
a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
out = dict(rejudged=len(second), parsed_both=len(pairs), parse_failures=sum(x[1] is None for x in second.values()),
           agreement_pct=round(100 * float((a == b).mean()), 1), cohen_kappa=round(float(cohen_kappa_score(a, b)), 3),
           o4mini_success_pct=round(100 * a.mean(), 1), luna_success_pct=round(100 * b.mean(), 1),
           confusion={"both success": int(((a == 1) & (b == 1)).sum()), "o4-mini only": int(((a == 1) & (b == 0)).sum()),
                      "Luna only": int(((a == 0) & (b == 1)).sum()), "both failure": int(((a == 0) & (b == 0)).sum())})
# outcomes per state under each judge
slot = defaultdict(dict)
for e in plan["states"]:
    db = sqlite3.connect(f"file:{e['root']}/state.sqlite3?mode=ro", uri=True)
    for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
        x = json.loads(raw)
        if x.get("mode") == "branch" and x.get("released") and x["state"] == e["state"]:
            slot[x["state"]].setdefault((x["candidate"], x["repeat"]), (bool(x["valid"]), bool(x["valid"]) and (x.get("reward") or 0) >= 1))
task = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])["task_id"] for e in plan["states"]}
picks = {}
for name, root in {"Pre-execution": "s-pre", "Post-execution": "s-post", "K=1": "s-k1", "K=3": "s-k3"}.items():
    picks[name] = defaultdict(list)
    for f in (G / root / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[name][j["state"]].append(j["candidate"])


def outcome(s, i, k, judge):
    valid, success = slot[s][i, k]
    if judge == "luna" and (s, i, k) in second and second[(s, i, k)][1] is not None:
        success = bool(second[(s, i, k)][1])
    return valid, success


res = {}
for judge in ("o4-mini", "luna"):
    per = {}
    for s in sorted(slot):
        def rate(i, draws):
            cells = [outcome(s, i, k, judge) for k in draws if (i, k) in slot[s]]
            cells = [c[1] for c in cells if c[0]]
            return float(np.mean(cells)) if cells else None
        hold = [rate(i, range(3, 8)) for i in range(5)]; fit = [rate(i, range(3)) for i in range(5)] if judge else None
        fit = [None if not any(slot[s].get((i, k), (0, 0))[0] for k in range(3)) else
               float(np.mean([slot[s][i, k][1] for k in range(3) if (i, k) in slot[s] and slot[s][i, k][0]])) for i in range(5)]
        if None in hold or None in fit or not all(s in picks[n] for n in picks):
            continue
        hold, fit = np.array(hold), np.array(fit)
        per[s] = {n: float(np.mean([hold[c] for c in picks[n][s]])) for n in picks}
        per[s] |= {"Uniform": float(hold.mean()), "Actor": float(hold[0]), "Oracle (draws 0-2)": float(hold[fit == fit.max()].mean())}
    res[judge] = per
common = sorted(set(res["o4-mini"]) & set(res["luna"])); g = [task[s] for s in common]
out["scored_states"], out["scored_tasks"] = len(common), len(set(g))
names = list(res["o4-mini"][common[0]])
out["success_pct"] = {n: {j: round(100 * np.mean([res[j][s][n] for s in common]), 2) for j in res} for n in names}
out["paired"] = {f"{x} - {y}": {j: cluster_boot([res[j][s][x] - res[j][s][y] for s in common], g) for j in res}
                 for x, y in (("K=3", "Post-execution"), ("K=3", "Pre-execution"), ("K=1", "Post-execution"), ("Post-execution", "Pre-execution"),
                              ("K=3", "Uniform"), ("K=3", "Oracle (draws 0-2)"))}
(J / "judge-comparison.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
