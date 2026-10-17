"""Draw-split curve for the screenshot-only, no-history conditions (pre, post, K = 1, K = 3; 2 randomized-order calls
per state) on the main set (all labelled states, completely blank pages removed), valid draws only."""
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

OUT = Path(__file__).resolve().parent
B = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")
G = B / "lookahead-repr-20261015"; NW = B / "lookahead-new-20261016"
LIN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-lineage")
piotr = set(read(LIN / "task-membership.private.json")["piotr_training_task_counts"])
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))
RUNS = {"Before execution": ["s-pre"], "Immediate result": ["s-post"], "Lookahead K=1": ["s-k1"], "Lookahead K=3": ["s-k3", "s-k3b"]}
plans = [read(B / "lookahead-fast1k-20261013" / "k3" / "plan.json"), read(B / "lookahead-v2-20261013" / "k3" / "plan.json"), read(NW / "s-k3b" / "plan.json")]
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
task, keep = {}, set()
for p in plans:
    for e in p["states"]:
        a = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
        task[e["state"]] = a["task_id"]
        o = a["before_observation"]; url = o.get("active_tab_url") or ""; sp = o.get("selection_page") or {}
        if not ((url == "" or url.startswith("about:blank")) and not (sp.get("interactive_elements") or []) and not (sp.get("text") or "").strip()):
            keep.add(e["state"])                       # main set: completely blank pages removed
picks = {}
for name, root in RUNS.items():
    picks[name] = defaultdict(list)
    for f in [f for r in root for d in (G / r, NW / r) for f in (d / "judgments").glob("*.json")]:
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[name][j["state"]].append(j["candidate"])
# prior-work judge formulations (formulation-faithful, GPT-6 Luna): their pick sets, ties split
for name, m in (("WebArbiter formulation", "arbiter"), ("WebShepherd formulation", "shepherd"), ("PRO-CUA formulation", "procua")):
    picks[name] = {}
    for root in (B / "prm-formulations-20261016", NW / "prm-formulations"):
        for f in (root / "results" / m).glob("*.json"):
            r = read(f); picks[name][r["state"]] = r["picks"]
states = sorted(keep & set.intersection(*(set(v) for v in picks.values())))


def rate(s, i, draws):
    cells = [slot[s][i, k] for k in draws if (i, k) in slot[s] and slot[s][i, k][0]]
    return float(np.mean([c[1] for c in cells])) if cells else None


result = {"valid_only": {}}
for n in range(2, 7):
    per = {}
    for s in states:
        fit = [rate(s, i, range(n)) for i in range(5)]; hold = [rate(s, i, range(n, 8)) for i in range(5)]
        if None in fit or None in hold:
            continue
        fit, hold = np.array(fit), np.array(hold)
        per[s] = {k: float(np.mean([hold[c] for c in picks[k][s]])) for k in picks}
        per[s] |= {"Uniform": float(hold.mean()), "Actor first sample": float(hold[0]), "Best on observed draws": float(hold[fit == fit.max()].mean())}
    st = sorted(per); g = [task[s] for s in st]
    pooled = {"states": len(st), "task_groups": len(set(g))}
    pooled |= {k: cluster_boot([per[s][k] for s in st], g) for k in per[st[0]]}
    pooled["minus_actor"] = {k: cluster_boot([per[s][k] - per[s]["Actor first sample"] for s in st], g) for k in per[st[0]] if k != "Actor first sample"}
    result["valid_only"][n] = {"pooled": pooled}
    print(n, len(st), {k: round(v[0], 1) for k, v in pooled.items() if isinstance(v, list)}, flush=True)
(OUT / "curve_piotr_main.json").write_text(json.dumps(result, indent=1))
