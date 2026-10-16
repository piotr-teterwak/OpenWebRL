"""Randomized vs fixed candidate order for pre-execution, post-execution, K = 3 and K = 3 current screenshot only.

Fixed: the earlier runs (3 canonical calls + 1 shuffled; average single call). Random: 2 calls per state, each in its own
seeded random order. Scored on held-out draws 3-7. Sets: Piotr-task states, valid draws only (reporting set); all
states, invalid = failure. Paired, task-clustered differences. Also pick rate by displayed position."""
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

W = Path(__file__).resolve().parent; B = W.parent
F, E = B / "lookahead-fast1k-20261013", B / "lookahead-v2-20261013"
LIN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-lineage")
piotr = set(read(LIN / "task-membership.private.json")["piotr_training_task_counts"])
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))
CONDS = {"before": "Pre-execution", "immediate": "Post-execution", "k3": "K=3", "k3-img1": "K=3, current screenshot only"}
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


def load(root):
    out = defaultdict(list)
    if (root / "judgments").is_dir():
        for f in (root / "judgments").glob("*.json"):
            for j in json.loads(f.read_text()):
                if j["valid"]:
                    out[j["state"]].append((j["candidate"], j["order"].index(j["candidate"])))
    return out


fixed = {c: {**load(E / c), **load(F / c)} for c in CONDS}
rand = {c: load(W / c) for c in CONDS}
out = {"pick_rate_by_displayed_position": {}}
for c in CONDS:
    for kind, d in (("fixed", fixed[c]), ("random", rand[c])):
        pos = [p for v in d.values() for _, p in v]
        out["pick_rate_by_displayed_position"][f"{CONDS[c]}, {kind}"] = [round(100 * x, 1) for x in np.bincount(pos, minlength=5) / len(pos)]
for setname, valid_only, keep in (("Piotr tasks, valid draws only", True, lambda s: meta[s]["piotr"]),
                                  ("all states, invalid = failure", False, lambda s: True)):
    res = {}
    for c in CONDS:
        ss = sorted(s for s in set(fixed[c]) & set(rand[c]) if keep(s))
        per = {}
        for s in ss:
            h = []
            for i in range(5):
                cells = [slot[s][i, k] for k in range(3, 8) if (i, k) in slot[s]]
                cells = [x for x in cells if x[0]] if valid_only else cells
                h.append(np.mean([x[1] for x in cells]) if cells else None)
            if None in h:
                continue
            h = np.array(h)
            per[s] = dict(fixed=float(np.mean([h[i] for i, _ in fixed[c][s]])), random=float(np.mean([h[i] for i, _ in rand[c][s]])),
                          uniform=float(h.mean()), actor=float(h[0]))
        st = sorted(per); g = [meta[s]["task"] for s in st]
        res[CONDS[c]] = dict(states=len(st), tasks=len(set(g)),
                             fixed=round(100 * np.mean([per[s]["fixed"] for s in st]), 2), random=round(100 * np.mean([per[s]["random"] for s in st]), 2),
                             uniform=round(100 * np.mean([per[s]["uniform"] for s in st]), 2), actor=round(100 * np.mean([per[s]["actor"] for s in st]), 2),
                             random_minus_fixed=cluster_boot([per[s]["random"] - per[s]["fixed"] for s in st], g), _per=per)
    # gaps to K = 3 under each ordering, on states shared by both conditions
    for base in ("Pre-execution", "Post-execution"):
        for kind in ("fixed", "random"):
            ss = sorted(set(res["K=3"]["_per"]) & set(res[base]["_per"]))
            res[f"K=3 - {base}, {kind} order"] = dict(states=len(ss), diff=cluster_boot(
                [res["K=3"]["_per"][s][kind] - res[base]["_per"][s][kind] for s in ss], [meta[s]["task"] for s in ss]))
    for v in res.values():
        v.pop("_per", None)
    out[setname] = res
(W / "random-vs-fixed.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
