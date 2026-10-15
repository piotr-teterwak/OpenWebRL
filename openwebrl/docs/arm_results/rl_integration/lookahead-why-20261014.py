"""Why does K = 3 lookahead give better selection labels than pre- or post-execution? (421 fast1k states, no new calls)

Per state: each condition's 4 Luna picks; held-out success per candidate (draws 3-7, invalid = failure); draw-0 outcome;
features of the evidence Luna saw (draw-0 rollout ended within 3 steps, post-action screenshots distinct across
candidates, no-op actions); domain and depth. Writes why-lookahead.json."""
import base64
import io
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from selection_outcome_experiments import cluster_boot, read  # noqa: E402

V = Path(__file__).resolve().parent
CONDS = ["before", "immediate", "k1", "k2", "k3"]
K = 3


def small(raw):
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L").resize((96, 54)), dtype=np.float32)


plan = read(V / "k3" / "plan.json")
slot = defaultdict(dict)
for root in {e["root"] for e in plan["states"]}:
    db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
    for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
        r = json.loads(raw)
        if r.get("mode") == "branch" and r.get("released"):
            v = bool(r["valid"])
            slot[r["state"]].setdefault((r["candidate"], r["repeat"]), v and (r.get("reward") or 0) >= 1)
picks, shown = {c: defaultdict(list) for c in CONDS}, {c: defaultdict(list) for c in CONDS}
for c in CONDS:
    for f in (V / c / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[c][j["state"]].append(j["candidate"])
                if j["teacher_repeat"] != "permuted":
                    shown[c][j["state"]].append(j["order"].index(j["candidate"]))     # displayed position
states = sorted(set.intersection(*(set(picks[c]) for c in CONDS)))

rows = []
for e in plan["states"]:
    s = e["state"]
    if s not in set(states):
        continue
    sd = Path(e["root"]) / "states" / s
    anchor = read(read(sd / "anchor-reference.json")["path"])
    depth = anchor["depth"]
    url = anchor["before_observation"].get("active_tab_url") or ""
    host = urlparse(url).netloc
    domain = ("error/blank" if (not host or url.startswith(("about:", "chrome-error", "chrome://")))
              else "arxiv" if "arxiv" in host else "other")
    before_img = small(Path(anchor["before_observation"]["screenshot"]["path"]).read_bytes())
    post, ended, steps = {}, {}, {}
    for d in read(sd / "release.json")["attempt_directories"]:
        d = Path(d)
        if not (d / "immediate.json").exists():
            continue
        im = read(d / "immediate.json")
        if im["repeat"] != 0:
            continue
        i = im["candidate"]
        post[i] = small(Path(im["observation"]["screenshot"]["path"]).read_bytes())
        files = list((d / "rollouts").rglob("*.json"))
        r = read(files[0])
        n_turns = len(r["llm_input_texts"].split("<|im_start|>assistant\n")) - 1
        ended[i] = n_turns - 1 <= depth + K           # the episode's last turn is within the K shown steps
        steps[i] = n_turns - 1 - depth
    hold = np.array([np.mean([slot[s][i, k] for k in range(3, 8)]) for i in range(5)])
    d0 = np.array([float(slot[s][i, 0]) for i in range(5)])
    # distinct post-action screenshots (mean-abs-diff > 2 on a 0-255 scale counts as different)
    distinct = []
    for i in range(5):
        if all(np.abs(post[i] - post[j]).mean() > 2 for j in distinct):
            distinct.append(i)
    noop = sum(np.abs(post[i] - before_img).mean() <= 2 for i in range(5))
    rows.append(dict(state=s, task=read(read(sd / "anchor-reference.json")["path"])["task_id"], domain=domain, depth=depth,
                     hold=hold.tolist(), d0=d0.tolist(), distinct_post=len(distinct), noop_actions=int(noop),
                     ended_within_k=int(sum(ended.values())), any_ended=bool(any(ended.values())),
                     score={c: float(np.mean([hold[x] for x in picks[c][s]])) for c in CONDS},
                     pick_d0_success={c: float(np.mean([d0[x] for x in picks[c][s]])) for c in CONDS},
                     mode={c: Counter(picks[c][s]).most_common(1)[0][0] for c in CONDS},
                     agree={c: Counter(picks[c][s]).most_common(1)[0][1] / len(picks[c][s]) for c in CONDS},
                     pos1={c: float(np.mean([p == 0 for p in shown[c][s]])) for c in CONDS},
                     votes={c: [picks[c][s].count(i) / len(picks[c][s]) for i in range(5)] for c in CONDS},
                     uniform=float(hold.mean()), d0_oracle=float(hold[d0 == d0.max()].mean())))

g = lambda rs: [r["task"] for r in rs]
boot = lambda vals, rs: cluster_boot(vals, g(rs))
out = {"states": len(rows), "tasks": len({r["task"] for r in rows})}
out["success"] = {c: boot([r["score"][c] for r in rows], rows) for c in CONDS}
out["success"]["uniform"] = boot([r["uniform"] for r in rows], rows)
out["success"]["oracle_draw0_only"] = boot([r["d0_oracle"] for r in rows], rows)
# 1. disagreement: where K = 3's majority pick differs, which pick is better on held-out draws
out["disagreement"] = {}
for other in ("before", "immediate"):
    diff = [r for r in rows if r["mode"]["k3"] != r["mode"][other]]
    dv = [r["hold"][r["mode"]["k3"]] - r["hold"][r["mode"][other]] for r in diff]
    out["disagreement"][f"k3_vs_{other}"] = dict(states_differ=len(diff), share=round(len(diff) / len(rows), 3),
        k3_better=sum(x > 0 for x in dv), other_better=sum(x < 0 for x in dv), tie=sum(x == 0 for x in dv),
        mean_heldout_diff_pp=round(100 * float(np.mean(dv)), 2) if dv else None)
# 2-3. strata: K = 3 minus post / pre within each
strata = {"any draw-0 continuation ended within 3 steps": lambda r: r["any_ended"],
          "no continuation ended within 3 steps": lambda r: not r["any_ended"],
          "post screenshots: 1 distinct (post cannot discriminate)": lambda r: r["distinct_post"] == 1,
          "post screenshots: 2-3 distinct": lambda r: 2 <= r["distinct_post"] <= 3,
          "post screenshots: 4-5 distinct": lambda r: r["distinct_post"] >= 4,
          "some candidates are no-ops": lambda r: r["noop_actions"] >= 1,
          "domain: arxiv": lambda r: r["domain"] == "arxiv", "domain: error/blank": lambda r: r["domain"] == "error/blank",
          "domain: other": lambda r: r["domain"] == "other",
          "depth 1-2": lambda r: r["depth"] <= 2, "depth 3-5": lambda r: 3 <= r["depth"] <= 5, "depth 6+": lambda r: r["depth"] >= 6}
out["strata"] = {}
for name, f in strata.items():
    rs = [r for r in rows if f(r)]
    if len(rs) < 15:
        out["strata"][name] = dict(states=len(rs)); continue
    out["strata"][name] = dict(states=len(rs), tasks=len({r["task"] for r in rs}),
        success={c: round(100 * float(np.mean([r["score"][c] for r in rs])), 2) for c in CONDS + ["uniform"] if c != "uniform"} |
                {"uniform": round(100 * float(np.mean([r["uniform"] for r in rs])), 2)},
        k3_minus_post=boot([r["score"]["k3"] - r["score"]["immediate"] for r in rs], rs),
        k3_minus_pre=boot([r["score"]["k3"] - r["score"]["before"] for r in rs], rs))
# 3. how often the pick's own draw 0 succeeded (the lookahead is the start of draw 0)
out["pick_draw0_success_pct"] = {c: round(100 * float(np.mean([r["pick_d0_success"][c] for r in rows])), 2) for c in CONDS}
out["pick_draw0_success_pct"]["uniform"] = round(100 * float(np.mean([np.mean(r["d0"]) for r in rows])), 2)
# 4. teacher behaviour
out["self_agreement"] = {c: round(float(np.mean([r["agree"][c] for r in rows])), 3) for c in CONDS}
out["picks_displayed_first_pct"] = {c: round(100 * float(np.mean([r["pos1"][c] for r in rows])), 1) for c in CONDS}
out["vote_share_vs_heldout_corr"] = {}
for c in CONDS:
    xs = [v for r in rows for v in r["votes"][c]]; ys = [h - np.mean(r["hold"]) for r in rows for h in r["hold"]]
    out["vote_share_vs_heldout_corr"][c] = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
out["feature_counts"] = dict(any_ended=sum(r["any_ended"] for r in rows),
                             distinct_post=dict(sorted(Counter(r["distinct_post"] for r in rows).items())),
                             domains=dict(Counter(r["domain"] for r in rows)))
(V / "why-lookahead.json").write_text(json.dumps(dict(summary=out, rows=rows), indent=1))
print(json.dumps(out, indent=1))
