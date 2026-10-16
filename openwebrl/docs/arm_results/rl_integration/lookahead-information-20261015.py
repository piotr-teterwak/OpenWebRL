"""What does the 3-step lookahead show that post-execution evidence does not? (421 fast1k states, existing labels only)

For each candidate's draw-0 continuation (the same steps Luna saw): text signals in the next 3 steps. Compares the
signals for K = 3's and post-execution's majority picks where they differ, relates each signal to held-out success, and
writes local example cases (raw trajectory text; not for publication)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parent
SRC = V / "k3" / "source"
sys.path[:0] = [str(SRC), str(SRC / "scripts"), "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts"]
import run_branch_lookahead_labels as L  # noqa: E402
from selection_outcome_experiments import cluster_boot  # noqa: E402

L.K, L.FEEDBACK, L.HIDE_ANSWER = 3, True, True
read = lambda p: json.loads(Path(p).read_text())
why = read(V / "why-lookahead.json")["rows"]
plan = {e["state"]: e for e in read(V / "k3" / "plan.json")["states"]}
TOOL = re.compile(r'"name":\s*"([a-z_]+)"')
NAV = re.compile(r"(?:navigated to|Page navigated to|goto_url` navigated to)\s*(\S+)", re.I)


def action_key(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    return re.sub(r"\s+", "", m.group(1)) if m else text.strip()[:200]


def signals(cand_text, imm, look):
    steps = [s["action"] for s in look["steps"]]
    joined = " ".join(steps)
    tools = TOOL.findall(joined)
    keys = [action_key(cand_text)] + [action_key(s) for s in steps if "[the episode ended" not in s]
    imm_text = json.dumps(imm.get("tool_feedback", {}).get("tool_responses", []), ensure_ascii=False)
    return {
        "goes back within 3 steps": "go_back" in tools,
        "repeats an action": len(keys) != len(set(keys)),
        "error or failure reply": bool(re.search(r"\b(error|failed|fail|timeout|not found|invalid|unable)\b", joined, re.I)),
        "'no visible navigation' reply": "no visible navigation" in joined.lower(),
        "candidate itself: 'no visible navigation'": "no visible navigation" in imm_text.lower(),
        "actor stops within 3 steps": bool(look["ended"]),
        "reaches 2+ new pages": len(set(NAV.findall(joined))) >= 2,
        "types or searches": any(t in ("write", "type", "search", "fill") for t in tools),
        "scrolls only": bool(tools) and all(t in ("scroll", "scroll_down", "scroll_up") for t in tools),
    }


cases, feats = [], {}
for r in why:
    e = plan[r["state"]]
    anchor = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
    ev = L.draw0(e)
    per = []
    for i in range(5):
        look = L.lookahead(ev[i][0], anchor["depth"], anchor["candidates"][i]["output"][0])
        per.append(dict(cand=L.abbreviate(anchor["candidates"][i]["output"][0]), imm=ev[i][1], look=look,
                        sig=signals(anchor["candidates"][i]["output"][0], ev[i][1], look) if look else None))
    feats[r["state"]] = per
    cases.append(dict(r=r, anchor=anchor, per=per))

NAMES = list(next(p["sig"] for c in cases for p in c["per"] if p["sig"]).keys())
out = {"states": len(cases)}
# 1. signals of K = 3's vs post-execution's pick where they differ
diff = [c for c in cases if c["r"]["mode"]["k3"] != c["r"]["mode"]["immediate"]]
out["disagreement_states"] = len(diff)
out["signal_rate_pct_k3_pick_vs_post_pick"] = {
    n: [round(100 * np.mean([c["per"][c["r"]["mode"]["k3"]]["sig"][n] for c in diff]), 1),
        round(100 * np.mean([c["per"][c["r"]["mode"]["immediate"]]["sig"][n] for c in diff]), 1)] for n in NAMES}
# 2. association of each signal with held-out success (within-state: candidate success minus state mean)
assoc = {}
for n in NAMES:
    with_, without = [], []
    for c in cases:
        hold = np.array(c["r"]["hold"]); dm = hold - hold.mean()
        for i, p in enumerate(c["per"]):
            (with_ if p["sig"][n] else without).append(dm[i])
    assoc[n] = dict(candidates_with=len(with_), share_pct=round(100 * len(with_) / (len(with_) + len(without)), 1),
                    heldout_vs_state_mean_pp=round(100 * float(np.mean(with_)), 2) if with_ else None)
out["signal_vs_heldout_success"] = assoc
(V / "lookahead-information.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))


# 3. local examples: K = 3 right / post wrong (largest held-out gap, consistent K = 3 votes, distinct tasks), and reverse
def fmt(c, title):
    r, a = c["r"], c["anchor"]
    task = a["task"] if isinstance(a["task"], str) else json.dumps(a["task"])
    url = a["before_observation"].get("active_tab_url", "")
    lines = [f"## {title}", f"- **Task:** {task.strip()}", f"- **Page:** {url}  (decision depth {a['depth']})",
             f"- **Picks (4 calls):** pre {r['votes']['before']}, post {r['votes']['immediate']}, K=3 {r['votes']['k3']}  (vote share per candidate 1-5)",
             f"- **Held-out success, draws 3-7:** " + ", ".join(f"c{i + 1} {100 * h:.0f}%" for i, h in enumerate(r["hold"])), ""]
    for i, p in enumerate(c["per"]):
        mark = (" ← K=3" if i == r["mode"]["k3"] else "") + (" ← post" if i == r["mode"]["immediate"] else "")
        fb = json.dumps(p["imm"].get("tool_feedback", {}).get("tool_responses", []), ensure_ascii=False)[:220]
        lines.append(f"**Candidate {i + 1}{mark}** (held-out {100 * r['hold'][i]:.0f}%): `{p['cand'][:220]}`")
        lines.append(f"  - immediate reply: {fb}")
        for k, s in enumerate(p["look"]["steps"], 1):
            lines.append(f"  - next step {k}: {re.sub(chr(10), ' ', s['action'])[:300]}")
        on = [n for n, v in p["sig"].items() if v]
        lines.append(f"  - signals: {', '.join(on) if on else 'none'}")
    return "\n".join(lines) + "\n"


def pick(pool, n):
    chosen, tasks = [], set()
    for c in pool:
        if c["r"]["task"] not in tasks:
            chosen.append(c); tasks.add(c["r"]["task"])
        if len(chosen) == n:
            break
    return chosen


gap = lambda c: c["r"]["hold"][c["r"]["mode"]["k3"]] - c["r"]["hold"][c["r"]["mode"]["immediate"]]
good = sorted([c for c in diff if gap(c) >= 0.4 and c["r"]["agree"]["k3"] >= 0.75], key=gap, reverse=True)
bad = sorted([c for c in diff if gap(c) <= -0.4 and c["r"]["agree"]["immediate"] >= 0.75], key=gap)
md = ["# Lookahead examples (local only: raw trajectory text)", "",
      f"{len(good)} states where K = 3's consistent pick beats post-execution's by ≥ 40 pp held-out; {len(bad)} where post-execution wins by ≥ 40 pp.", ""]
md += [fmt(c, f"K = 3 right, post-execution wrong #{k + 1} ({c['r']['domain']})") for k, c in enumerate(pick(good, 6))]
md += [fmt(c, f"Post-execution right, K = 3 wrong #{k + 1} ({c['r']['domain']})") for k, c in enumerate(pick(bad, 3))]
(V / "lookahead-examples.local.md").write_text("\n".join(md))
print("examples:", len(good), len(bad))
