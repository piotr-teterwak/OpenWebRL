#!/usr/bin/env python3
"""Why does immediate post-action evidence not improve teacher picks? (CPU, saved data only, no API calls)

States: the labelled states with held-out draws 3-7 and complete draw-0 evidence. Outcomes are scored on
draws 3-7 (invalid = 0). Immediate evidence per candidate and draw comes from immediate.json.

  1. Does the pick change?  modal (action-equivalent) before vs after pick, per teacher; success when it changes.
  2. Is the evidence informative?  distinct post-action results (URL + screenshot) among a state's distinct actions.
  3. Are visible failures avoided?  candidates whose action errored or left the screen unchanged: pick rates
     before vs after, and their own held-out success.
  4. Does the immediate result predict the outcome?  within-state success of candidates by immediate-result type.
  5. Does the evidence replay?  same URL / screenshot for a candidate across draws 0-7.
  6. Gain (after - before) by evidence informativeness and by how much the candidates' outcomes differ.
  7. Cases where the after pick changed.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt_selection_head_outcomes import ORIG, R, outcomes  # noqa: E402
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402

LUNA, SOL, SOL_AFTER = R / "luna-labels-20261011", R / "sol-labels-20261011", R / "sol-after-labels-20261011"
OUT = R / "post-action-analysis-20261011"


def read(p):
    return json.loads(Path(p).read_text())


def boot(diffs, groups, n=10000, seed=0):
    g = defaultdict(list)
    for d, t in zip(diffs, groups):
        g[t].append(d)
    g = list(g.values())
    if not g:
        return None
    rng, m = random.Random(seed), []
    for _ in range(n):
        x = [rng.choice(g) for _ in g]
        m.append(sum(map(sum, x)) / sum(map(len, x)))
    m.sort()
    return [100 * sum(diffs) / len(diffs), 100 * m[int(.025 * n)], 100 * m[int(.975 * n)]]


def main():
    cell = outcomes()
    # teacher judgments: (teacher, condition) -> state -> picks (draw-0 subset for after, matching Sol's layout)
    picks = defaultdict(lambda: defaultdict(list))
    for root in (ORIG, LUNA):
        db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") != "teacher":
                continue
            for j in read(r["judgments_path"]):
                if not j["valid"]:
                    continue
                if j["condition"] == "before":
                    picks["Luna", "before"][j["state"]].append(j["candidate"])
                elif j["execution_repeat"] == 0:
                    picks["Luna", "after"][j["state"]].append(j["candidate"])
    for root, cond in ((SOL, "before"), (SOL_AFTER, "after")):
        for f in (root / "judgments").glob("*.json"):
            for j in read(f):
                if j["valid"]:
                    picks["Sol", cond][j["state"]].append(j["candidate"])
    # anchors and releases (all draws, for evidence analysis only)
    where = {}
    for root in (ORIG, R / "luna-labels-20261011"):
        for item in read(root / "plan.json")["schedule"]:
            where.setdefault(item["state"], (root, item["task_id"]))
    sources = read(LUNA / "plan.json")["source_collections"]
    states = [s for s in picks["Sol", "after"] if picks["Luna", "after"].get(s) and picks["Luna", "before"].get(s)
              and picks["Sol", "before"].get(s) and all(all((i, r) in cell[s] for r in range(3, 8)) for i in range(5))]
    rows = []
    for s in states:
        root, task = where[s]
        ref = read(root / "states" / s / "anchor-reference.json")
        anchor = read(ref["path"])
        # full release from the original collection (evidence root for new states lists only draws 0-2)
        rel_path = root / "states" / s / "release.json"
        rel = read(rel_path)
        if "source_release" in rel:
            rel = read(rel["source_release"])
        imm = {}
        for d in rel["attempt_directories"]:
            f = Path(d) / "immediate.json"
            if f.exists():
                x = read(f)
                imm[x["candidate"], x["repeat"]] = x
        calls = [parse_calls(c["output"][0]) for c in anchor["candidates"]]
        eq = [[j for j in range(5) if j == i or equivalent(calls[i], calls[j])] for i in range(5)]
        before_shot = anchor["before_observation"]["screenshot"]["sha256"]
        before_url = anchor["before_observation"].get("active_tab_url", "")
        def evidence(i, r):
            x = imm.get((i, r))
            if x is None:
                return None
            obs = x["observation"]
            resp = " ".join(t.get("tool_response", "") for t in x["tool_feedback"].get("tool_responses", []))
            return dict(url=obs.get("active_tab_url", ""), shot=obs["screenshot"]["sha256"],
                        unchanged=obs["screenshot"]["sha256"] == before_shot,
                        navigated=obs.get("active_tab_url", "") != before_url,
                        error=not resp.strip().startswith("Succeed") if resp.strip() else False,
                        terminated=bool(x.get("terminated")))
        ev0 = [evidence(i, 0) for i in range(5)]
        hold = [sum(cell[s][i, r] for r in range(3, 8)) / 5 for i in range(5)]
        early = [sum(cell[s][i, r] for r in range(3)) / 3 for i in range(5)]   # independent of the scoring draws
        rows.append(dict(state=s, task=task, collection="historical" if root == ORIG else "new", calls=calls, eq=eq,
                         kinds=[c[0]["name"] if c else "unparsed" for c in calls], ev0=ev0,
                         ev_all={(i, r): evidence(i, r) for i in range(5) for r in range(8)},
                         hold=hold, early=early, task_text=anchor["task"]))
    n = len(rows)
    groups = [r["task"] for r in rows]
    report = {"states": n, "by_collection": dict(Counter(r["collection"] for r in rows))}

    def expected(r, plist):
        return sum(r["hold"][c] for c in plist) / len(plist)

    def modal(r, plist):
        # action-equivalence class with the most votes; ties broken by lowest index
        votes = Counter(min(r["eq"][c]) for c in plist)
        return max(sorted(votes), key=lambda k: votes[k])

    # 1. does the pick change, and is the changed pick better?
    change = {}
    for teacher in ("Luna", "Sol"):
        b, a = picks[teacher, "before"], picks[teacher, "after"]
        changed = [r for r in rows if modal(r, b[r["state"]]) != modal(r, a[r["state"]])]
        d_all = [expected(r, a[r["state"]]) - expected(r, b[r["state"]]) for r in rows]
        d_ch = [expected(r, a[r["state"]]) - expected(r, b[r["state"]]) for r in changed]
        better = sum(r["hold"][modal(r, a[r["state"]])] > r["hold"][modal(r, b[r["state"]])] for r in changed)
        worse = sum(r["hold"][modal(r, a[r["state"]])] < r["hold"][modal(r, b[r["state"]])] for r in changed)
        change[teacher] = {"states_modal_pick_changed": len(changed), "share_changed": len(changed) / n,
                           "after_minus_before_all": boot(d_all, groups),
                           "after_minus_before_changed_states": boot(d_ch, [r["task"] for r in changed]),
                           "changed_pick_better": better, "changed_pick_worse": worse,
                           "changed_pick_tied": len(changed) - better - worse}
    report["1_pick_changes"] = change

    # 2. evidence informativeness: distinct post-action results among distinct actions (draw 0)
    def informative(r):
        reps = {min(g) for g in r["eq"]}                        # one candidate per distinct action
        results = {(r["ev0"][i]["url"], r["ev0"][i]["shot"]) for i in reps}
        return len(reps), len(results)
    info = Counter()
    for r in rows:
        k, m = informative(r)
        r["distinct_actions"], r["distinct_results"] = k, m
        info["single distinct action" if k == 1 else ("all distinct actions look identical after execution" if m == 1
             else ("some distinct actions look identical" if m < k else "every distinct action looks different"))] += 1
    report["2_evidence_informativeness"] = dict(info)

    # 3. visible failures: errored or unchanged-screen candidates
    fail_stats = {"candidates": 0, "errored": 0, "unchanged_screen": 0}
    vis = []
    for r in rows:
        for i, e in enumerate(r["ev0"]):
            fail_stats["candidates"] += 1
            fail_stats["errored"] += e["error"]; fail_stats["unchanged_screen"] += e["unchanged"]
            if e["error"] or e["unchanged"]:
                vis.append((r, i))
    rel_success = [r["hold"][i] - sum(r["hold"]) / 5 for r, i in vis]
    fail_stats["visible_failure_candidates"] = len(vis)
    fail_stats["their_within_state_relative_success_pp"] = 100 * sum(rel_success) / max(1, len(rel_success))
    for teacher in ("Luna", "Sol"):
        for cond in ("before", "after"):
            P = picks[teacher, cond]
            share = [sum(c == i for c in P[r["state"]]) / len(P[r["state"]]) for r, i in vis]
            fail_stats[f"{teacher}_{cond}_pick_share_of_visible_failures"] = sum(share) / max(1, len(share))
    fail_stats["uniform_pick_share_of_visible_failures"] = 0.2
    report["3_visible_failures"] = fail_stats

    # 4. does the immediate result predict the held-out outcome? within-state relative success by result type
    pred = defaultdict(list)
    for r in rows:
        mean = sum(r["hold"]) / 5
        for i, e in enumerate(r["ev0"]):
            key = ("terminated (done)" if e["terminated"] else "errored" if e["error"] else
                   "screen unchanged" if e["unchanged"] else "navigated to a new URL" if e["navigated"] else "same URL, screen changed")
            pred[key].append(r["hold"][i] - mean)
    report["4_immediate_result_vs_outcome"] = {k: {"candidates": len(v), "relative_success_pp": 100 * sum(v) / len(v)}
                                               for k, v in sorted(pred.items(), key=lambda kv: -len(kv[1]))}

    # 5. replay consistency of the evidence across draws 0-7
    same_url = same_shot = pairs = 0
    for r in rows:
        for i in range(5):
            e0 = r["ev_all"][i, 0]
            for rr in range(1, 8):
                e = r["ev_all"][i, rr]
                if e0 and e:
                    pairs += 1; same_url += e["url"] == e0["url"]; same_shot += e["shot"] == e0["shot"]
    report["5_replay_consistency"] = {"candidate_draw_pairs": pairs, "same_url_as_draw0": same_url / pairs,
                                      "same_screenshot_as_draw0": same_shot / pairs}

    # 6. gain by informativeness and by outcome spread among candidates
    strata = defaultdict(list)
    for r in rows:
        spread = max(r["early"]) - min(r["early"])   # stratify on draws 0-2, never on the scored draws
        key_info = "evidence distinguishes all distinct actions" if r["distinct_results"] == r["distinct_actions"] > 1 else "evidence partly or not distinguishing"
        key_spread = "draws 0-2: outcomes differ by >=2 of 3" if spread >= .6 else ("draws 0-2: outcomes differ by 1 of 3" if spread > 0 else "draws 0-2: outcomes identical")
        for teacher in ("Luna", "Sol"):
            d = expected(r, picks[teacher, "after"][r["state"]]) - expected(r, picks[teacher, "before"][r["state"]])
            strata[teacher, "info", key_info].append((d, r["task"]))
            strata[teacher, "spread", key_spread].append((d, r["task"]))
    # absolute success per picker within each informativeness stratum, paired on the same states
    def stratum(r):
        if r["distinct_actions"] == 1: return "single distinct action"
        if r["distinct_results"] == r["distinct_actions"]: return "every distinct action looks different"
        return "some or all distinct actions look identical"
    absolute = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = stratum(r)
        for t in ("Luna", "Sol"):
            for c in ("before", "after"):
                absolute[k][f"{t} {c}"].append(expected(r, picks[t, c][r["state"]]))
        absolute[k]["Uniform"].append(sum(r["hold"]) / 5)
        absolute[k]["Outcome reference: best on draws 0-2"].append(
            sum(r["hold"][i] for i in range(5) if r["early"][i] == max(r["early"])) / sum(1 for i in range(5) if r["early"][i] == max(r["early"])))
        absolute[k]["Hindsight best on draws 3-7"].append(max(r["hold"]))
    strat_rows = {k: [r for r in rows if stratum(r) == k] for k in absolute}
    report["6a_absolute_by_informativeness"] = {
        k: {"states": len(strat_rows[k]), **{name: 100 * sum(v) / len(v) for name, v in d.items()},
            **{f"{t} after - before": boot([expected(r, picks[t, "after"][r["state"]]) - expected(r, picks[t, "before"][r["state"]]) for r in strat_rows[k]],
                                         [r["task"] for r in strat_rows[k]]) for t in ("Luna", "Sol")},
            "Sol after - uniform": boot([expected(r, picks["Sol", "after"][r["state"]]) - sum(r["hold"]) / 5 for r in strat_rows[k]], [r["task"] for r in strat_rows[k]]),
            "Luna after - uniform": boot([expected(r, picks["Luna", "after"][r["state"]]) - sum(r["hold"]) / 5 for r in strat_rows[k]], [r["task"] for r in strat_rows[k]])}
        for k, d in absolute.items()}
    report["6_gain_by_stratum"] = {f"{t} | {kind} | {k}": {"states": len(v), "after_minus_before_pp": boot([x for x, _ in v], [g for _, g in v])}
                                   for (t, kind, k), v in sorted(strata.items())}

    # 7. cases: the changed picks with the largest outcome swings
    cases = []
    for teacher in ("Luna", "Sol"):
        b, a = picks[teacher, "before"], picks[teacher, "after"]
        for r in rows:
            mb, ma = modal(r, b[r["state"]]), modal(r, a[r["state"]])
            if mb != ma:
                cases.append(dict(teacher=teacher, state=r["state"], task=r["task_text"][:160],
                                  before_pick=dict(index=mb, action=r["calls"][mb], held_out_success=r["hold"][mb], evidence=r["ev0"][mb]),
                                  after_pick=dict(index=ma, action=r["calls"][ma], held_out_success=r["hold"][ma], evidence=r["ev0"][ma]),
                                  swing=r["hold"][ma] - r["hold"][mb]))
    cases.sort(key=lambda c: c["swing"])
    report["7_case_counts"] = {"changed_cases": len(cases)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
