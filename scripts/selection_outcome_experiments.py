#!/usr/bin/env python3
"""Early-signal experiments on outcome-labelled branching states (CPU; cached features; no new collection).

Same 138 states, 5 task-grouped folds, labels (draws 0-5) and scoring rules as
adapt_selection_head_outcomes.py. Held-out states are scored on all draws 0-7 (no held-out
draw enters training) and on draws 6-7 for parity with the outcome reference.

  probe   Ridge probes (scikit-learn RidgeCV; penalty by 4-fold task-grouped CV inside training
          folds only), predicting within-state-centred success from within-state-centred features.
  power   Variance decomposition (split-half draws) and states needed for 80% power.
  depth   Success by collection x decision depth and by action type; depth-reweighted comparisons.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_selection_head_outcomes import FIT, HOLD, R, SOURCES, outcomes, picks_from, roles  # noqa: E402
from train_selection_head import load, predict  # noqa: E402

ALL = range(8)


def build(cache):
    """Rows, features, draws and saved picks for every state with features and draws 0-7."""
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, equivalent, parse_calls
    saved = torch.load(R / "heads/pointwise/head.pt", map_location="cpu")
    config = HeadConfig.from_json(saved["config"])
    teacher = CandidateSetHead(config); teacher.load_state_dict(saved["state_dict"]); teacher.eval()
    full = argparse.Namespace(layers="9,18,27,36", pools="thought_mean,action_mean,last")
    head_args = argparse.Namespace(layers=",".join(map(str, config.layers)), pools=",".join(config.pools))
    cell, role = outcomes(), roles()
    rows, feats, states, draws, valid_logits, others = [], [], [], [], [], defaultdict(dict)
    for panels, features, directory in SOURCES:
        (split,) = load(full, torch, panels / "panels.jsonl", features)[0].values()
        (hsplit,) = load(head_args, torch, panels / "panels.jsonl", features)[0].values()
        tlog = predict(teacher, hsplit, "cpu", torch)
        for name, chosen in picks_from(directory, split["rows"]).items():
            others[name].update(chosen)
        for k, row in enumerate(split["rows"]):
            s = row["state_id"]
            if not all((i, r) in cell[s] for i in range(5) for r in range(8)):
                continue
            calls = [parse_calls(c) for c in row["candidates"]]
            row = dict(row, role=role.get(s, "historical"),
                       action_kinds=[c[0]["name"] if c else "unparsed" for c in calls],
                       equivalent=[[j for j in range(5) if j == i or equivalent(calls[i], calls[j])] for i in range(5)])
            rows.append(row); feats.append(split["candidates"][k]); states.append(split["state"][k])
            draws.append([[float(cell[s][i, r]) for r in range(8)] for i in range(5)])
            valid_logits.append(tlog[k])
    torch.save({"rows": rows, "features": torch.stack(feats).half(), "state": torch.stack(states).half(),
                "draws": torch.tensor(draws), "teacher_logits": torch.stack(valid_logits),
                "others": {k: dict(v) for k, v in others.items()}, "layers": [9, 18, 27, 36],
                "pools": ["thought_mean", "action_mean", "last"]}, cache)


def folds_of(rows, n_folds=5):
    """Identical to adapt_selection_head_outcomes.py: hash-ordered tasks, greedy balance."""
    by_task = defaultdict(list)
    for k, row in enumerate(rows):
        by_task[row["task_id"]].append(k)
    fold_of, sizes = {}, [0] * n_folds
    for task in sorted(by_task, key=lambda t: hashlib.sha256(t.encode()).hexdigest()):
        f = min(range(n_folds), key=lambda j: (sizes[j], j))
        for k in by_task[task]:
            fold_of[k] = f
        sizes[f] += len(by_task[task])
    return fold_of, sizes


def cluster_boot(diff, groups, n=10000, seed=0):
    """Mean of per-state differences with a whole-task-group percentile bootstrap."""
    g = defaultdict(list)
    for d, t in zip(diff, groups):
        g[t].append(d)
    g = list(g.values()); rng = random.Random(seed); m = []
    for _ in range(n):
        x = [rng.choice(g) for _ in g]
        m.append(sum(map(sum, x)) / sum(map(len, x)))
    m.sort()
    return [100 * sum(diff) / len(diff), 100 * m[int(.025 * n)], 100 * m[int(.975 * n)]]


def references(data):
    """Per-state chosen-action success for the saved selectors and baselines (draws 0-7 and 6-7)."""
    import numpy as np
    rows, D = data["rows"], data["draws"].numpy()
    rate = lambda reps: D[:, :, list(reps)].mean(-1)
    out = {}
    for reps_name, reps in (("0_7", ALL), ("6_7", HOLD)):
        r = rate(reps)
        out[reps_name] = {name: np.array([r[k, chosen[row["state_id"]]] for k, row in enumerate(rows)])
                          for name, chosen in data["others"].items()}
        out[reps_name]["Actor's first sample"] = r[:, 0]
        out[reps_name]["Uniform"] = r.mean(1)
        if reps_name == "6_7":   # its selection draws (0-5) must not be scored
            fit = equivalent_rates(data, FIT)
            out[reps_name]["Outcome reference: best on draws 0-5"] = np.array(
                [r[k, fit[k] == fit[k].max()].mean() for k in range(len(rows))])
    return out


def equivalent_rates(data, reps):
    import numpy as np
    D = data["draws"].numpy()[:, :, list(reps)].mean(-1)
    return np.array([[D[k, g].mean() for g in row["equivalent"]] for k, row in enumerate(data["rows"])])


def probe(data, out):
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    rows, n = data["rows"], len(data["rows"])
    fold_of, sizes = folds_of(rows)
    assert sizes == [28, 29, 27, 27, 27], sizes
    X_all = data["features"].float().numpy().reshape(len(rows), 5, 12, -1)   # layer-major x (thought, action, last)
    layers, pools = data["layers"], data["pools"]
    slot = lambda l, p: layers.index(l) * len(pools) + pools.index(p)
    feature_sets = {
        "Ridge, all 12 feature blocks": list(range(12)),
        "Ridge, layers 18/27/36, all pools (teacher-head inputs)": [slot(l, p) for l in (18, 27, 36) for p in pools],
        "Ridge, layer 27 action tokens": [slot(27, "action_mean")],
        "Ridge, layer 36 end token": [slot(36, "last")],
    }
    y_rate = equivalent_rates(data, FIT)                  # [n, 5]
    y = y_rate - y_rate.mean(1, keepdims=True)            # within-state comparison only
    D = data["draws"].numpy()
    teacher = data["teacher_logits"].numpy()
    alphas = np.logspace(0, 9, 28)
    results, fits = {}, {}
    for name, blocks in feature_sets.items():
        X = X_all[:, :, blocks, :].reshape(n, 5, -1)
        X = X - X.mean(1, keepdims=True)                   # within-state centring
        pick, chosen_alpha, train_hit = np.zeros(n, int), [], []
        for f in range(5):
            tr = np.array([k for k in range(n) if fold_of[k] != f]); te = np.array([k for k in range(n) if fold_of[k] == f])
            mu, sd = X[tr].reshape(-1, X.shape[-1]).mean(0), X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
            Xtr = ((X[tr] - mu) / sd).reshape(-1, X.shape[-1]); Xte = ((X[te] - mu) / sd).reshape(-1, X.shape[-1])
            task_rows = np.repeat([rows[k]["task_id"] for k in tr], 5)   # all five candidates of a state stay together
            inner = list(GroupKFold(n_splits=4).split(Xtr, groups=task_rows))
            model = RidgeCV(alphas=alphas, cv=inner).fit(Xtr, y[tr].reshape(-1))
            chosen_alpha.append(float(model.alpha_))
            pick[te] = model.predict(Xte).reshape(len(te), 5).argmax(1)
            ptr = model.predict(Xtr).reshape(len(tr), 5).argmax(1)
            train_hit.append(float(np.mean([y_rate[k, p] == y_rate[k].max() for k, p in zip(tr, ptr)])))
        results[name] = pick
        fits[name] = {"alpha_per_fold": chosen_alpha, "train_best_action_hit": train_hit,
                      "dims": int(X.shape[-1]), "picks_changed_vs_teacher_head": int((pick != teacher.argmax(1)).sum())}
    refs = references(data)
    groups = [row["task_id"] for row in rows]
    report = {"states": n, "fold_sizes": sizes, "probe_fit": fits, "success": {}, "minus_uniform": {}, "minus_teacher_head": {}}
    for reps_name, reps in (("0_7", ALL), ("6_7", HOLD)):
        r = D[:, :, list(reps)].mean(-1)
        scores = {**{k: r[np.arange(n), v] for k, v in results.items()}, **refs[reps_name]}
        report["success"][reps_name] = {k: 100 * float(v.mean()) for k, v in scores.items()}
        report["minus_uniform"][reps_name] = {k: cluster_boot(list(v - scores["Uniform"]), groups) for k, v in scores.items() if k != "Uniform"}
        report["minus_teacher_head"][reps_name] = {k: cluster_boot(list(v - scores["Teacher-trained head"]), groups)
                                                  for k, v in scores.items() if k != "Teacher-trained head"}
    (out / "probe.json").write_text(json.dumps(report, indent=1))
    for k in sorted(report["success"]["0_7"], key=lambda x: -report["success"]["0_7"][x]):
        u = report["minus_uniform"]["0_7"].get(k); t = report["minus_teacher_head"]["0_7"].get(k)
        fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
        print(f"{k:58s} {report['success']['0_7'][k]:6.2f}%  vs uniform {fmt(u)}  vs teacher {fmt(t)}")
    for k, v in fits.items():
        print(k, "alpha", [f"{a:.3g}" for a in v["alpha_per_fold"]], "train hit %.2f" % (sum(v["train_best_action_hit"]) / 5),
              "changed", v["picks_changed_vs_teacher_head"])


def power(data, out):
    """Between-state versus draw-noise variance of per-state differences, and states needed for power."""
    import numpy as np
    from scipy.stats import norm
    rows, D = data["rows"], data["draws"].numpy()
    n = len(rows)
    halves = (range(0, 8, 2), range(1, 8, 2))         # interleaved split-half draws
    def per_state(rule, reps):
        r = D[:, :, list(reps)].mean(-1)
        if rule == "Uniform": return r.mean(1)
        if rule == "Actor's first sample": return r[:, 0]
        chosen = data["others"][rule]
        return np.array([r[k, chosen[row["state_id"]]] for k, row in enumerate(rows)])
    comparisons = [("Piotr ARM", "Uniform"), ("Teacher-trained head", "Uniform"), ("Piotr ARM", "Teacher-trained head"),
                   ("Actor's first sample", "Uniform"), ("Kev-style step 600", "Uniform")]
    z = norm.ppf(0.975) + norm.ppf(0.80)
    report = {"states": n, "note": "two-sided alpha 0.05, power 0.80; state-level variance (most task groups hold one state)", "comparisons": {}}
    for a, b in comparisons:
        d8 = per_state(a, ALL) - per_state(b, ALL)
        d1, d2 = (per_state(a, h) - per_state(b, h) for h in halves)
        between = float(np.cov(d1, d2)[0, 1])            # covariance of independent halves = true between-state variance
        total8 = float(d8.var(ddof=1))
        noise8 = max(total8 - between, 0.0)              # draw noise at eight draws
        need = {}
        for delta in (0.02, 0.03, 0.05):
            for m in (4, 8, 16):
                var_m = max(between, 0) + noise8 * 8 / m
                need[f"delta_{int(delta*100)}pp_draws_{m}"] = int(np.ceil((z ** 2) * var_m / delta ** 2))
        report["comparisons"][f"{a} - {b}"] = {"mean_pp": 100 * float(d8.mean()), "sd_per_state_8draws": float(np.sqrt(total8)),
            "between_state_var": between, "draw_noise_var_8draws": noise8,
            "noise_share": noise8 / total8 if total8 else None, "states_needed": need}
    # outcome-selection reference versus uniform uses no saved picks: all states qualify
    (out / "power.json").write_text(json.dumps(report, indent=1))
    for k, v in report["comparisons"].items():
        s = v["states_needed"]
        print(f"{k:40s} mean {v['mean_pp']:+.2f} pp, sd {v['sd_per_state_8draws']:.3f}, draw-noise share {v['noise_share']:.2f} | "
              f"states for 3 pp: {s['delta_3pp_draws_4']}/{s['delta_3pp_draws_8']}/{s['delta_3pp_draws_16']} (4/8/16 draws); "
              f"2 pp @8: {s['delta_2pp_draws_8']}; 5 pp @8: {s['delta_5pp_draws_8']}")


def depth(data, out):
    import numpy as np
    rows, D = data["rows"], data["draws"].numpy()
    n = len(rows)
    r8 = D.mean(-1)
    all_refs = references(data)
    refs = {**all_refs["0_7"], "Outcome reference: best on draws 0-5": all_refs["6_7"]["Outcome reference: best on draws 0-5"]}
    band = lambda d: "0" if d == 0 else ("1-3" if d <= 3 else "4+")
    lineage = lambda row: {"branch124": "historical", "fresh64succ": "fresh"}.get(row["split"], "fresh" if row["split"] == "fresh64" else row["split"])
    table = defaultdict(lambda: defaultdict(list))
    for k, row in enumerate(rows):
        for key in ((lineage(row), band(row["depth"])), (lineage(row), "all"), ("all", band(row["depth"]))):
            for name in ("Actor's first sample", "Uniform", "Piotr ARM", "Teacher-trained head", "Outcome reference: best on draws 0-5"):
                table[key][name].append(refs[name][k])
    report = {"by_lineage_depth": {f"{a} | depth {b}": {"states": len(v["Uniform"]), **{m: 100 * float(np.mean(x)) for m, x in v.items()}}
                                   for (a, b), v in sorted(table.items())}}
    # depth-reweighted historical: weight historical states so their depth-band mix matches non-historical states
    hist = [k for k, row in enumerate(rows) if row["split"] == "branch124"]
    new = [k for k, row in enumerate(rows) if row["split"] != "branch124"]
    target = Counter(band(rows[k]["depth"]) for k in new); have = Counter(band(rows[k]["depth"]) for k in hist)
    w = np.array([target[band(rows[k]["depth"])] / len(new) / (have[band(rows[k]["depth"])] / len(hist)) for k in hist])
    report["historical_reweighted_to_new_depth_mix"] = {
        name: {"raw": 100 * float(refs[name][hist].mean()), "reweighted": 100 * float((refs[name][hist] * w).sum() / w.sum()),
               "new_states": 100 * float(refs[name][new].mean())}
        for name in ("Actor's first sample", "Uniform", "Piotr ARM", "Teacher-trained head")}
    report["depth_band_counts"] = {"historical": dict(have), "new": dict(target)}
    # action types: mean candidate success (draws 0-7) and how often each selector picks each type
    kinds = defaultdict(list); picked = defaultdict(Counter)
    for k, row in enumerate(rows):
        for i, kind in enumerate(row["action_kinds"]):
            kinds[kind].append(r8[k, i])
        for name in ("Piotr ARM", "Teacher-trained head", "Kev-style step 600"):
            picked[name][row["action_kinds"][data["others"][name][row["state_id"]]]] += 1
        picked["Actor's first sample"][row["action_kinds"][0]] += 1
        best = np.flatnonzero(r8[k] == r8[k].max())
        for i in best:
            picked["Best on draws 0-7 (hindsight)"][row["action_kinds"][i]] += 1 / len(best)
    report["action_kinds"] = {kind: {"candidates": len(v), "mean_success": 100 * float(np.mean(v))} for kind, v in
                              sorted(kinds.items(), key=lambda kv: -len(kv[1]))}
    report["picked_kinds"] = {name: {k: round(v, 2) for k, v in c.most_common()} for name, c in picked.items()}
    (out / "depth.json").write_text(json.dumps(report, indent=1))
    for k, v in report["by_lineage_depth"].items():
        first = v["Actor's first sample"]
        print(f"{k:28s} n={v['states']:3d}  first {first:.1f}  uniform {v['Uniform']:.1f}  "
              f"Piotr {v['Piotr ARM']:.1f}  teacher {v['Teacher-trained head']:.1f}  outcome-ref(6-7) {v['Outcome reference: best on draws 0-5']:.1f}")
    print("historical reweighted:", json.dumps(report["historical_reweighted_to_new_depth_mix"]))
    print("action kinds:", json.dumps(report["action_kinds"]))
    print("picked kinds:", json.dumps(report["picked_kinds"]))


def combine(data, out):
    """Combinations of existing selectors and an action-type prior learned inside training folds."""
    import numpy as np
    rows, D = data["rows"], data["draws"].numpy()
    n = len(rows)
    fold_of, sizes = folds_of(rows)
    groups = [row["task_id"] for row in rows]
    sid = [row["state_id"] for row in rows]
    pick = {name: np.array([data["others"][name][s] for s in sid]) for name in ("Piotr ARM", "Teacher-trained head", "Kev-style step 600")}
    pick["Actor's first sample"] = np.zeros(n, int)
    eq = lambda k, a, b: b in rows[k]["equivalent"][a]
    def vote(names, tie):
        out_pick = np.zeros(n, int)
        for k in range(n):
            cand = [pick[m][k] for m in names]
            support = [sum(eq(k, c, d) for d in cand) for c in cand]
            best = max(support)
            winners = [c for c, v in zip(cand, support) if v == best]
            out_pick[k] = pick[tie][k] if any(eq(k, w, pick[tie][k]) for w in winners) else winners[0]
        return out_pick
    combos = {"Vote: Piotr, teacher head, actor first (ties to Piotr)": vote(["Piotr ARM", "Teacher-trained head", "Actor's first sample"], "Piotr ARM"),
              "Vote: Piotr, teacher head, Kev-style 600 (ties to Piotr)": vote(["Piotr ARM", "Teacher-trained head", "Kev-style step 600"], "Piotr ARM")}
    # action-type prior: success rate of each action type on training folds (draws 0-5); pick the best type,
    # breaking ties by the teacher head's logit
    teacher = data["teacher_logits"].numpy()
    prior_pick, prior_teacher = np.zeros(n, int), np.zeros(n, int)
    fit = D[:, :, list(FIT)].mean(-1)
    for f in range(5):
        tr = [k for k in range(n) if fold_of[k] != f]
        stats = defaultdict(list)
        for k in tr:
            for i, kind in enumerate(rows[k]["action_kinds"]):
                stats[kind].append(fit[k, i] - fit[k].mean())   # within-state relative success
        prior = {kind: float(np.mean(v)) for kind, v in stats.items()}
        for k in (k for k in range(n) if fold_of[k] == f):
            score = np.array([prior.get(kind, 0.0) for kind in rows[k]["action_kinds"]])
            prior_pick[k] = int(np.lexsort((-teacher[k], -score))[0])
            # teacher head unless it picks a type with a clearly negative prior and a better type exists
            t = int(teacher[k].argmax())
            prior_teacher[k] = t if score[t] >= score.max() - 0.10 else int(np.argmax(score + 1e-3 * teacher[k]))
    combos["Action-type prior (training folds), teacher logit tie-break"] = prior_pick
    combos["Teacher head, overridden when its action type is >10 pp worse"] = prior_teacher
    report = {"states": n, "success": {}, "minus_uniform": {}, "minus_piotr": {}}
    for reps_name, reps in (("0_7", ALL), ("6_7", HOLD)):
        r = D[:, :, list(reps)].mean(-1)
        scores = {**{k: r[np.arange(n), v] for k, v in {**pick, **combos}.items()}, "Uniform": r.mean(1)}
        report["success"][reps_name] = {k: 100 * float(v.mean()) for k, v in scores.items()}
        report["minus_uniform"][reps_name] = {k: cluster_boot(list(v - scores["Uniform"]), groups) for k, v in scores.items() if k != "Uniform"}
        report["minus_piotr"][reps_name] = {k: cluster_boot(list(v - scores["Piotr ARM"]), groups) for k, v in scores.items() if k != "Piotr ARM"}
    # agreement analysis: are Piotr and the teacher head right on different states?
    r8 = D.mean(-1)
    agree = np.array([eq(k, pick["Piotr ARM"][k], pick["Teacher-trained head"][k]) for k in range(n)])
    report["piotr_teacher_agreement"] = {
        "agree_states": int(agree.sum()), "disagree_states": int((~agree).sum()),
        "agree": {"Piotr": 100 * float(r8[agree, pick["Piotr ARM"][agree]].mean()), "Uniform": 100 * float(r8[agree].mean())},
        "disagree": {"Piotr": 100 * float(r8[~agree, pick["Piotr ARM"][~agree]].mean()),
                     "Teacher head": 100 * float(r8[~agree, pick["Teacher-trained head"][~agree]].mean()),
                     "Uniform": 100 * float(r8[~agree].mean())}}
    (out / "combine.json").write_text(json.dumps(report, indent=1))
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    for k in sorted(report["success"]["0_7"], key=lambda x: -report["success"]["0_7"][x]):
        print(f"{k:62s} {report['success']['0_7'][k]:6.2f}%  vs uniform {fmt(report['minus_uniform']['0_7'].get(k))}  vs Piotr {fmt(report['minus_piotr']['0_7'].get(k))}")
    print(json.dumps(report["piotr_teacher_agreement"]))


def joint(data, out, steps=200, weights=(0.0, 0.3, 1.0, 3.0), lr=1e-4, teacher_batch=16, outcome_batch=8, seed=0):
    """Continue the teacher head on GPT-5.5 panels plus an outcome loss; weight chosen by nested CV."""
    import numpy as np
    import torch
    from sklearn.model_selection import GroupKFold
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, selection_loss
    torch.set_num_threads(16)
    saved = torch.load(R / "heads/pointwise/head.pt", map_location="cpu")
    config = HeadConfig.from_json(saved["config"])
    head_args = argparse.Namespace(layers=",".join(map(str, config.layers)), pools=",".join(config.pools))
    rows, D = data["rows"], data["draws"].numpy()
    n = len(rows)
    # outcome-state features in the head's layout: layers 18/27/36 x all pools
    full = data["features"].reshape(n, 5, 12, -1)
    keep = [data["layers"].index(l) * 3 + j for l in config.layers for j in range(3)]
    Oc = full[:, :, keep, :].reshape(n, 5, -1)
    Os = data["state"].reshape(n, len(data["layers"]), -1)[:, [data["layers"].index(l) for l in config.layers], :].reshape(n, -1)
    check = CandidateSetHead(config); check.load_state_dict(saved["state_dict"]); check.eval()
    with torch.no_grad():
        err = (check(Oc, Os)[0] - data["teacher_logits"]).abs().max().item()
    assert err < 1e-3, f"rebuilt outcome-state inputs do not reproduce the teacher head ({err})"
    print("input layout check passed, max logit difference", err, flush=True)
    fit = torch.tensor(equivalent_rates(data, FIT), dtype=torch.float32)
    informative = (fit.max(1).values > fit.min(1).values)
    mask = torch.ones(max(teacher_batch, outcome_batch), 5, dtype=torch.bool)
    teach = load(head_args, torch, R / "panels/panels.jsonl", R / "features-piotr")[0]["train"]
    Tc, Ts, Tt = teach["candidates"], teach["state"], teach["target"].float()
    print("teacher panels", len(Tt), flush=True)

    def train(idx, w):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
        idx = [k for k in idx if informative[k]]
        for _ in range(steps):
            t = torch.from_numpy(rng.integers(0, len(Tt), teacher_batch))
            logits, _ = head(Tc[t], Ts[t])
            loss = selection_loss(logits, Tt[t], mask[:teacher_batch])
            if w > 0:
                o = torch.tensor(rng.choice(idx, outcome_batch))
                ol, _ = head(Oc[o], Os[o])
                target = fit[o] / fit[o].sum(1, keepdim=True)
                loss = loss + w * -(target * torch.log_softmax(ol, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        return head

    def picks(head, idx):
        with torch.no_grad():
            return head(Oc[idx], Os[idx])[0].argmax(1).numpy()

    fold_of, sizes = folds_of(rows)
    groups = [row["task_id"] for row in rows]
    chosen_w, outer = [], {w: np.zeros(n, int) for w in weights}
    nested = np.zeros(n, int)
    for f in range(5):
        tr = np.array([k for k in range(n) if fold_of[k] != f]); te = np.array([k for k in range(n) if fold_of[k] == f])
        inner_score = {w: [] for w in weights}
        for itr, ite in GroupKFold(n_splits=4).split(tr, groups=[groups[k] for k in tr]):
            a, b = tr[itr], tr[ite]
            for w in weights:
                p = picks(train(a, w), b)
                inner_score[w].append(float(fit[b, p].mean()))   # inner held-out scored on draws 0-5 only
        best = max(weights, key=lambda w: (np.mean(inner_score[w]), -w))
        chosen_w.append({"fold": f, "chosen": best, "inner_success_draws_0_5": {str(w): float(np.mean(v)) for w, v in inner_score.items()}})
        for w in weights:
            outer[w][te] = picks(train(tr, w), te)
        nested[te] = outer[best][te]
        print("fold", f, "chosen weight", best, {w: round(np.mean(v) * 100, 2) for w, v in inner_score.items()}, flush=True)
    report = {"states": n, "steps": steps, "lr": lr, "teacher_batch": teacher_batch, "outcome_batch": outcome_batch,
              "weights": list(weights), "chosen": chosen_w, "success": {}, "minus_uniform": {}, "minus_teacher_head": {}}
    teacher_pick = np.array([data["others"]["Teacher-trained head"][row["state_id"]] for row in rows])
    for reps_name, reps in (("0_7", ALL), ("6_7", HOLD)):
        r = D[:, :, list(reps)].mean(-1)
        scores = {"Teacher + outcome loss, weight by nested CV": r[np.arange(n), nested],
                  **{f"Teacher + outcome loss, weight {w}": r[np.arange(n), outer[w]] for w in weights},
                  "Teacher-trained head": r[np.arange(n), teacher_pick], "Uniform": r.mean(1)}
        report["success"][reps_name] = {k: 100 * float(v.mean()) for k, v in scores.items()}
        report["minus_uniform"][reps_name] = {k: cluster_boot(list(v - scores["Uniform"]), groups) for k, v in scores.items() if k != "Uniform"}
        report["minus_teacher_head"][reps_name] = {k: cluster_boot(list(v - scores["Teacher-trained head"]), groups)
                                                  for k, v in scores.items() if k != "Teacher-trained head"}
    report["picks_changed_vs_teacher_head"] = {str(w): int((outer[w] != teacher_pick).sum()) for w in weights}
    (out / "joint.json").write_text(json.dumps(report, indent=1))
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    for k in sorted(report["success"]["0_7"], key=lambda x: -report["success"]["0_7"][x]):
        print(f"{k:48s} {report['success']['0_7'][k]:6.2f}%  vs uniform {fmt(report['minus_uniform']['0_7'].get(k))}  vs teacher {fmt(report['minus_teacher_head']['0_7'].get(k))}")
    print("picks changed vs teacher head", report["picks_changed_vs_teacher_head"])


def train_scratch(config, C, S, T, seed, epochs=20, lr=3e-4, batch=32):
    """One shared recipe for every label type: fresh pointwise head, standardizers fit on its training data."""
    import torch
    from openwebrl.selection_head import CandidateSetHead, selection_loss
    torch.manual_seed(seed)
    head = CandidateSetHead(config)
    head.fit_standardizers(C, S)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * ((len(T) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    gen = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(epochs):
        order = torch.randperm(len(T), generator=gen)
        for i in range(0, len(T), batch):
            b = order[i:i + batch]
            logits, _ = head(C[b], S[b])
            loss = selection_loss(logits, T[b], mask[:len(b)])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval()
    return head


def head_inputs(data, config, idx=None):
    """Outcome-state features in the teacher head's layout (layers 18/27/36 x all pools)."""
    n = len(data["rows"])
    full = data["features"].reshape(n, 5, 12, -1)
    keep = [data["layers"].index(l) * 3 + j for l in config.layers for j in range(3)]
    C = full[:, :, keep, :].reshape(n, 5, -1)
    S = data["state"].reshape(n, len(data["layers"]), -1)[:, [data["layers"].index(l) for l in config.layers], :].reshape(n, -1)
    return C, S


def labels(data, out, seeds=(0, 1)):
    """A: teacher-label learning curve and matched-scale comparison; B: pre- versus post-action supervision."""
    import numpy as np
    import sqlite3
    import torch
    from openwebrl.selection_head import HeadConfig, equivalent, parse_calls
    from adapt_selection_head_outcomes import ORIG
    torch.set_num_threads(16)
    rng_master = np.random.default_rng(0)
    saved = torch.load(R / "heads/pointwise/head.pt", map_location="cpu")
    config = HeadConfig.from_json(saved["config"])
    head_args = argparse.Namespace(layers=",".join(map(str, config.layers)), pools=",".join(config.pools))
    report = {"recipe": "fresh pointwise head; AdamW lr 3e-4 one-cycle, wd 0.05, 20 epochs, batch 32; seeds 0 and 1 (logits averaged)"}

    # ---- teacher panels, grouped by state and task group
    tsplits = load(head_args, torch, R / "panels/panels.jsonl", R / "features-piotr")[0]
    teach, tval = tsplits["train"], tsplits["validation"]
    TC, TS, TT = teach["candidates"], teach["state"], teach["target"].float()
    by_group = defaultdict(lambda: defaultdict(list))
    for k, row in enumerate(teach["rows"]):
        by_group[row["task_group"]][row["base_state_id"]].append(k)
    groups_sorted = sorted(by_group, key=lambda g: hashlib.sha256(g.encode()).hexdigest())

    def teacher_subset(n_states, seed, one_panel=False):
        """Whole task groups in a seeded order until n_states states are covered."""
        order = list(groups_sorted); random.Random(seed).shuffle(order)
        idx, states = [], 0
        for g in order:
            for st, panels in by_group[g].items():
                if states >= n_states: break
                idx += panels[:1] if one_panel else panels; states += 1
            if states >= n_states: break
        return idx

    def val_agreement(head):
        with torch.no_grad():
            lg = predict(head, tval, "cpu", torch)
        return float((tval["target"].gather(1, lg.argmax(1, keepdim=True)) > 0).float().mean())

    # ---- A: learning curve on the 138 eight-draw states (all draws), teacher states are disjoint by construction
    rows, D = data["rows"], data["draws"].numpy()
    n = len(rows)
    C8, S8 = head_inputs(data, config)
    r8 = D.mean(-1)
    groups8 = [row["task_id"] for row in rows]
    overlap = np.array([bool(row.get("piotr_task_overlap")) for row in rows])
    curve = {}
    parts = os.environ.get("LABELS_PARTS", "A1,A2,B").split(",")
    sizes = tuple(int(x) for x in os.environ.get("LABELS_SIZES", "110,300,1000,2692").split(","))
    for n_states in (sizes if "A1" in parts else ()):
        for one_panel in ((False, True) if n_states <= 300 else (False,)):
            total, agree, panels_used = 0, [], 0
            for seed in seeds:
                idx = teacher_subset(n_states, seed, one_panel)
                panels_used = len(idx)
                h = train_scratch(config, TC[idx], TS[idx], TT[idx], seed)
                agree.append(val_agreement(h))
                with torch.no_grad():
                    total = total + h(C8, S8)[0]
            pick = total.argmax(1).numpy()
            name = f"GPT-5.5 labels, {n_states} states" + (", one panel each" if one_panel else "")
            sc = r8[np.arange(n), pick]
            curve[name] = {"panels": panels_used, "success": 100 * float(sc.mean()),
                           "success_without_overlap_tasks": 100 * float(sc[~overlap].mean()),
                           "minus_uniform": cluster_boot(list(sc - r8.mean(1)), groups8),
                           "held_out_teacher_agreement": float(np.mean(agree))}
            print(name, json.dumps(curve[name]), flush=True)
    report["A_teacher_curve_138"] = curve
    if "B" not in parts and "A2" not in parts:
        (out / "labels.json").write_text(json.dumps(report, indent=1)); return
    report["A_reference_138"] = {"uniform": 100 * float(r8.mean()), "states": n, "overlap_states": int(overlap.sum())}

    # ---- A2 + B share the same student; B uses the original 124 states with Luna before/after labels
    cell = outcomes()
    (hist,) = load(head_args, torch, R / "branch124/panels.jsonl", R / "features-branch124")[0].values()
    HC, HS, hrows = hist["candidates"], hist["state"], hist["rows"]
    m = len(hrows)
    db = sqlite3.connect(f"file:{ORIG}/state.sqlite3?mode=ro", uri=True)
    luna = defaultdict(lambda: {"before": [], "after": [], "after_draw0": [], "lookahead": []})
    for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
        rec = json.loads(raw)
        if rec.get("mode") == "teacher":
            for j in json.load(open(rec["judgments_path"])):
                if j["valid"]:
                    luna[j["state"]][j["condition"]].append(j["candidate"])
                    if j["condition"] == "after" and j["execution_repeat"] == 0:   # same evidence draw as lookahead
                        luna[j["state"]]["after_draw0"].append(j["candidate"])
    for f in (R / "luna-lookahead-noanswer-20261011" / "judgments").glob("*.json"):
        for j in json.load(open(f)):
            if j["valid"]:
                luna[j["state"]]["lookahead"].append(j["candidate"])
    def soft(choices, eqv):
        t = np.zeros(5)
        for c in choices:
            g = eqv[c]; t[g] += 1 / len(g)
        return t / t.sum() if t.sum() else None
    eqvs = [[[j for j in range(5) if j == i or equivalent(pc[i], pc[j])] for i in range(5)]
            for pc in ([parse_calls(c) for c in row["candidates"]] for row in hrows)]
    def outcome_target(k, reps):
        s = hrows[k]["state_id"]
        raw = np.array([np.mean([cell[s][i, r] for r in reps if (i, r) in cell[s]]) for i in range(5)])
        pooled = np.array([raw[g].mean() for g in eqvs[k]])
        return pooled / pooled.sum() if pooled.max() > pooled.min() else None
    label_sets = {
        "Luna before (4 judgments)": [soft(luna[r["state_id"]]["before"], eqvs[k]) for k, r in enumerate(hrows)],
        "Luna after (10 judgments, evidence from draws 0-2)": [soft(luna[r["state_id"]]["after"], eqvs[k]) for k, r in enumerate(hrows)],
        "Outcomes on draws 0-2 (same executions as Luna after)": [outcome_target(k, range(3)) for k in range(m)],
        "Luna after (4 judgments, draw-0 evidence)": [soft(luna[r["state_id"]]["after_draw0"], eqvs[k]) for k, r in enumerate(hrows)],
        "Luna lookahead, answers hidden (4 judgments, draw 0)": [soft(luna[r["state_id"]]["lookahead"], eqvs[k]) for k, r in enumerate(hrows)],
    }
    sol_dir = R / "sol-labels-20261011" / "judgments"
    if sol_dir.exists():
        sol = {f.stem: [j["candidate"] for j in json.load(open(f)) if j["valid"]] for f in sol_dir.glob("*.json")}
        label_sets["Sol before (4 judgments)"] = [soft(sol.get(r["state_id"], []), eqvs[k]) if sol.get(r["state_id"]) else None
                                                 for k, r in enumerate(hrows)]
    only = os.environ.get("LABELS_ONLY")
    if only:
        label_sets = {k: v for k, v in label_sets.items() if k.startswith(only)}
    # held-out score: every saved draw of the state (labels of held-out states never enter training)
    hr = np.array([[np.mean([cell[r["state_id"]][i, rep] for rep in range(32) if (i, rep) in cell[r["state_id"]]]) for i in range(5)]
                   for r in hrows])
    hgroups = [r["task_id"] for r in hrows]
    by_task = defaultdict(list)
    for k, r in enumerate(hrows): by_task[r["task_id"]].append(k)
    hfold, sizes = {}, [0] * 5
    for t in sorted(by_task, key=lambda t: hashlib.sha256(t.encode()).hexdigest()):
        f = min(range(5), key=lambda j: (sizes[j], j))
        for k in by_task[t]: hfold[k] = f
        sizes[f] += len(by_task[t])
    # new collections (pilot, fresh): train on all historical labels, test on these
    new = [k for k, row in enumerate(rows) if row["split"] != "branch124"]
    Cn, Sn = C8[new], S8[new]; rn = r8[new]; gn = [groups8[k] for k in new]
    B = {"cv_124": {}, "train_124_test_new": {}, "fold_sizes": sizes, "labelled_states": {}}
    extra = [] if os.environ.get("LABELS_ONLY") else [("GPT-5.5 labels, matched states (other tasks)", None)]
    for name, targets in list(label_sets.items()) + extra:
        pick_cv = np.zeros(m, int); total_new = 0
        for f in range(5):
            tr = [k for k in range(m) if hfold[k] != f]; te = [k for k in range(m) if hfold[k] == f]
            total = 0
            for seed in seeds:
                if targets is None:   # same number of training states, drawn from the teacher pool
                    idx = teacher_subset(len(tr), 100 + seed); h = train_scratch(config, TC[idx], TS[idx], TT[idx], seed)
                else:
                    use = [k for k in tr if targets[k] is not None]
                    h = train_scratch(config, HC[use], HS[use], torch.tensor(np.stack([targets[k] for k in use]), dtype=torch.float32), seed)
                with torch.no_grad():
                    total = total + h(HC[te], HS[te])[0]
            pick_cv[te] = total.argmax(1).numpy()
        for seed in seeds:
            if targets is None:
                idx = teacher_subset(m, 200 + seed); h = train_scratch(config, TC[idx], TS[idx], TT[idx], seed)
            else:
                use = [k for k in range(m) if targets[k] is not None]
                h = train_scratch(config, HC[use], HS[use], torch.tensor(np.stack([targets[k] for k in use]), dtype=torch.float32), seed)
            with torch.no_grad():
                total_new = total_new + h(Cn, Sn)[0]
        pick_new = total_new.argmax(1).numpy()
        B["labelled_states"][name] = m if targets is None else int(sum(t is not None for t in targets))
        sc = hr[np.arange(m), pick_cv]
        B["cv_124"][name] = {"success": 100 * float(sc.mean()), "minus_uniform": cluster_boot(list(sc - hr.mean(1)), hgroups)}
        sn = rn[np.arange(len(new)), pick_new]
        B["train_124_test_new"][name] = {"success": 100 * float(sn.mean()), "minus_uniform": cluster_boot(list(sn - rn.mean(1)), gn)}
        print(name, json.dumps(B["cv_124"][name]), json.dumps(B["train_124_test_new"][name]), flush=True)
    B["cv_124"]["Uniform"] = {"success": 100 * float(hr.mean())}
    B["train_124_test_new"]["Uniform"] = {"success": 100 * float(rn.mean()), "states": len(new)}
    # A2: the 138-state outcome CV with this recipe, versus the same number of teacher states
    fold_of, _ = folds_of(rows)
    fit = equivalent_rates(data, FIT)
    A2 = {}
    for name in ("Outcome labels, draws 0-5", "GPT-5.5 labels, matched states (other tasks)"):
        pick = np.zeros(n, int)
        for f in range(5):
            tr = [k for k in range(n) if fold_of[k] != f]; te = [k for k in range(n) if fold_of[k] == f]
            total = 0
            for seed in seeds:
                if name.startswith("Outcome"):
                    use = [k for k in tr if fit[k].max() > fit[k].min()]
                    T = torch.tensor(fit[use] / fit[use].sum(1, keepdims=True), dtype=torch.float32)
                    h = train_scratch(config, C8[use], S8[use], T, seed)
                else:
                    idx = teacher_subset(len(tr), 300 + seed); h = train_scratch(config, TC[idx], TS[idx], TT[idx], seed)
                with torch.no_grad():
                    total = total + h(C8[te], S8[te])[0]
            pick[te] = total.argmax(1).numpy()
        sc = r8[np.arange(n), pick]
        A2[name] = {"success": 100 * float(sc.mean()), "minus_uniform": cluster_boot(list(sc - r8.mean(1)), groups8)}
        print("A2", name, json.dumps(A2[name]), flush=True)
    report["A2_matched_cv_138"] = A2
    report["B_pre_post"] = B
    (out / ("labels.json" if parts == ["A1", "A2", "B"] else f"labels-{'-'.join(parts)}.json")).write_text(json.dumps(report, indent=1))


def teachers(data, out):
    """Teachers and selectors as pickers, scored on held-out draws 3-7 (no teacher evidence touched them)."""
    import numpy as np
    import sqlite3
    from openwebrl.selection_head import equivalent, parse_calls
    from adapt_selection_head_outcomes import ORIG
    LUNA = R / "luna-labels-20261011"; SOL = R / "sol-labels-20261011"
    cell = outcomes()
    picks = defaultdict(lambda: defaultdict(list))          # name -> state -> list of picked candidates
    def add_luna(root):
        db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") != "teacher": continue
            for j in json.load(open(r["judgments_path"])):
                if j["valid"]:
                    picks["Luna before" if j["condition"] == "before" else "Luna after (evidence draws 0-2)"][j["state"]].append(j["candidate"])
                    if j["condition"] == "after" and j["execution_repeat"] == 0:   # same layout as Sol after
                        picks["Luna after (draw-0 evidence)"][j["state"]].append(j["candidate"])
    add_luna(ORIG); add_luna(LUNA)
    for f in (SOL / "judgments").glob("*.json"):
        for j in json.load(open(f)):
            if j["valid"]: picks["Sol before"][j["state"]].append(j["candidate"])
    for name, d in (("Luna lookahead, answers hidden", "luna-lookahead-noanswer-20261011"), ("Luna lookahead, answers shown", "luna-lookahead-labels-20261011")):
        for f in (R / d / "judgments").glob("*.json"):
            for j in json.load(open(f)):
                if j["valid"]: picks[name][j["state"]].append(j["candidate"])
    for f in (R / "sol-after-labels-20261011" / "judgments").glob("*.json"):
        for j in json.load(open(f)):
            if j["valid"]: picks["Sol after (draw-0 evidence)"][j["state"]].append(j["candidate"])
    # saved selector picks (Piotr and actor-based) where scored
    for name, chosen in data["others"].items():
        for st, c in chosen.items(): picks[name][st].append(c)
    hist_piotr = {json.loads(l)["state_id"]: json.loads(l)["selected_index"] for l in open(R / "piotr_arm.jsonl")}
    for st, c in hist_piotr.items(): picks["Piotr ARM"].setdefault(st, [c])
    # states: every labelled state with draws 3-7 for all five candidates
    states = sorted(s for s in set(picks["Luna before"]) | set(picks["Sol before"])
                    if all(all((i, r) in cell[s] for r in range(3, 8)) for i in range(5)))
    task = {}
    for root in (ORIG, LUNA):
        for item in read(root / "plan.json")["schedule"]:
            task[item["state"]] = item["task_id"]
    rate = lambda s, reps: np.array([np.mean([cell[s][i, r] for r in reps]) for i in range(5)])
    hold = {s: rate(s, range(3, 8)) for s in states}
    fit = {s: rate(s, range(3)) for s in states}
    score = {}
    for name, P in picks.items():
        score[name] = {s: float(np.mean([hold[s][c] for c in P[s]])) for s in states if P.get(s)}
    score["Uniform"] = {s: float(hold[s].mean()) for s in states}
    score["Actor's first sample"] = {s: float(hold[s][0]) for s in states}
    score["Outcome reference: best on draws 0-2"] = {s: float(hold[s][fit[s] == fit[s].max()].mean()) for s in states}
    score["Hindsight best on draws 3-7 (optimistic)"] = {s: float(hold[s].max()) for s in states}
    def paired(a, b, subset):
        st = [s for s in subset if s in score[a] and s in score[b]]
        if not st:
            return (None, None, None), 0
        return cluster_boot([score[a][s] - score[b][s] for s in st], [task[s] for s in st]), len(st)
    names = ["Luna lookahead, answers shown", "Luna lookahead, answers hidden", "Sol after (draw-0 evidence)", "Luna after (draw-0 evidence)", "Sol before", "Luna before", "Luna after (evidence draws 0-2)", "Piotr ARM", "Teacher-trained head",
             "Kev-style step 600", "Actor's first sample", "Outcome reference: best on draws 0-5", "Outcome reference: best on draws 0-2",
             "Uniform", "Hindsight best on draws 3-7 (optimistic)"]
    lineage = {s: ("historical" if s in hist_piotr else "new") for s in states}
    report = {"states": len(states), "by_lineage": dict(Counter(lineage.values())), "scored": {}, "paired": {}}
    for subset_name, subset in (("all", states), ("new", [s for s in states if lineage[s] == "new"]),
                                ("historical", [s for s in states if lineage[s] == "historical"])):
        report["scored"][subset_name] = {n: {"states": sum(s in score[n] for s in subset),
                                             "success": 100 * float(np.mean([score[n][s] for s in subset if s in score[n]]))}
                                         for n in names if n in score and any(s in score[n] for s in subset)}
        report["paired"][subset_name] = {}
        for a, b in [("Luna lookahead, answers hidden", "Uniform"), ("Luna lookahead, answers shown", "Uniform"),
                     ("Luna lookahead, answers hidden", "Luna after (draw-0 evidence)"), ("Luna lookahead, answers hidden", "Luna before"),
                     ("Luna lookahead, answers shown", "Luna lookahead, answers hidden"), ("Luna lookahead, answers shown", "Luna before"),
                     ("Outcome reference: best on draws 0-2", "Luna lookahead, answers shown"),
                     ("Sol after (draw-0 evidence)", "Uniform"), ("Luna after (draw-0 evidence)", "Uniform"),
                     ("Sol after (draw-0 evidence)", "Sol before"), ("Luna after (draw-0 evidence)", "Luna before"),
                     ("Sol after (draw-0 evidence)", "Luna after (draw-0 evidence)"),
                     ("Outcome reference: best on draws 0-2", "Sol after (draw-0 evidence)"),
                     ("Sol before", "Uniform"), ("Luna before", "Uniform"), ("Luna after (evidence draws 0-2)", "Uniform"),
                     ("Sol before", "Luna before"), ("Luna after (evidence draws 0-2)", "Luna before"),
                     ("Sol before", "Luna after (evidence draws 0-2)"), ("Sol before", "Piotr ARM"), ("Piotr ARM", "Uniform"),
                     ("Outcome reference: best on draws 0-2", "Uniform"), ("Outcome reference: best on draws 0-2", "Sol before")]:
            if a in score and b in score:
                (m, lo, hi), k = paired(a, b, subset)
                if k: report["paired"][subset_name][f"{a} - {b}"] = {"states": k, "pp": [m, lo, hi]}
    # teacher agreement with each other (action-equivalent majority picks)
    (out / "teachers.json").write_text(json.dumps(report, indent=1))
    for sub in ("all", "new", "historical"):
        print(f"== {sub}")
        for n, v in sorted(report["scored"][sub].items(), key=lambda kv: -kv[1]["success"]):
            print(f"  {n:45s} n={v['states']:3d} {v['success']:6.2f}%")
        for k, v in report["paired"][sub].items():
            print(f"  {k:70s} n={v['states']:3d} {v['pp'][0]:+.2f} [{v['pp'][1]:+.2f}, {v['pp'][2]:+.2f}]")


def stage_a(out, root=None):
    """Lookahead stage (a): matched Luna before / immediate result / 3-step lookahead (answers hidden) on fresh
    Piotr-matched states, every pick scored on held-out draws 3-7; task-group bootstrap."""
    import numpy as np
    import sqlite3
    root = Path(root or R / "luna-stagea-matched-20261012")
    plan = read(root / "plan.json")
    cell, task = defaultdict(dict), {}
    for shard in sorted({e["root"] for e in plan["states"]}):
        db = sqlite3.connect(f"file:{shard}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") == "branch" and r.get("released"):
                cell[r["state"]][(r["candidate"], r["repeat"])] = bool(r["valid"]) and (r.get("reward") or 0) >= 1
    for e in plan["states"]:
        ref = read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")
        task[e["state"]] = read(ref["path"])["task_id"]
    names = {"before": "Luna before", "after": "Luna immediate result (draw 0)", "after_lookahead": "Luna lookahead, answers hidden"}
    picks = defaultdict(lambda: defaultdict(list))
    for f in (root / "judgments").glob("*.json"):
        for j in json.loads(f.read_text()):
            if j["valid"]:
                picks[names[j["condition"]]][j["state"]].append(j["candidate"])
    # matched set: states with all three conditions labelled
    states = sorted(set.intersection(*(set(picks[n]) for n in names.values())))
    rate = lambda s, reps: np.array([np.mean([cell[s][i, r] for r in reps]) for i in range(5)])
    hold = {s: rate(s, range(3, 8)) for s in states}
    fit = {s: rate(s, range(3)) for s in states}
    score = {n: {s: float(np.mean([hold[s][c] for c in picks[n][s]])) for s in states} for n in names.values()}
    score["Uniform"] = {s: float(hold[s].mean()) for s in states}
    score["Actor's first sample"] = {s: float(hold[s][0]) for s in states}
    score["Outcome reference: best on draws 0-2"] = {s: float(hold[s][fit[s] == fit[s].max()].mean()) for s in states}
    score["Hindsight best on draws 3-7 (optimistic)"] = {s: float(hold[s].max()) for s in states}
    groups = [task[s] for s in states]
    report = dict(states=len(states), task_groups=len(set(groups)),
                  judgments={n: sum(len(picks[n][s]) for s in states) for n in names.values()},
                  success={n: 100 * float(np.mean(list(v.values()))) for n, v in score.items()}, paired={})
    L, I, B = names["after_lookahead"], names["after"], names["before"]
    for a, b in [(L, I), (L, B), (I, B), (L, "Uniform"), (I, "Uniform"), (B, "Uniform"),
                 ("Outcome reference: best on draws 0-2", L), ("Outcome reference: best on draws 0-2", "Uniform")]:
        report["paired"][f"{a} - {b}"] = cluster_boot([score[a][s] - score[b][s] for s in states], groups)
    (out / "stage_a.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


LOOKAHEAD_CONDITIONS = {"before": "Before execution", "immediate": "Immediate result", "k1": "Lookahead K=1",
                        "k2": "Lookahead K=2", "k3": "Lookahead K=3", "k5": "Lookahead K=5"}


def lookahead_scores(base, fit=range(3), hold=range(3, 8)):
    """Per-state held-out (default draws 3-7) success of each consistent-format condition under `base`, plus task ids."""
    import numpy as np
    import sqlite3
    conds = {c: n for c, n in LOOKAHEAD_CONDITIONS.items() if (base / c / "judgments").is_dir()}
    plan = read(base / "k3" / "plan.json")
    cell = outcomes()                                 # original campaigns (first lookahead study states)
    for shard in {e["root"] for e in plan["states"] if (Path(e["root"]) / "state.sqlite3").exists()}:
        db = sqlite3.connect(f"file:{shard}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") == "branch" and r.get("released"):
                cell[r["state"]].setdefault((r["candidate"], r["repeat"]), bool(r["valid"]) and (r.get("reward") or 0) >= 1)
    task = {e["state"]: read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])["task_id"] for e in plan["states"]}
    picks = {}
    for c in conds:
        picks[c] = defaultdict(list)
        for f in (base / c / "judgments").glob("*.json"):
            for j in json.loads(f.read_text()):
                if j["valid"]:
                    picks[c][j["state"]].append(j["candidate"])
    states = sorted(s for s in set.intersection(*(set(v) for v in picks.values()))
                    if all((i, k) in cell[s] for i in range(5) for k in hold))
    fit_draws, hold = fit, {s: np.array([np.mean([cell[s][i, k] for k in hold]) for i in range(5)]) for s in states}
    fit = {s: np.array([np.mean([cell[s][i, k] for k in fit_draws]) for i in range(5)]) for s in states if all((i, k) in cell[s] for i in range(5) for k in fit_draws)}
    score = {conds[c]: {s: float(np.mean([hold[s][x] for x in picks[c][s]])) for s in states} for c in conds}
    score["Uniform"] = {s: float(hold[s].mean()) for s in states}
    score["Outcome reference: best on observed draws"] = {s: float(hold[s][fit[s] == fit[s].max()].mean()) for s in fit}
    return states, score, {s: task[s] for s in states}


def lookahead_report(states, score, task):
    import numpy as np
    groups = lambda st: [task[s] for s in st]
    report = dict(states=len(states), task_groups=len({task[s] for s in states}),
                  success={n: {"states": len(v), "pct": 100 * float(np.mean(list(v.values())))} for n, v in score.items()},
                  minus_uniform={}, minus_k3={}, minus_immediate={})
    for n in score:
        st = [s for s in states if s in score[n]]
        if n != "Uniform":
            report["minus_uniform"][n] = cluster_boot([score[n][s] - score["Uniform"][s] for s in st], groups(st))
        if n not in ("Lookahead K=3", "Uniform"):
            report["minus_k3"][n] = cluster_boot([score[n][s] - score["Lookahead K=3"][s] for s in st], groups(st))
        if n.startswith("Lookahead"):
            report["minus_immediate"][n] = cluster_boot([score[n][s] - score["Immediate result"][s] for s in st], groups(st))
    return report


def lookahead_v2(out, bases=None, name="lookahead_v2", fit=range(3), hold=range(3, 8)):
    """Consistent-format lookahead (before, immediate, K = 1/2/3/5 where run), scored on draws 3-7.

    One report per base, and with several bases a pooled report over the conditions they share (states are
    distinct across bases; the bootstrap clusters by task id, so a task in both collections is one cluster)."""
    bases = [Path(b) for b in (bases or [R / "lookahead-v2-20261013"])]
    parts = {b.name: lookahead_scores(b, fit, hold) for b in bases}
    report = {k: lookahead_report(*v) for k, v in parts.items()}
    report["draws"] = dict(outcome_reference_fit=list(fit), scored=list(hold))
    if len(bases) > 1:
        shared = set.intersection(*(set(sc) for _, sc, _ in parts.values()))
        states, score, task = [], defaultdict(dict), {}
        for st, sc, tk in parts.values():
            assert not set(states) & set(st), "a state appears in two collections"
            states += st; task.update(tk)
            for n in shared:
                score[n].update(sc[n])
        report["pooled"] = lookahead_report(states, dict(score), task)
        report["task_overlap"] = {f"{a} & {b}": len(set(parts[a][2].values()) & set(parts[b][2].values()))
                                  for i, a in enumerate(parts) for b in list(parts)[i + 1:]}
    (out / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "probe", "power", "depth", "combine", "joint", "labels", "teachers", "stage_a", "lookahead_v2"])
    ap.add_argument("--out", type=Path, default=R / "outcome-experiments-20261010")
    ap.add_argument("--bases", nargs="*", help="lookahead_v2: label roots (default: the 315-state v2 rerun)")
    ap.add_argument("--name", default="lookahead_v2", help="lookahead_v2: report file stem")
    ap.add_argument("--fit-draws", type=int, default=3, help="lookahead_v2: outcome reference picks on draws 0..N-1; scoring uses draws N-7")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "stage_a":
        stage_a(args.out); return
    if args.command == "lookahead_v2":
        lookahead_v2(args.out, args.bases, args.name, range(args.fit_draws), range(args.fit_draws, 8)); return
    import torch
    cache = args.out / "dataset.pt"
    if args.command == "build":
        build(cache); return
    data = torch.load(cache, map_location="cpu", weights_only=False)
    {"probe": probe, "power": power, "depth": depth, "combine": combine, "joint": joint, "labels": labels, "teachers": teachers}[args.command](data, args.out)


if __name__ == "__main__":
    main()
