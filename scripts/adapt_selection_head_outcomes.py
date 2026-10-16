#!/usr/bin/env python3
"""Adapt the teacher-trained selection head to continuation outcomes (CPU; features already cached).

Pre-registered design (2026-10-10):
* States: branching states with saved features and draws 0-7 for all five candidates.
* Labels: each candidate's success rate on draws 0-5 (invalid = 0), pooled over action-equivalent
  candidates. Draws 6-7 are never used for training or model choice.
* Evaluation: 5-fold cross-validation grouped by task id; every state is scored once, out of fold,
  by the chosen candidate's success on draws 6-7. All references are scored on the same draws.
* Primary model: warm start from heads/pointwise/head.pt; soft cross-entropy to rates / sum(rates)
  over states whose candidates do not all tie; AdamW lr 1e-4, weight decay 0.05, 20 epochs,
  batch 16, seeds 0 and 1 (reported separately and as a logit average).
* Secondary: a randomly initialized head with the same recipe; a best-action (ties spread) target;
  the collections' own train / validation_seen_task / validation_unseen_task roles.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_selection_head import load, predict  # noqa: E402

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")
P = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation")
ORIG = P / "outcome-reward-20260927/critic-comparison-20261004/continuation-branches-preparation-20261006/overnight-v4-5x3"
B = P / "branch-selector-scaleup-20261008"
LEDGERS = [ORIG, ORIG / "extensions/finish149-20261007/extra-two-preparation/run",
           B / "inventory-topup-preparation-20261010/controller-v1/run", B / "pilot32-v1",
           B / "pilot32-v1/extra24-preparation", B / "expansion64-20261010-v1",
           B / "next-fresh-batch-preparation-20261010/staged-campaign-v1"]
ROLE_ROOTS = [B / "pilot32-v1", B / "expansion64-20261010-v1", B / "next-fresh-batch-preparation-20261010/staged-campaign-v1"]
# (panels dir, features dir, selector-picks dir) per collection
SOURCES = [(R / "branch124", R / "features-branch124", None)] + [
    (R / "branch-new" / L, R / "score-new-354319" / L / "features", R / "score-new-354319" / L)
    for L in ("pilot32", "fresh64", "fresh64succ")]
if os.environ.get("SELECTION_SOURCES"):   # manifest written by selection_final_rescore.sbatch
    SOURCES = [(R / "branch124", R / "features-branch124", None)] + [
        (Path(x["panels"]), Path(x["features"]), Path(x["picks"]))
        for x in json.loads(Path(os.environ["SELECTION_SOURCES"]).read_text())]
FIT, HOLD = range(6), range(6, 8)


def outcomes():
    """{state: {(candidate, repeat): success}} from released records; invalid = 0."""
    cell = defaultdict(dict)
    for root in LEDGERS:
        db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") != "branch" or not r.get("released"):
                continue
            v = bool(r["valid"]) and (r.get("reward") or 0) >= 1
            assert cell[r["state"]].setdefault((r["candidate"], r["repeat"]), v) == v, "conflicting outcome slot"
    cell["3cec061d636f899f"].setdefault((4, 0), False)  # unreleased committed invalid draw, counted as zero
    return cell


def roles():
    out = {}
    for root in ROLE_ROOTS:
        for d in (root / "states").iterdir():
            ref = d / "anchor-reference.json"
            if ref.exists():
                anchor = json.loads(Path(json.loads(ref.read_text())["path"]).read_text())
                out[d.name] = anchor["state_metadata"].get("split")
    return out


def picks_from(directory, rows):
    """Saved picks of the other selectors on these states."""
    out = {}
    if directory is None:   # original 124 states: picks from their first scoring
        out["Piotr ARM"] = {r["state_id"]: r["selected_index"] for r in map(json.loads, open(R / "piotr_arm.jsonl"))}
        for name, sub in [("Teacher-trained head", "heads/pointwise"), ("Kev-style step 150", "kev-353512"),
                          ("Kev-style step 600", "kev2-353914")]:
            out[name] = json.loads((R / sub / "metrics.json").read_text())["branch_head_picks"]
    else:
        out["Piotr ARM"] = {r["state_id"]: r["selected_index"] for r in map(json.loads, open(directory / "piotr_arm.jsonl"))}
        for name, sub in [("Teacher-trained head", "heads/pointwise"), ("Kev-style step 150", "kev-353512")]:
            out[name] = json.loads((directory / sub / "metrics.json").read_text())["branch_head_picks"]
        kev600 = directory / "kev2-353914" / "metrics.json"   # final rescore keeps it beside the other picks
        if not kev600.exists():
            kev600 = R / "score-new-kev2-354320" / directory.name / "metrics.json"
        out["Kev-style step 600"] = json.loads(kev600.read_text())["branch_head_picks"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, equivalent, parse_calls
    torch.set_num_threads(4)
    saved = torch.load(R / "heads/pointwise/head.pt", map_location="cpu")
    config = HeadConfig.from_json(saved["config"])
    args.layers, args.pools = ",".join(map(str, config.layers)), ",".join(config.pools)

    cell, role = outcomes(), roles()
    rows, cands, state_vecs, others = [], [], [], defaultdict(dict)
    for panels, features, directory in SOURCES:
        data, _ = load(args, torch, panels / "panels.jsonl", features)
        (split,) = data.values()
        for name, chosen in picks_from(directory, split["rows"]).items():
            others[name].update(chosen)
        for k, row in enumerate(split["rows"]):
            s = row["state_id"]
            if all((i, r) in cell[s] for i in range(5) for r in range(8)):
                rows.append(row); cands.append(split["candidates"][k]); state_vecs.append(split["state"][k])
    cands, state_vecs = torch.stack(cands), torch.stack(state_vecs)
    n = len(rows)

    # draws 0-5 rates pooled over action-equivalent candidates; draws 6-7 per candidate for scoring
    fit_rate, hold_rate = torch.zeros(n, 5), torch.zeros(n, 5)
    for k, row in enumerate(rows):
        s, calls = row["state_id"], [parse_calls(c) for c in row["candidates"]]
        raw = [sum(cell[s][i, r] for r in FIT) / len(FIT) for i in range(5)]
        for i in range(5):
            group = [j for j in range(5) if j == i or equivalent(calls[i], calls[j])]
            fit_rate[k, i] = sum(raw[j] for j in group) / len(group)
            hold_rate[k, i] = sum(cell[s][i, r] for r in HOLD) / len(HOLD)
    informative = fit_rate.max(1).values > fit_rate.min(1).values

    # folds grouped by task id: deterministic hash order, greedy balance by state count
    by_task = defaultdict(list)
    for k, row in enumerate(rows):
        by_task[row["task_id"]].append(k)
    fold_of, sizes = {}, [0] * args.folds
    for task in sorted(by_task, key=lambda t: hashlib.sha256(t.encode()).hexdigest()):
        f = min(range(args.folds), key=lambda j: (sizes[j], j))
        for k in by_task[task]:
            fold_of[k] = f
        sizes[f] += len(by_task[task])

    def train(train_idx, seed, warm, target_kind):
        torch.manual_seed(seed)
        head = CandidateSetHead(config)
        if warm:
            head.load_state_dict(saved["state_dict"])
        else:   # same fixed standardization, fresh weights
            fresh = head.state_dict()
            fresh.update({k: v for k, v in saved["state_dict"].items() if "mean" in k or "inv_std" in k})
            head.load_state_dict(fresh)
        idx = [k for k in train_idx if informative[k]]
        rate = fit_rate[idx]
        if target_kind == "rates":
            target = rate / rate.sum(1, keepdim=True)
        else:   # best action on draws 0-5, ties spread uniformly
            best = (rate == rate.max(1, keepdim=True).values).float()
            target = best / best.sum(1, keepdim=True)
        with torch.no_grad():
            lg0 = predict(head, {"candidates": cands[idx], "state": state_vecs[idx]}, "cpu", torch)
            initial = -(target * torch.log_softmax(lg0, -1)).sum(-1).mean().item()
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        gen = torch.Generator().manual_seed(seed)
        head.train()
        for _ in range(args.epochs):
            order = torch.randperm(len(idx), generator=gen)
            for i in range(0, len(idx), args.batch_size):
                b = order[i:i + args.batch_size]
                rows_b = torch.tensor(idx)[b]
                logits, _ = head(cands[rows_b], state_vecs[rows_b])
                loss = -(target[b] * torch.log_softmax(logits, -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            lg = predict(head, {"candidates": cands[idx], "state": state_vecs[idx]}, "cpu", torch)
            hit = (rate.gather(1, lg.argmax(1, keepdim=True)) == rate.max(1, keepdim=True).values).float().mean().item()
            final = -(target * torch.log_softmax(lg, -1)).sum(-1).mean().item()
        fit_log.append({"warm": warm, "target": target_kind, "seed": seed, "train_states": len(idx),
                        "train_best_action_hit": hit, "train_loss_final": final, "train_loss_initial": initial})
        return head

    def logits_of(head, idx):
        with torch.no_grad():
            return predict(head, {"candidates": cands[idx], "state": state_vecs[idx]}, "cpu", torch)

    fit_log = []
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = {"Outcome-adapted head (warm start)": (True, "rates"),
                "Outcome-trained head (from scratch)": (False, "rates"),
                "Outcome-adapted head, best-action target": (True, "best")}
    picks, train_picks = defaultdict(dict), defaultdict(dict)
    for name, (warm, kind) in variants.items():
        for f in range(args.folds):
            tr = [k for k in range(n) if fold_of[k] != f]; te = [k for k in range(n) if fold_of[k] == f]
            total, total_tr = 0, 0
            for seed in seeds:
                head = train(tr, seed, warm, kind)
                lg = logits_of(head, te)
                for k, l in zip(te, lg):
                    picks[f"{name}, seed {seed}"][k] = int(l.argmax())
                total, total_tr = total + lg, total_tr + logits_of(head, tr)
            for k, l in zip(te, total):
                picks[name][k] = int(l.argmax())
            train_picks[name][f] = {k: int(l.argmax()) for k, l in zip(tr, total_tr)}
        print("trained", name, flush=True)

    # role-based secondary: train on 'train' role, evaluate on validation roles
    role_of = [role.get(r["state_id"], "historical") for r in rows]
    tr = [k for k in range(n) if role_of[k] == "train"]
    te = [k for k in range(n) if role_of[k].startswith("validation")]
    total = sum(logits_of(train(tr, seed, True, "rates"), te) for seed in seeds)
    role_picks = {k: int(l.argmax()) for k, l in zip(te, total)}

    # references, all scored on draws 6-7
    score = {}
    for name, chosen in picks.items():
        score[name] = [float(hold_rate[k, chosen[k]]) for k in range(n)]
    for name, chosen in others.items():
        score[name] = [float(hold_rate[k, chosen[rows[k]["state_id"]]]) for k in range(n)]
    score["Actor's first sample"] = [float(hold_rate[k, 0]) for k in range(n)]
    score["Uniform"] = [float(hold_rate[k].mean()) for k in range(n)]
    best_fit = [(fit_rate[k] == fit_rate[k].max()).nonzero().flatten().tolist() for k in range(n)]
    score["Outcome reference: best on draws 0-5"] = [float(hold_rate[k, b].mean()) for k, b in zip(range(n), best_fit)]
    score["Hindsight best on draws 6-7 (optimistic)"] = [float(hold_rate[k].max()) for k in range(n)]

    groups = defaultdict(list)
    for k, row in enumerate(rows):
        groups[row["task_id"]].append(k)
    group_list = list(groups.values())

    def boot(a, b, idx_filter=None):
        keep = [g for g in ([k for k in g if idx_filter is None or k in idx_filter] for g in group_list) if g]
        d = {k: score[a][k] - score[b][k] for g in keep for k in g}
        rng, means = random.Random(0), []
        for _ in range(args.bootstrap):
            draw = [rng.choice(keep) for _ in keep]
            means.append(sum(d[k] for g in draw for k in g) / sum(map(len, draw)))
        means.sort()
        return [100 * sum(d.values()) / len(d), 100 * means[int(.025 * len(means))], 100 * means[int(.975 * len(means))]]

    mean = lambda name, idx=range(n): 100 * sum(score[name][k] for k in idx) / len(list(idx))
    primary = "Outcome-adapted head (warm start)"
    result = {
        "design": __doc__, "args": {k: str(v) for k, v in vars(args).items()},
        "states": n, "task_groups": len(groups), "informative_states": int(informative.sum()),
        "fold_sizes": sizes, "collections": dict(Counter(r["split"] for r in rows)),
        "roles": dict(Counter(role_of)),
        "success_on_draws_6_7": {name: mean(name) for name in score},
        "minus_uniform": {name: boot(name, "Uniform") for name in score if name != "Uniform"},
        "minus_teacher_head": {name: boot(name, "Teacher-trained head") for name in score if name != "Teacher-trained head"},
        "minus_piotr": {name: boot(name, "Piotr ARM") for name in score if name != "Piotr ARM"},
    }
    vk = sorted(role_picks)
    role_scores = {"adapted (train role)": [float(hold_rate[k, role_picks[k]]) for k in vk]}
    for name in ("Teacher-trained head", "Piotr ARM", "Uniform", "Outcome reference: best on draws 0-5"):
        role_scores[name] = [score[name][k] for k in vk]
    result["role_split"] = {r: {name: 100 * sum(v[i] for i, k in enumerate(vk) if role_of[k] == r) /
                                max(1, sum(role_of[k] == r for k in vk)) for name, v in role_scores.items()}
                            | {"states": sum(role_of[k] == r for k in vk)}
                            for r in ("validation_seen_task", "validation_unseen_task")}
    teacher_idx = {k: others["Teacher-trained head"][rows[k]["state_id"]] for k in range(n)}
    result["picks_changed_vs_teacher_head"] = {name: sum(chosen[k] != teacher_idx[k] for k in range(n)) for name, chosen in picks.items()}
    result["training_fit"] = fit_log
    # Breakdown: training states (draws 0-5 = trained on; 6-7 = unseen draws) and held-out states
    # (draws 6-7 as reported; all draws 0-7, valid because held-out states never enter training).
    rate_on = lambda k, i, reps: sum(cell[rows[k]["state_id"]][i, r] for r in reps) / len(reps)
    ALL = range(8)
    teacher = lambda k: teacher_idx[k]
    breakdown = {"by_fold": [], "held_out_all_draws": {}, "by_collection": {}, "by_role": {}}
    for f in range(args.folds):
        tr = [k for k in range(n) if fold_of[k] != f]; te = [k for k in range(n) if fold_of[k] == f]
        row = {"fold": f, "train_states": len(tr), "train_informative": int(sum(informative[k] for k in tr)), "held_out_states": len(te)}
        for name in variants:
            tp = train_picks[name][f]
            row[name] = {
                "train_draws_0_5": 100 * sum(rate_on(k, tp[k], FIT) for k in tr) / len(tr),
                "train_draws_6_7": 100 * sum(rate_on(k, tp[k], HOLD) for k in tr) / len(tr),
                "held_out_draws_6_7": 100 * sum(rate_on(k, picks[name][k], HOLD) for k in te) / len(te),
                "held_out_draws_0_7": 100 * sum(rate_on(k, picks[name][k], ALL) for k in te) / len(te)}
        for name, chooser in [("Teacher-trained head", teacher), ("Uniform", None)]:
            val = (lambda k, reps: rate_on(k, chooser(k), reps)) if chooser else \
                  (lambda k, reps: sum(rate_on(k, i, reps) for i in range(5)) / 5)
            row[name] = {"train_draws_0_5": 100 * sum(val(k, FIT) for k in tr) / len(tr),
                         "train_draws_6_7": 100 * sum(val(k, HOLD) for k in tr) / len(tr),
                         "held_out_draws_6_7": 100 * sum(val(k, HOLD) for k in te) / len(te),
                         "held_out_draws_0_7": 100 * sum(val(k, ALL) for k in te) / len(te)}
        breakdown["by_fold"].append(row)
    # held-out states scored on all eight draws, with task-group bootstrap
    for name in list(variants) + ["Teacher-trained head", "Piotr ARM", "Kev-style step 150", "Kev-style step 600", "Actor's first sample", "Uniform"]:
        if name in variants: ch = lambda k, name=name: picks[name][k]
        elif name == "Actor's first sample": ch = lambda k: 0
        elif name == "Uniform": ch = None
        else: ch = lambda k, name=name: others[name][rows[k]["state_id"]]
        score[name + " [all 8 draws]"] = [rate_on(k, ch(k), ALL) if ch else sum(rate_on(k, i, ALL) for i in range(5)) / 5 for k in range(n)]
    for name in [x for x in score if x.endswith("[all 8 draws]")]:
        breakdown["held_out_all_draws"][name] = {"success": mean(name),
            "minus_uniform": boot(name, "Uniform [all 8 draws]") if name != "Uniform [all 8 draws]" else None,
            "minus_teacher_head": boot(name, "Teacher-trained head [all 8 draws]") if name != "Teacher-trained head [all 8 draws]" else None}
    for label, key in [("by_collection", lambda k: rows[k]["split"]), ("by_role", lambda k: role_of[k])]:
        for g in sorted({key(k) for k in range(n)}):
            idx = [k for k in range(n) if key(k) == g]
            breakdown[label][g] = {"states": len(idx), **{name: mean(name, idx) for name in
                ["Outcome-adapted head (warm start) [all 8 draws]", "Outcome-trained head (from scratch) [all 8 draws]",
                 "Teacher-trained head [all 8 draws]", "Piotr ARM [all 8 draws]", "Uniform [all 8 draws]"]}}
    result["breakdown"] = breakdown
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=1))
    (args.output / "picks.json").write_text(json.dumps(
        {name: {rows[k]["state_id"]: c for k, c in chosen.items()} for name, chosen in picks.items()}, indent=1))
    print(json.dumps({k: result[k] for k in ("states", "task_groups", "informative_states", "fold_sizes", "collections", "roles")}))
    for name in sorted(score, key=lambda x: -mean(x)):
        u = result["minus_uniform"].get(name, [0, 0, 0]); t = result["minus_teacher_head"].get(name, [0, 0, 0])
        print(f"{name:48s} {mean(name):6.2f}%  vs uniform {u[0]:+6.2f} [{u[1]:+.2f}, {u[2]:+.2f}]  "
              f"vs teacher head {t[0]:+6.2f} [{t[1]:+.2f}, {t[2]:+.2f}]")
    print("role split", json.dumps(result["role_split"]))
    print("picks changed vs teacher head", json.dumps(result["picks_changed_vs_teacher_head"]))
    agg = defaultdict(list)
    for r in fit_log:
        agg[(r["warm"], r["target"])].append(r)
    for key, rs in agg.items():
        print("training fit", key, "best-action hit %.3f" % (sum(r["train_best_action_hit"] for r in rs) / len(rs)),
              "loss %.3f -> %.3f" % (sum(r["train_loss_initial"] for r in rs) / len(rs), sum(r["train_loss_final"] for r in rs) / len(rs)))


if __name__ == "__main__":
    main()
