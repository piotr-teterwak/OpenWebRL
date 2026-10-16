#!/usr/bin/env python3
"""Compare selectors on the 124 branching states (CPU only).

Each --picks file is JSONL {state_id, selected_index} (0-based, original candidate order); its stem
names the selector. Reports agreement with Luna before execution and chosen-action continuation
success (invalid = 0), each with paired state-bootstrap intervals against --reference (default
piotr_arm). States missing a pick are reported, not filled in.
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", type=Path, action="append", required=True)
    ap.add_argument("--reference", default="piotr_arm")
    ap.add_argument("--panels", type=Path, default=R / "branch124/panels.jsonl")
    ap.add_argument("--luna", type=Path, default=R / "branch124/luna_before.jsonl")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    from openwebrl.selection_head import equivalent, parse_calls
    panels = {r["state_id"]: r for r in map(json.loads, args.panels.open())}
    luna = {r["state_id"]: r["selected_indices"] for r in map(json.loads, args.luna.open())}
    picks = {p.stem: {r["state_id"]: r["selected_index"] for r in map(json.loads, p.open())
                      if r.get("selected_index") is not None} for p in args.picks}
    calls = {s: [parse_calls(c) for c in panels[s]["candidates"]] for s in panels}
    same = lambda s, a, b: a == b or equivalent(calls[s][a], calls[s][b])
    agree = lambda s, i: sum(same(s, i, j) for j in luna[s]) / len(luna[s])
    rate = lambda s, i: panels[s]["outcomes"][i]["successes"] / panels[s]["outcomes"][i]["trials"]
    mean = lambda x: sum(x) / len(x)
    rng = random.Random(0)

    def ci(diffs):
        draws = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(args.draws))
        return [round(draws[int(.025 * args.draws)], 4), round(draws[int(.975 * args.draws) - 1], 4)]

    ref = picks.get(args.reference)
    result = {}
    for name, chosen in picks.items():
        states = [s for s in panels if s in chosen]
        row = {"states": len(states), "missing": len(panels) - len(states),
               "luna_agreement": round(mean([agree(s, chosen[s]) for s in states]), 4),
               "continuation_success": round(mean([rate(s, chosen[s]) for s in states]), 4)}
        if ref is not None and name != args.reference:
            shared = [s for s in states if s in ref]
            for key, f in (("luna_agreement", agree), ("continuation_success", rate)):
                d = [f(s, chosen[s]) - f(s, ref[s]) for s in shared]
                row[f"{key}_minus_{args.reference}"] = round(mean(d), 4)
                row[f"{key}_minus_{args.reference}_ci"] = ci(d)
        result[name] = row
        print(json.dumps({name: row}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
