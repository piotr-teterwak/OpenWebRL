#!/usr/bin/env python3
"""Build teacher-labeled candidate panels for the selection head (CPU only).

Joins Piotr's pinned OpenWebRL actor release (states, five-candidate draws and
draw-level GPT-5.5 selections), reuses the task-grouped joint-data v2 split and
quarantine, and writes one panel per labeled draw with an equivalence-aware
soft target. No models or images are loaded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.selection_head import equivalence_target, parse_calls  # noqa: E402

ARM = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction")
REVISION = "0d83b48c1659cac47a1044ef88fb573d3c16e180"


def read_jsonl(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=ARM / "joint-data-inputs-0d83b48/openwebrl_actor")
    ap.add_argument("--joint", type=Path, default=ARM / "joint-data-v2-20260910")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    states = {}
    for row in read_jsonl(args.release / "states_full.jsonl"):
        image = args.release / "state_images" / Path(row["images"][0]).name
        states[row["state_id"]] = {"episode_id": row["episode_id"], "turn": row["turn"],
                                   "image": str(image), "image_exists": image.exists(),
                                   "prompt_sha256": hashlib.sha256(row["prompt_text"].encode()).hexdigest()}
    splits = {r["id"].removeprefix("Piotr:"): r for r in read_jsonl(args.joint / "split-manifest.jsonl")
              if r["source"] == "Piotr"}
    quarantined = {r["base_id"].removeprefix("Piotr:") for r in read_jsonl(args.joint / "quarantine-states.jsonl")
                   if r["source"] == "Piotr"}
    labels = {}
    for row in read_jsonl(args.release / "labels_drawlevel.jsonl"):
        if row["state_id"] in labels:
            counts["duplicate_label_ids"] += 1
        labels[row["state_id"]] = row["selection"]

    # Draw IDs that occur with different candidate sets cannot be joined to a label.
    seen, conflicting = {}, set()
    for row in read_jsonl(args.release / "candidates_merged.jsonl"):
        digest = hashlib.sha256(json.dumps(row["candidates"]).encode()).hexdigest()
        if seen.setdefault(row["state_id"], digest) != digest:
            conflicting.add(row["base_state_id"])
    counts["conflicting_base_states"] = len(conflicting)

    written, emitted = Counter(), set()
    with open(args.output / "panels.jsonl", "w") as out:
        for row in read_jsonl(args.release / "candidates_merged.jsonl"):
            counts["draw_records"] += 1
            sid, base = row["state_id"], row["base_state_id"]
            reason = None
            if sid in emitted:
                reason = "repeated_draw_record"
            elif sid not in labels:
                reason = "unlabeled_draw"
            elif not isinstance(labels[sid], int):
                reason = "missing_teacher_selection"
            elif base in conflicting:
                reason = "conflicting_draw_ids"
            elif base in quarantined:
                reason = "joint_v2_quarantine"
            elif base not in states:
                reason = "missing_state"
            elif not states[base]["image_exists"]:
                reason = "missing_image"
            elif base not in splits or splits[base].get("excluded"):
                reason = "not_in_split"
            elif Path(states[base]["image"]).name != Path(row["images"][0]).name:
                reason = "image_mismatch"
            elif len(row["candidates"]) != 5 or not 0 <= labels[sid] < 5:
                reason = "bad_panel_shape"
            if reason:
                counts["skipped_" + reason] += 1
                continue
            emitted.add(sid)
            calls = [parse_calls(c) for c in row["candidates"]]
            selection = labels[sid]
            target = equivalence_target(calls, selection)
            informative = 0 < sum(t > 0 for t in target) < 5
            split = splits[base]["split"]
            record = {"state_id": sid, "base_state_id": base, **{k: states[base][k] for k in
                      ("episode_id", "turn", "image", "prompt_sha256")},
                      "split": split, "task_group": splits[base]["task_group"],
                      "temperature": row["temperature"], "selection": selection, "target": target,
                      "parseable": [c is not None for c in calls], "informative": informative,
                      "candidates": row["candidates"]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written[(split, "panels")] += 1
            written[(split, "informative")] += informative
            written[(split, "selected_unparseable")] += calls[selection] is None
    bases = Counter()
    for sid in emitted:
        bases[sid.split("#", 1)[0]] += 1
    summary = {"release_revision": REVISION, "counts": dict(counts),
               "panels": {f"{s}/{k}": v for (s, k), v in sorted(written.items())},
               "base_states": len(bases), "panels_per_base_state_max": max(bases.values(), default=0)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
