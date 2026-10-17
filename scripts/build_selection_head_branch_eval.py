#!/usr/bin/env python3
"""Build the outcome-labeled branching evaluation for the selection head (CPU only).

For each accepted branching state: the five fixed candidates, the pre-action
screenshot, and each candidate's continuation outcomes (invalid = 0, matching
the operational estimand). The actor's decision prompt was not saved verbatim;
it is rebuilt from a continuation rollout's conversation text and accepted only
if its SHA-256 equals the prompt hash journaled when the candidates were drawn.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

RUN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/outcome-reward-20260927/"
           "critic-comparison-20261004/continuation-branches-preparation-20261006/overnight-v4-5x3")
PIOTR_STATES = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/"
                    "joint-data-inputs-0d83b48/openwebrl_actor/states_full.jsonl")
ASSISTANT = "<|im_start|>assistant\n"
PLACEHOLDER = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>"
FALLBACK_VARIANT = 4  # "<think>" prompt, screenshot restored directly before \n</observation>; matches 107/114 verified states


def norm(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_candidates(conversation, depth):
    """Possible turn-`depth` prompts from a later conversation text.

    Older screenshots are stripped from history, so the decision turn's
    placeholder may need to be restored. Every variant is hash-checked.
    """
    starts = [m.start() for m in re.finditer(re.escape(ASSISTANT), conversation)]
    if len(starts) <= depth:
        return []
    end = starts[depth] + len(ASSISTANT)
    prefix = conversation[:end]
    head, user = prefix.rsplit("<|im_start|>user\n", 1)
    variants = [prefix + "<think>\n", prefix]
    if PLACEHOLDER not in user:
        for anchor in ("\n</observation>", "</observation>"):
            if anchor in user:
                restored = head + "<|im_start|>user\n" + user.replace(anchor, "\n" + PLACEHOLDER + anchor, 1)
                variants += [restored + "<think>\n", restored]
                restored = head + "<|im_start|>user\n" + user.replace(anchor, PLACEHOLDER + anchor, 1)
                variants += [restored + "<think>\n", restored]
    return variants


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", default="branch124", help="split name written to every panel")
    ap.add_argument("--expect-trials", type=int, default=3,
                    help="required draws per candidate in this run's release; 0 accepts any complete panel")
    ap.add_argument("--states-from", help="comma-separated label plans (plan.json with root/state entries); only states "
                    "with non-empty judgments next to each plan are used, each read from its own campaign root")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    piotr_tasks = set()
    with open(PIOTR_STATES) as stream:
        for line in stream:
            found = re.findall(r"<user_request>(.*?)</user_request>", json.loads(line)["prompt_text"], re.S)
            if found:
                piotr_tasks.add(norm(found[-1]))

    counts, panels, states = Counter(), [], []
    if args.states_from:
        entries, seen = [], set()
        for plan_path in args.states_from.split(","):
            done = Path(plan_path).parent / "judgments"
            for e in read_json(plan_path)["states"]:
                j = done / f"{e['state']}.json"
                if e["state"] not in seen and j.exists() and read_json(j):
                    seen.add(e["state"]); entries.append((Path(e["root"]), e["state"]))
    else:
        entries = [(args.run, s) for s in read_json(args.run / "accepted-states.json")["state_ids"]]
    for run, state_id in entries:
        anchor_path = Path(read_json(run / "states" / state_id / "anchor-reference.json")["path"])
        anchor = read_json(anchor_path)
        depth = anchor["depth"]
        reserved = [read_json(p)["record"] for p in sorted((anchor_path.parent / "events").glob("*.json"))
                    if read_json(p).get("event") == "proposal_batch_reserved"]
        reserved = [r for r in reserved if r["turn"] == depth]
        if len(reserved) != 1:
            counts["skipped_missing_prompt_hash"] += 1
            continue
        target_hash, prompt_tokens = reserved[0]["prompt_sha256"], reserved[0]["prompt_tokens"]

        outcomes = defaultdict(lambda: {"successes": 0, "trials": 0, "valid": 0})
        prompt, fallback = None, None
        for attempt in read_json(run / "states" / state_id / "release.json")["attempt_directories"]:
            result = read_json(Path(attempt) / "result.json")
            o = outcomes[result["candidate"]]
            o["trials"] += 1
            o["valid"] += bool(result["valid"])
            o["successes"] += bool(result["valid"]) and (result["reward"] or 0) > 0
            if prompt is None:
                for rollout in Path(attempt).glob("rollouts/**/*.json"):
                    text = read_json(rollout).get("llm_input_texts") or ""
                    variants = prompt_candidates(text, depth)
                    prompt = next((v for v in variants if sha(v) == target_hash), None)
                    if prompt:
                        break
                    if fallback is None and len(variants) > FALLBACK_VARIANT:
                        fallback = variants[FALLBACK_VARIANT]
        verified = prompt is not None
        if not verified:
            # Replayed prefixes can word the tool feedback differently from the
            # original discovery run, whose prompt text was not saved. Keep the
            # replay-based prompt for a labeled sensitivity analysis only.
            if fallback is None:
                counts["skipped_prompt_not_reconstructed"] += 1
                continue
            prompt = fallback
            counts["prompt_unverified_replay_feedback"] += 1
        if sorted(outcomes) != list(range(5)) or any(o["trials"] != args.expect_trials if args.expect_trials else o["trials"] < 1 for o in outcomes.values()):
            counts["skipped_incomplete_outcomes"] += 1
            continue
        candidates = anchor["candidates"]
        if any(c["output"][3] != "stop" for c in candidates):
            counts["skipped_truncated_candidate"] += 1
            continue
        states.append({"state_id": state_id, "prompt_text": prompt})
        panels.append({
            "state_id": state_id, "base_state_id": state_id, "task_id": anchor["task_id"], "depth": depth,
            "split": args.split, "task_group": anchor["task_id"],
            "image": anchor["before_observation"]["screenshot"]["path"],
            "screenshot_sha256": anchor["before_observation"]["screenshot"]["sha256"],
            "prompt_sha256": target_hash, "server_prompt_tokens": prompt_tokens, "prompt_verified": verified,
            "task": anchor["task"], "url": anchor["before_observation"].get("active_tab_url", ""),
            "history": [entry["output"][0] for entry in anchor["prefix"]],
            "candidates": [c["output"][0] for c in candidates],
            "candidate_token_ids": [c["output"][1] for c in candidates],
            "action_keys": [c["action_key"] for c in candidates],
            "outcomes": [outcomes[i] for i in range(5)],
            "piotr_task_overlap": norm(anchor["task"]) in piotr_tasks})
        counts["accepted"] += 1
        counts["prompt_verified"] += verified

    with open(args.output / "panels.jsonl", "w") as out:
        for panel in panels:
            out.write(json.dumps(panel, ensure_ascii=False) + "\n")
    with open(args.output / "states.jsonl", "w") as out:
        for state in states:
            out.write(json.dumps(state, ensure_ascii=False) + "\n")

    rate = lambda o: o["successes"] / o["trials"]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    reference = {
        "states": len(panels),
        "prompt_verified_states": sum(p["prompt_verified"] for p in panels),
        "actor_first": mean([rate(p["outcomes"][0]) for p in panels]),
        "uniform": mean([mean([rate(o) for o in p["outcomes"]]) for p in panels]),
        "hindsight_oracle": mean([max(rate(o) for o in p["outcomes"]) for p in panels]),
        "piotr_task_overlap_states": sum(p["piotr_task_overlap"] for p in panels),
    }
    summary = {"counts": dict(counts), "reference": reference,
               "published_reference_124": {"actor_first": 0.2984, "uniform": 0.3242, "hindsight_oracle": 0.5000}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
