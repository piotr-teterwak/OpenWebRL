#!/usr/bin/env python3
"""GPT-6.1 Sol before-execution labels on branching states (no execution evidence is ever sent).

Same request as the Luna before condition (openwebrl.arm_branch_teacher.request with no immediate rows),
with only the model and reasoning effort changed; 3 repeats + 1 shuffled order per state.
States: the 124 historical states and the 160 new states labelled by Luna in luna-labels-20261011.

Budget: a local ledger reserves each call's worst case (request_bound at Sol's conservative cache-write
input rate and its output rate) before dispatch, refuses calls that would exceed the approved call or USD
caps, and settles actual usage after. Missing usage keeps the full reservation charged and halts.

  prepare | dry-run | register --job ID | execute --job ID
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
REPO = Path("/gpfs/projects/krishna/zixianma/OpenWebRL")
ROOT = RUNTIME / "selection-head-20261008/sol-labels-20261011"
LUNA = RUNTIME / "selection-head-20261008/luna-labels-20261011"
HIST = RUNTIME / ("arm-turn-bonus-preparation/outcome-reward-20260927/critic-comparison-20261004/"
                  "continuation-branches-preparation-20261006/overnight-v4-5x3")
MODEL, EFFORT = "gpt-6.1-sol", "medium"
CONDITION = os.environ.get("SOL_CONDITION", "before")   # "after": immediate evidence from draw 0 only
EVIDENCE_DRAW = 0
PRICE = {"input_cache_write": 2.50, "output": 10.0}     # USD per million tokens (repo-verified Sol 6.1 table)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=1)); os.replace(t, p)


def scoreable():
    """States with draws 3-7 for every candidate (the held-out scoring set)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from adapt_selection_head_outcomes import outcomes
    cell = outcomes()
    return {s for s, c in cell.items() if all(all((i, r) in c for r in range(3, 8)) for i in range(5))}


def states():
    out = []
    for item in read(LUNA / "plan.json")["schedule"]:
        out.append(dict(state=item["state"], collection=item["collection"],
                        anchor_reference=str(LUNA / "states" / item["state"] / "anchor-reference.json")))
    for sid in read(HIST / "accepted-states.json")["state_ids"]:
        out.append(dict(state=sid, collection="historical", anchor_reference=str(HIST / "states" / sid / "anchor-reference.json")))
    assert len({s["state"] for s in out}) == len(out)
    if CONDITION == "after":
        keep = scoreable()
        for e in out:
            base = LUNA if e["collection"] != "historical" else HIST
            e["release"] = str(base / "states" / e["state"] / "release.json")
        out = [e for e in out if e["state"] in keep]
    return out


def build(src, entry):
    """The four before-condition requests for one state, identical to Luna's except model/effort."""
    sys.path.insert(0, str(src))
    from openwebrl.arm_branch_protocol import seed
    from openwebrl.arm_branch_teacher import request
    ref = read(entry["anchor_reference"]); anchor = read(ref["path"])
    canonical = list(range(len(anchor["candidates"])))
    permuted = canonical.copy()
    random.Random(seed(entry["state"], "permutation")).shuffle(permuted)   # same permutation rule as the worker
    if permuted == canonical:
        permuted = canonical[1:] + canonical[:1]
    jobs = [(t, canonical) for t in range(3)] + [("permuted", permuted)]
    rows = None
    if CONDITION == "after":
        immediate = {}
        for d in read(entry["release"])["attempt_directories"]:
            f = Path(d) / "immediate.json"
            if f.exists():
                row = read(f)
                if row["repeat"] == EVIDENCE_DRAW:       # only draw-0 evidence is ever opened into a request
                    immediate[row["candidate"]] = row
        if sorted(immediate) != canonical:
            return []                                    # no complete draw-0 evidence: no calls for this state
        rows = [immediate[c] for c in canonical]
    out = []
    for teacher_repeat, order in jobs:
        p = request(anchor, rows, "index", order)
        p["model"], p["reasoning"] = MODEL, {"effort": EFFORT}
        out.append((teacher_repeat, order, p))
    return out


def bound_usd(p):
    sys.path.insert(0, str(ROOT / "source"))
    from openwebrl.controlled_sft_state import request_bound
    text, outcap = request_bound(p)
    return (text * PRICE["input_cache_write"] + outcap * PRICE["output"]) / 1e6


class Ledger:
    def __init__(self, root, caps):
        self.path, self.caps = root / "ledger.json", caps
        if not self.path.exists():
            write(self.path, dict(calls=0, reserved_usd=0.0, charged_usd=0.0, settled=0, halt=None))

    def _update(self, fn):
        with open(str(self.path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.path); result = fn(state); write(self.path, state)
            return result

    def reserve(self, usd):
        def fn(s):
            if s["halt"]: raise RuntimeError("ledger halted: " + s["halt"])
            if s["calls"] + 1 > self.caps["calls"]: raise RuntimeError("call cap reached")
            if s["charged_usd"] + s["reserved_usd"] + usd > self.caps["usd"]: raise RuntimeError("USD cap reached")
            s["calls"] += 1; s["reserved_usd"] += usd
        self._update(fn)

    def settle(self, reserved, usage):
        def fn(s):
            s["reserved_usd"] -= reserved
            it, ot = (usage or {}).get("input_tokens"), (usage or {}).get("output_tokens")
            if type(it) is not int or type(ot) is not int:
                s["charged_usd"] += reserved; s["halt"] = "missing usage; reservation kept"
                return None
            cost = (it * PRICE["input_cache_write"] + ot * PRICE["output"]) / 1e6
            s["charged_usd"] += cost; s["settled"] += 1
            return cost
        return self._update(fn)


def parse(raw, n):
    if not str(raw.get("model", "")).startswith(MODEL) or raw.get("status") != "completed":
        raise ValueError("wrong model or incomplete response")
    texts = [c["text"] for item in raw.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    if len(texts) != 1: raise ValueError("expected one JSON object")
    value = json.loads(texts[0])
    if set(value) != {"selection"} or type(value["selection"]) is not int or not 1 <= value["selection"] <= n:
        raise ValueError("invalid selection")
    return value


async def label_state(entry, src, ledger, client, sem, key, outdir):
    out = outdir / f"{entry['state']}.json"
    if out.exists(): return "done"
    built = build(src, entry)
    if not built:
        write(out, []); return "no evidence"
    judgments = []
    for teacher_repeat, order, p in built:
        label = dict(condition=CONDITION, execution_repeat=EVIDENCE_DRAW if CONDITION == "after" else None, teacher_repeat=teacher_repeat, order=order,
                     state=entry["state"], form="index", model=MODEL, effort=EFFORT)
        reserved = bound_usd(p)
        async with sem:
            ledger.reserve(reserved)
            try:
                r = await client.post("https://api.openai.com/v1/responses", json=p, headers={"Authorization": "Bearer " + key})
                raw = r.json()
            except Exception as exc:
                ledger.settle(reserved, None); raise RuntimeError(f"transport failure {type(exc).__name__}") from exc
            cost = ledger.settle(reserved, raw.get("usage"))
            (outdir / "raw").mkdir(exist_ok=True)
            with open(outdir / "raw" / f"{entry['state']}.jsonl", "a") as f:
                f.write(json.dumps(dict(label=label, status=r.status_code, response=raw)) + "\n")
            if cost is None: raise RuntimeError("missing usage; halted")
            try:
                r.raise_for_status(); value = parse(raw, len(order))
                judgments.append(dict(label, valid=True, candidate=order[value["selection"] - 1], output=value,
                                      usage=raw["usage"], charged_usd=cost))
            except Exception as exc:
                judgments.append(dict(label, valid=False, reason=type(exc).__name__, charged_usd=cost))
    write(out, judgments)
    return "labelled"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "dry-run", "register", "execute"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--job")
    args = ap.parse_args()
    root = args.root.resolve()
    global CONDITION
    if args.command != "prepare":   # the frozen plan, not the environment, decides the condition
        CONDITION = "after" if read(root / "plan.json")["condition"].startswith("after") else "before"
    if args.command == "prepare":
        if (root / "plan.json").exists(): raise SystemExit("already prepared; plan is frozen")
        src = root / "source"; root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--exclude", "docs", "--exclude", "__pycache__",
                        str(REPO / "openwebrl"), str(REPO / "scripts"), str(src) + "/"], check=True)
        sts = states()
        write(root / "plan.json", dict(root=str(root), model=MODEL, effort=EFFORT, price_usd_per_million=PRICE,
              condition=("before only (no execution evidence)" if CONDITION == "before" else f"after, immediate evidence from draw {EVIDENCE_DRAW} only"),
              calls_per_state=4, states=sts, source=str(src)))
        print(json.dumps(dict(states=len(sts), calls=4 * len(sts))))
    elif args.command == "dry-run":
        plan = read(root / "plan.json"); src = Path(plan["source"])
        total, n, imgs = 0.0, 0, 0
        for e in plan["states"]:
            for _, _, p in build(src, e):
                total += bound_usd(p); n += 1; imgs = max(imgs, json.dumps(p).count("data:image/"))
                assert p["model"] == MODEL and (("Immediate result" in json.dumps(p)) == (CONDITION == "after"))
        rep = dict(states=len(plan["states"]), states_without_draw0_evidence=sum(1 for e in plan["states"] if not build(src, e)), calls=n, max_images_per_call=imgs, worst_case_usd=round(total, 2))
        write(root / "dry-run.json", rep); print(json.dumps(rep))
    elif args.command == "register":
        a = read(root / "approval.json"); a.setdefault("attempts", []).append({"job_id": str(args.job), "registered_unix": time.time()})
        write(root / "approval.json", a)
    else:
        plan, approval = read(root / "plan.json"), read(root / "approval.json")
        job = str(args.job)
        if job != os.getenv("SLURM_JOB_ID") or not any(a["job_id"] == job for a in approval.get("attempts", [])):
            raise SystemExit("must run inside a registered, approved Slurm job")
        from hashlib import sha256
        if approval.get("approved") is not True or approval.get("plan_sha256") != sha256((root / "plan.json").read_bytes()).hexdigest():
            raise SystemExit("approval does not match the frozen plan")
        from dotenv import dotenv_values
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(REPO / ".env").get("OPENAI_API_KEY")
        if not key: raise SystemExit("missing credential")
        ledger = Ledger(root, approval["caps"])
        outdir = root / "judgments"; outdir.mkdir(exist_ok=True)

        async def run():
            import httpx
            sem = asyncio.Semaphore(16)
            async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
                pending = [e for e in plan["states"] if not (outdir / f"{e['state']}.json").exists()]
                for i in range(0, len(pending), 16):   # batches of 16 states, 4 sequential calls each
                    results = await asyncio.gather(*(label_state(e, Path(plan["source"]), ledger, client, sem, key, outdir)
                                                     for e in pending[i:i + 16]), return_exceptions=True)
                    errors = [r for r in results if isinstance(r, Exception)]
                    write(root / "current.json", dict(job_id=job, done=len(list(outdir.glob("*.json"))), states=len(plan["states"]),
                                                      ledger=read(root / "ledger.json"), updated_unix=time.time(),
                                                      errors=[repr(e) for e in errors]))
                    if errors: raise SystemExit("stopping on error: " + repr(errors[0]))
        asyncio.run(run())
        write(root / "closeout.json", dict(completed=True, ledger=read(root / "ledger.json")))


if __name__ == "__main__":
    main()
