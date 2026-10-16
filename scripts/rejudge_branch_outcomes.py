#!/usr/bin/env python3
"""Re-judge saved branch continuations with a second outcome judge (GPT-6 Luna) on the exact logged judge input.

Each judged continuation's attempt journal records the original o4-mini request (`judge_reserved`: system prompt, task,
trajectory text, final answer, screenshots) and its verdict. `prepare` samples states (seeded) and lists every valid,
judged held-out continuation (draws 3-7) of every candidate; continuations without an extractable answer were scored 0
without a judge and are left as they are. `execute` replays each logged request unchanged to the second judge (system
prompt -> instructions, user content -> input) under a hard-capped ledger and records both verdicts.

Usage: rejudge_branch_outcomes.py prepare --root R --states-from PLAN[,PLAN] --n 100 [--piotr-only]
       rejudge_branch_outcomes.py dry-run --root R
       rejudge_branch_outcomes.py execute --root R
"""
import argparse
import ast
import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_branch_sol_labels import REPO, Ledger, read, write  # noqa: E402
from openwebrl.eval.reward_online_mind2web import _parse_verdict as parse_judge_verdict  # noqa: E402  (Status: "success"|"failure")

MODEL, EFFORT = "gpt-6-luna", "medium"
PRICE = {"input_cache_write": 0.125, "output": 0.5}           # USD per million tokens, as the labelling runs
HELD_OUT = range(3, 8)
TRANSIENT = {429, 500, 502, 503, 504}
PIOTR = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
             "parallel-fresh64-preparation-20261010/piotr-lineage/task-membership.private.json")
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def event(path):
    record = read(path)["record"]
    return ast.literal_eval(record) if isinstance(record, str) else record


def judged_items(entry):
    """Valid held-out continuations of one state that reached the judge, with their logged request and verdict."""
    root = Path(entry["root"])
    db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
    items, skipped = [], dict(not_valid=0, no_judge_call=0, no_logged_request=0)
    for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
        x = json.loads(raw)
        if x.get("mode") != "branch" or not x.get("released") or x["state"] != entry["state"] or x["repeat"] not in HELD_OUT:
            continue
        if not x["valid"]:
            skipped["not_valid"] += 1; continue
        if not x.get("judge_http_attempts"):
            skipped["no_judge_call"] += 1; continue
        events = sorted((root / "attempts" / x["attempt_id"] / "events").glob("*.json"))
        req = [f for f in events if read(f)["event"] == "judge_reserved"]
        got = [f for f in events if read(f)["event"] == "judge_received"]
        if not req or not got:
            skipped["no_logged_request"] += 1; continue
        items.append(dict(state=entry["state"], candidate=x["candidate"], repeat=x["repeat"], attempt_id=x["attempt_id"],
                          request_event=str(req[-1]), response_event=str(got[-1]), original_reward=float(x.get("reward") or 0)))
    return items, skipped


def to_responses(request):
    """The logged chat-completions request as a Responses request for the second judge; content unchanged."""
    system = [m["content"] for m in request["messages"] if m["role"] == "system"]
    content = []
    for m in request["messages"]:
        if m["role"] != "user":
            continue
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for c in parts:
            if c["type"] == "text":
                content.append({"type": "input_text", "text": c["text"]})
            elif c["type"] == "image_url":
                content.append({"type": "input_image", "image_url": c["image_url"]["url"], "detail": c["image_url"].get("detail", "auto")})
    return dict(model=MODEL, reasoning={"effort": EFFORT}, max_output_tokens=request.get("max_completion_tokens", 4096), store=False,
                instructions="\n".join(system), input=[{"role": "user", "content": content}])


def bound_usd(p):
    text = sum(len(c.get("text", "")) for c in p["input"][0]["content"]) + len(p["instructions"])
    images = sum(c["type"] == "input_image" for c in p["input"][0]["content"])
    est_in = text / 2.5 + images * 2000                       # conservative: chars per token, tokens per high-detail image
    return (est_in * PRICE["input_cache_write"] + p["max_output_tokens"] * PRICE["output"]) / 1e6


async def judge_one(item, client, key, ledger, sem, outdir):
    out = outdir / f"{item['attempt_id']}.json"
    if out.exists():
        return
    p = to_responses(event(item["request_event"])["request"])
    reserved = bound_usd(p)
    async with sem:
        for attempt in range(4):
            ledger.reserve(reserved)
            try:
                r = await client.post("https://api.openai.com/v1/responses", json=p, headers={"Authorization": "Bearer " + key})
                raw = r.json() if r.status_code not in TRANSIENT else {}
                transient = r.status_code in TRANSIENT
            except Exception:
                transient, raw, r = True, {}, None
            if transient and not (raw.get("usage") or {}):
                ledger._update(lambda s: s.update(reserved_usd=s["reserved_usd"] - reserved, charged_usd=s["charged_usd"] + reserved))
                if attempt == 3:
                    raise RuntimeError("transient provider failure persisted")
                await asyncio.sleep(5 * 2 ** attempt)
                continue
            break
        usage = raw.get("usage") or {}
        charged = ledger.settle(reserved, {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]} if "input_tokens" in usage else None)
        if charged is None:
            raise RuntimeError("missing usage; halted")
    texts = [c["text"] for o in raw.get("output", []) if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text"]
    text = texts[0] if len(texts) == 1 else ""
    verdict = parse_judge_verdict(text) if text else None
    original_text = event(item["response_event"])["response"]["choices"][0]["message"]["content"]
    write(out, dict(item, judge=MODEL, effort=EFFORT, status=raw.get("status"), second_reward=verdict, second_text=text, usage=usage,
                    original_judge_reward=parse_judge_verdict(original_text), original_text=original_text))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "dry-run", "execute"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--states-from"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--piotr-only", action="store_true"); ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        if (root / "plan.json").exists():
            raise SystemExit("already prepared; plan is frozen")
        piotr = set(read(PIOTR)["piotr_training_task_counts"]) if args.piotr_only else None
        pool = []
        for plan_path in args.states_from.split(","):
            for e in read(plan_path)["states"]:
                done = Path(plan_path).parent / "judgments" / f"{e['state']}.json"
                if not (done.exists() and read(done)):
                    continue                                  # labelled states only
                if piotr is not None:
                    a = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
                    if norm(a["task"] if isinstance(a["task"], str) else json.dumps(a["task"])) not in piotr:
                        continue
                pool.append(e)
        random.Random(args.seed).shuffle(pool)
        chosen, items, skipped = pool[:args.n], [], {}
        for e in chosen:
            it, sk = judged_items(e)
            items += it
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
        root.mkdir(parents=True, exist_ok=True)
        write(root / "plan.json", dict(model=MODEL, effort=EFFORT, seed=args.seed, held_out_draws=list(HELD_OUT), price_usd_per_million=PRICE,
                                       states=chosen, items=items, skipped=skipped, pool=len(pool), piotr_only=args.piotr_only,
                                       note="replays each logged o4-mini judge request unchanged to the second judge"))
        print(json.dumps(dict(states=len(chosen), items=len(items), skipped=skipped, pool=len(pool))))
    elif args.command == "dry-run":
        plan = read(root / "plan.json")
        bounds = [bound_usd(to_responses(event(i["request_event"])["request"])) for i in plan["items"]]
        rep = dict(states=len(plan["states"]), calls=len(plan["items"]), worst_case_usd=round(sum(bounds), 2), skipped=plan["skipped"])
        write(root / "dry-run.json", rep); print(json.dumps(rep))
    else:
        plan, approval = read(root / "plan.json"), read(root / "approval.json")
        if approval.get("approved") is not True or approval.get("plan_sha256") != hashlib.sha256((root / "plan.json").read_bytes()).hexdigest():
            raise SystemExit("approval does not match the frozen plan")
        from dotenv import dotenv_values
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(REPO / ".env").get("OPENAI_API_KEY")
        import run_branch_sol_labels as sol
        sol.PRICE = PRICE          # the shared ledger settles at Luna's rates (it defaults to Sol's)
        ledger = Ledger(root, approval["caps"])
        outdir = root / "verdicts"; outdir.mkdir(exist_ok=True)

        async def run():
            import httpx
            sem = asyncio.Semaphore(args.concurrency)
            async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
                for coro in asyncio.as_completed([judge_one(i, client, key, ledger, sem, outdir) for i in plan["items"]]):
                    try:
                        await coro
                    except RuntimeError as exc:
                        print("stopping on error:", repr(exc), flush=True)
                        return
        asyncio.run(run())
        print(json.dumps(dict(verdicts=len(list(outdir.glob("*.json"))), items=len(plan["items"]))))


if __name__ == "__main__":
    main()
