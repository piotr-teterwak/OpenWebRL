#!/usr/bin/env python3
"""Luna before/after labels for already-released branching states, in a teacher-only campaign root.

Reuses openwebrl.arm_branch_teacher unchanged. Each state's release.json in the new root lists only
the execution draws used as evidence (default 0-2), and the plan's continuation_repetitions equals
their count, so the worker opens and shows only those draws: 4 before + 10 after calls per state,
the same layout as the historical 124 states. Later draws (3-7, including held-out 6-7) are never read.

  prepare   build root, plan, task file, per-state references and filtered releases (no network)
  dry-run   build every request offline; check evidence draws, call counts and worst-case reservation
  register  add an approved Slurm job id to approval.json (after the user approves the exact caps)
  execute   inside that Slurm job: validate the allocation, then run teacher workers (CPU only)
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
REPO = Path("/gpfs/projects/krishna/zixianma/OpenWebRL")
B = RUNTIME / "arm-turn-bonus-preparation/branch-selector-scaleup-20261008"
SOURCES = {"pilot32": B / "pilot32-v1", "fresh64": B / "expansion64-20261010-v1",
           "fresh64succ": B / "next-fresh-batch-preparation-20261010/staged-campaign-v1"}
ROOT = RUNTIME / "selection-head-20261008/luna-labels-20261011"
EVIDENCE = (0, 1, 2)
LIMITS = {"browser_sessions": 0, "browser_sessions_per_condition": 0, "canonical_judge_http_attempts": 0,
          "canonical_judge_usd": 0.0, "concurrent_browsers": 0, "jev_selector_requests": 0, "local_kev_requests": 0,
          "local_sft_generations": 0, "luna_selector_requests": 2400, "luna_usd": 15.0}
RESOURCES = {"collectors": 8, "cpus": 4, "gpus": 0, "memory_gib": 16, "total_seconds": 14400}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=1)); os.replace(tmp, path)


def prepare(root):
    if (root / "plan.json").exists():
        raise SystemExit("root already prepared; plans are frozen")
    schedule, tasks, states_meta = [], {}, {}
    for name, source in SOURCES.items():
        plan = read(source / "plan.json")
        task_rows = {json.loads(x)["metadata"]["task_id"]: json.loads(x)
                     for x in Path(plan["task_file"]).read_text().splitlines() if x.strip()}
        for sid in read(source / "accepted-states.json")["state_ids"]:
            sdir = source / "states" / sid
            if not (sdir / "release.json").exists():
                continue
            assert sid not in states_meta, "state id reused across collections"
            ref = read(sdir / "anchor-reference.json")
            anchor = read(ref["path"])
            release = read(sdir / "release.json")
            keep = []
            for d in release["attempt_directories"]:
                # only the draw index decides inclusion; invalid draws may lack immediate.json and the
                # worker then marks that draw's after-calls invalid without an API call
                repeat = read(Path(d) / "result.json")["repeat"]
                if repeat in EVIDENCE:
                    keep.append(d)
            assert len(keep) == 5 * len(EVIDENCE), (sid, len(keep))
            out = root / "states" / sid
            out.mkdir(parents=True)
            shutil.copy(sdir / "anchor-reference.json", out / "anchor-reference.json")
            write(out / "release.json", dict(release, attempt_directories=keep, filtered_evidence_repeats=list(EVIDENCE),
                                              source_release=str(sdir / "release.json")))
            tasks[anchor["task_id"]] = task_rows[anchor["task_id"]]
            states_meta[sid] = name
            schedule.append(dict(state=sid, condition="L01", task_id=anchor["task_id"], depth=anchor["depth"],
                                 category="primary", mode="teacher", item_id=f"{sid}-teacher", collection=name))
    (root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks.values()))
    source = root / "source"
    subprocess.run(["rsync", "-a", "--exclude", "docs", "--exclude", "__pycache__",
                    str(REPO / "openwebrl"), str(REPO / "scripts"), str(source) + "/"], check=True)
    plan = dict(root=str(root), experiment_id="luna-labels-new-branch-states-20261011",
                protocol_name="luna-before-after-index-v1 (arm_branch_teacher unchanged)",
                candidate_count=5, continuation_repetitions=len(EVIDENCE), evidence_repeats=list(EVIDENCE),
                conditions={"L01": {"actor": "official OpenWebRL-4B-SFT"}}, task_file=str(root / "tasks.jsonl"),
                resources=RESOURCES, limits=LIMITS, schedule=schedule, source=str(source),
                source_collections={k: str(v) for k, v in SOURCES.items()},
                calls_per_state=4 + 3 * len(EVIDENCE) + 1, states=len(schedule))
    write(root / "plan.json", plan)
    print(json.dumps(dict(states=len(schedule), by_collection={n: sum(1 for x in schedule if x["collection"] == n) for n in SOURCES},
                          calls=len(schedule) * plan["calls_per_state"])))


def dry_run(root):
    """Build all requests offline with the worker's own request(); no network, no ledger writes."""
    sys.path.insert(0, str(root / "source"))
    from openwebrl.arm_branch_teacher import request
    from openwebrl.controlled_sft_state import request_bound
    plan = read(root / "plan.json")
    total_calls, worst_usd, max_images, skipped = 0, 0.0, 0, 0
    for item in plan["schedule"]:
        sdir = root / "states" / item["state"]
        ref = read(sdir / "anchor-reference.json"); anchor = read(ref["path"])
        release = read(sdir / "release.json")
        immediate = {}
        for d in release["attempt_directories"]:
            if (Path(d) / "immediate.json").exists():
                row = read(Path(d) / "immediate.json"); immediate[row["candidate"], row["repeat"]] = row
        assert {r for _, r in immediate} <= set(plan["evidence_repeats"]), item["state"]
        canonical = list(range(5))
        jobs = [None] * 4 + [r for r in plan["evidence_repeats"] for _ in range(3)] + [plan["evidence_repeats"][0]]
        for r in jobs:
            if r is not None and not all((c, r) in immediate for c in canonical):
                skipped += 1; continue      # the worker records these as invalid without a request
            rows = None if r is None else [immediate[c, r] for c in canonical]
            payload = request(anchor, rows, "index", canonical)
            text, out = request_bound(payload)
            worst_usd += (text * 0.125 + out * 0.5) / 1e6
            max_images = max(max_images, json.dumps(payload).count("data:image/"))
            total_calls += 1
    report = dict(states=len(plan["schedule"]), calls=total_calls, after_calls_skipped_missing_evidence=skipped, max_images_per_call=max_images,
                  worst_case_reservation_usd=round(worst_usd, 2), caps=plan["limits"]["luna_selector_requests"],
                  usd_cap=plan["limits"]["luna_usd"], evidence_repeats=plan["evidence_repeats"])
    write(root / "dry-run.json", report)
    print(json.dumps(report))


def register(root, job):
    approval = read(root / "approval.json")
    approval.setdefault("attempts", []).append({"job_id": str(job), "registered_unix": time.time()})
    write(root / "approval.json", approval)


def execute(root, job):
    sys.path.insert(0, str(root / "source"))
    from openwebrl.arm_branch_state import BranchState
    from openwebrl.controlled_sft_state import digest
    plan, approval = read(root / "plan.json"), read(root / "approval.json")
    if str(job) != os.getenv("SLURM_JOB_ID") or f"/job_{job}/" not in Path("/proc/self/cgroup").read_text():
        raise SystemExit("must run inside the registered Slurm allocation")
    if approval.get("approved") is not True or approval["plan_sha256"] != digest(plan):
        raise SystemExit("approval does not match the frozen plan")
    prior = [a for a in approval["attempts"] if a["job_id"] != str(job)]
    used = 0
    for a in prior:   # every earlier attempt's elapsed time counts against the plan's total
        out = subprocess.run(["sacct", "-j", a["job_id"], "-X", "-n", "-o", "ElapsedRaw"], capture_output=True, text=True).stdout.split()
        used += int(out[0]) if out else 0
    limit = plan["resources"]["total_seconds"] - used
    if limit <= 600:
        raise SystemExit("approved time exhausted")
    deadline = time.time() + limit - 300
    journal = root / "scheduler-attempts" / str(job)
    write(journal / "allocation-validated.json", dict(job_id=str(job), plan_sha256=digest(plan), resources=plan["resources"],
                                                     limits=plan["limits"], prior_actual_seconds=used,
                                                     current_attempt_limit=limit, work_deadline_unix=deadline))
    from dotenv import dotenv_values
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root / "source"))
    for key, value in dotenv_values(REPO / ".env").items():
        if value and key.startswith("OPENAI_"):
            env.setdefault(key, value)
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("missing teacher credential")
    state = BranchState(root)
    pending = [x for x in plan["schedule"] if state.item(x["item_id"])[0]["status"] == "pending"]
    running = []
    python = str(RUNTIME / "venv/bin/python")
    while pending or running:
        if time.time() > deadline:
            raise SystemExit("allocation boundary reached; remaining items stay pending")
        snap = state.snapshot()
        if snap["halt"]:
            raise SystemExit("durable halt: " + json.dumps(snap["halt"]))
        for p, item in list(running):
            if p.poll() is not None:
                running.remove((p, item))
                if p.returncode != 0:
                    raise SystemExit(f"teacher worker failed for {item['item_id']} (see log)")
        while pending and len(running) < plan["resources"]["collectors"]:
            item = pending.pop(0)
            log = (journal / f"{item['item_id']}.log").open("a")
            p = subprocess.Popen([python, "-m", "openwebrl.arm_branch_teacher", "--root", str(root), "--item", item["item_id"]],
                                 cwd=str(root / "source"), env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, item))
        write(root / "current.json", dict(job_id=str(job), updated_unix=time.time(), pending=len(pending),
                                          running=len(running), **{k: v for k, v in snap.items() if k != "halt"}))
        time.sleep(5)
    write(journal / "closeout.json", dict(completed=True, snapshot=state.snapshot()))
    print("all teacher items finished")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "dry-run", "register", "execute"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--job")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        root.mkdir(parents=True, exist_ok=True); prepare(root)
    elif args.command == "dry-run":
        dry_run(root)
    elif args.command == "register":
        register(root, args.job)
    else:
        execute(root, args.job)


if __name__ == "__main__":
    main()
