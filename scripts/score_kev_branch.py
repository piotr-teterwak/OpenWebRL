#!/usr/bin/env python3
"""Score the 124 branching states with Kev-27B under the controlled selector protocol (one GPU).

Builds each state's selector input with controlled_selector_protocol.build_selection_batch from the
saved pre-action observation (page text, interactive elements, tabs, executed-action log) and the five
fixed candidates, exactly as in the local v2 study. Kev is order-sensitive, so each state is scored
under the protocol's own shuffle (primary) and under extra seeded permutations (order-averaged
sensitivity). Writes per-state probabilities in original candidate order, then Luna agreement and
chosen-action continuation success beside Piotr ARM and the selection heads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")


def server_environment():
    """Same pinned serving overrides as the audited Kev evaluations (scripts/evaluate_kev_pair.py)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("KEV_")}
    env.update(HF_HOME=str(RUNTIME / "hf-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_XET_CACHE=str(RUNTIME / "hf-cache/xet"),
               TRITON_CACHE_DIR=str(RUNTIME / "kev-preparation-20261004/triton-cache"),
               TORCHINDUCTOR_CACHE_DIR=str(RUNTIME / "kev-preparation-20261004/inductor-cache"),
               KEV_BACKEND="torch", KEV_DTYPE="bf16", KEV_FUSED="1", KEV_CUDA_GRAPHS="1",
               KEV_MERGE="1", KEV_LORA_SCALE="1", KEV_TRUNCATE_STATES="0", KEV_DATE_FACTS="0",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    return env


def batches_for(anchor, permutations):
    """Primary protocol batch plus extra distinct seeded orders (task_id only seeds the shuffle)."""
    from openwebrl.controlled_selector_protocol import build_selection_batch
    obs = anchor["before_observation"]
    make = lambda tid: build_selection_batch(
        task_id=tid, task=anchor["task"], turn=anchor["depth"], observation=obs,
        executed_actions=obs["controlled"]["executed_actions"],
        actor_outputs=[c["output"] for c in anchor["candidates"]])
    out, orders, k = [make(anchor["task_id"])], set(), 0
    orders.add(out[0].displayed_order)
    while len(out) < 1 + permutations:
        k += 1
        batch = make(f"{anchor['task_id']}#order{k}")
        if batch.displayed_order not in orders:
            orders.add(batch.displayed_order); out.append(batch)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--port", type=int, default=19311)
    ap.add_argument("--permutations", type=int, default=4, help="extra candidate orders per state")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="build all requests; no server")
    args = ap.parse_args()

    import httpx
    from build_selection_head_branch_eval import RUN, read_json
    from openwebrl import kev_eval
    from openwebrl.controlled_selector_protocol import selector_request, selector_result

    args.output.mkdir(parents=True, exist_ok=True)
    states = read_json(RUN / "accepted-states.json")["state_ids"][:args.limit]
    # Anchors carry full page snapshots; load one at a time and keep only the selector inputs.
    plans = {}
    for s in states:
        anchor = read_json(Path(read_json(RUN / "states" / s / "anchor-reference.json")["path"]))
        plans[s] = batches_for(anchor, args.permutations)
        del anchor
    print(json.dumps({"states": len(plans), "requests": sum(map(len, plans.values()))}), flush=True)
    if args.dry_run:
        return

    spec = kev_eval.model_spec("27b")
    endpoint = f"http://127.0.0.1:{args.port}/v1/systemone"
    log = open(args.output / "server.log", "a")
    server = subprocess.Popen([str(RUNTIME / "kev-eval-venv/bin/python"), "-m", "kev.serve", "--run", spec["server_run"],
                               "--host", "127.0.0.1", "--port", str(args.port)],
                              cwd=RUNTIME / "kev-upstream", env=server_environment(), stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("Kev server exited before readiness; see server.log")
            try:
                card = kev_eval.check_server(endpoint, spec)
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Kev server not ready within 900 s")
                time.sleep(5)
        (args.output / "server-identity.json").write_text(json.dumps(card, indent=1))
        done = set()
        path = args.output / "kev_branch.jsonl"
        if path.exists():
            done = {json.loads(l)["state_id"] for l in path.open()}
        with httpx.Client(timeout=600, trust_env=False) as client, path.open("a") as out:
            for sid in states:
                if sid in done:
                    continue
                orders = []
                for batch in plans[sid]:
                    response = client.post(endpoint, json=selector_request(batch, "kev"))
                    response.raise_for_status()
                    raw = response.json()
                    result = selector_result(batch, "kev", raw)   # validates model, probabilities, argmax
                    probs = raw["answers"]["selection"]["probabilities"]
                    original = [0.0] * 5
                    for displayed, original_index in enumerate(batch.displayed_order, start=1):
                        original[original_index] = probs[str(displayed)]
                    orders.append({"displayed_order": list(batch.displayed_order), "selected_index": result.original_index,
                                   "probabilities": original, "state_sha256": batch.state_sha256})
                mean = [sum(o["probabilities"][i] for o in orders) / len(orders) for i in range(5)]
                out.write(json.dumps({"state_id": sid, "selected_index": orders[0]["selected_index"],
                                      "order_averaged_index": max(range(5), key=lambda i: mean[i]),
                                      "order_averaged_probabilities": mean, "orders": orders}) + "\n")
                out.flush()
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    rows = [json.loads(l) for l in path.open()]
    with open(args.output / "kev_primary.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"state_id": r["state_id"], "selected_index": r["selected_index"]}) + "\n")
    with open(args.output / "kev_order_averaged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"state_id": r["state_id"], "selected_index": r["order_averaged_index"]}) + "\n")
    flips = sum(len({o["selected_index"] for o in r["orders"]}) > 1 for r in rows)
    print(json.dumps({"scored": len(rows), "states_whose_choice_depends_on_order": flips}), flush=True)


if __name__ == "__main__":
    main()
