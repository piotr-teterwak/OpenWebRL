#!/usr/bin/env python3
"""Same-hardware decision-latency benchmark for actor + selector configurations (one H200 per configuration).

Every configuration runs on one GPU: the actor's sglang server plus, where the method has one, its separate selector
co-located on the same GPU. The workload replays saved branching states (prompt, screenshot, task, history) through
the controller's own `ActionSelector` at the live concurrency (8 episodes in flight), so the actor requests,
candidate seeds, selector requests and selection code are those of the Online-Mind2Web runs. No browser is involved.

Per decision it records generation, selection and total wall time (from the selector's trace), and per configuration
throughput and peak GPU memory. Piotr configurations first re-score the saved five candidates of every state and
report agreement with the saved offline picks (`piotr_arm.jsonl`).

Lanes run configurations in parallel, one GPU each: --lanes 0,1.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE), str(SOURCE / "scripts")]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
R = RUNTIME / "selection-head-20261008"
CKPT = Path("/gpfs/scrubbed/zixianma/checkpoints/web")
ACTOR4, ACTOR8 = CKPT / "OpenWebRL-4B-SFT", CKPT / "OpenWebRL-8B-SFT-sglang"   # 8B: 4.57-layout view of the pinned v1 download
PIOTR_HF_SOURCE = RUNTIME / "evaluations/selectionarm-piotr-sameday30-20261007/source"
SAMPLING = dict(temperature=0.7, top_p=0.9, max_new_tokens=1024)

CONFIGS = {
    # name: actor, graphs, candidates, selector
    "sft4b_n1_eager": (ACTOR4, False, 1, None),
    "sft4b_n1": (ACTOR4, True, 1, None),
    "sft8b_n1": (ACTOR8, True, 1, None),
    "prior_n5": (ACTOR4, True, 5, "action_prior"),
    "piotr_hf": (ACTOR4, True, 5, "piotr_hf"),
    "piotr_sglang": (ACTOR4, True, 5, "piotr_sglang"),
    "headkv_eager": (ACTOR4, False, 5, "headkv"),
    "headkv": (ACTOR4, True, 5, "headkv"),
    "kev": (ACTOR4, True, 5, "kev"),
}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Services:
    def __init__(self, gpu, out):
        self.gpu, self.out, self.procs = gpu, out, []

    def spawn(self, name, cmd, env=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu), PYTHONPATH=str(SOURCE) + ":" + str(SOURCE / "scripts"),
                   **(env or {}))
        p = subprocess.Popen(cmd, env=env, stdout=open(self.out / f"{name}.log", "w"), stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=SOURCE)
        self.procs.append(p)
        return p

    def wait(self, url, proc, seconds=1800):
        import httpx
        deadline = time.time() + seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"service exited before ready: {url}")
            try:
                if httpx.get(url, timeout=3).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(url)

    def stop(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in self.procs:
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        # the next configuration on this GPU must start from free memory
        for _ in range(120):
            try:
                used = int(subprocess.check_output(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used",
                                                    "--format=csv,noheader,nounits"], text=True).split()[0])
            except Exception:
                break
            if used < 2048:
                break
            time.sleep(1)


def actor_cmd(model, port, graphs, mem, hooks=False):
    cmd = [str(RUNTIME / "venv/bin/python"), "-m", "sglang.launch_server", "--model-path", str(model), "--host", "127.0.0.1",
           "--port", str(port), "--dtype", "bfloat16", "--tp", "1", "--mem-fraction-static", str(mem),
           "--context-length", "32768", "--max-running-requests", "40", "--chunked-prefill-size", "4096",
           "--random-seed", "4500"]
    if not graphs:
        cmd.append("--disable-cuda-graph")
    else:
        cmd += ["--cuda-graph-max-bs", "48"]
    if hooks:
        from openwebrl.actor_head_capture import hook_specs
        cmd += ["--forward-hooks", json.dumps(hook_specs())]
    return cmd


def gpu_sampler(gpu, stop, peak):
    while not stop.is_set():
        try:
            used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True).split()[0])
            peak[0] = max(peak[0], used)
        except Exception:
            pass
        stop.wait(1)


async def piotr_parity(endpoint, states):
    """Re-score each state's saved five candidates; agreement with the saved offline (HF) picks."""
    from openwebrl.arm_inference import parse_selection, request_selection_result, split_response
    saved = {json.loads(l)["state_id"]: json.loads(l) for l in open(R / "piotr_arm.jsonl")}
    sem, rows = asyncio.Semaphore(8), []

    async def one(st):
        async with sem:
            payload = dict(mode="selection", task=st["task"], url=st["url"], history=[split_response(h) for h in st["history"]],
                           candidates=[split_response(c) for c in st["candidates"]],
                           screenshot=base64.b64encode(Path(st["image"]).read_bytes()).decode())
            t = time.monotonic()
            r = await request_selection_result(endpoint, payload, 300)
            ref = saved.get(st["state_id"], {})
            rows.append(dict(state=st["state_id"], pick=parse_selection(r["raw"], 5), saved=ref.get("selected_index"),
                             seconds=time.monotonic() - t, input_tokens=r.get("input_tokens"), saved_input_tokens=ref.get("input_tokens")))
    await asyncio.gather(*(one(st) for st in states))
    scored = [r for r in rows if r["saved"] is not None]
    return dict(states=len(scored), agree=sum(r["pick"] == r["saved"] for r in scored),
                input_tokens_match=sum(r["input_tokens"] == r["saved_input_tokens"] for r in scored),
                mean_seconds=statistics.mean(r["seconds"] for r in rows), rows=rows)


async def workload(actor_url, selector, states, out, concurrency, warmup):
    """Each state is one decision; `concurrency` decisions in flight, as the live controller's episode semaphore."""
    import httpx
    from openwebrl import generate_browser as generation
    from openwebrl.arm_inference import ActionSelector
    client = httpx.AsyncClient(trust_env=False, timeout=600)

    async def infer(url, text, sampling_params, image_data, timeout_secs=None, rid=None):
        payload = dict(text=text, sampling_params=sampling_params, return_logprob=True)
        if image_data:
            payload["image_data"] = image_data
        if rid:
            payload["rid"] = rid
        r = await client.post(url, json=payload)
        r.raise_for_status()
        o = r.json()
        meta = o["meta_info"]
        pairs = meta["output_token_logprobs"]
        tokens, logps = [p[1] for p in pairs], [p[0] for p in pairs]
        text = generation._ensure_im_end_w_new_line(o["text"], tokens, logps)
        return text, tokens, logps, meta.get("finish_reason", {}).get("type", "stop")

    mode, endpoint = selector
    sem = asyncio.Semaphore(concurrency)

    async def decide(k, st, directory):
        async with sem:
            sel = ActionSelector(mode, endpoint, directory, seed=45, record_telemetry=True)
            image = "data:image/png;base64," + base64.b64encode(Path(st["image"]).read_bytes()).decode()
            await sel(infer=infer, url=actor_url, input_text=st["prompt_text"], sampling_params=dict(SAMPLING),
                      images=[image], observation={"screenshot": Path(st["image"]).read_bytes(), "active_tab_url": st["url"]},
                      history=st["history"], task=st["task"], task_id=st["state_id"], turn=k, timeout=600)

    await asyncio.gather(*(decide(k, st, out / "warmup") for k, st in enumerate(states[:warmup])))
    started = time.monotonic()
    await asyncio.gather(*(decide(k, st, out / "selections") for k, st in enumerate(states)))
    wall = time.monotonic() - started
    await client.aclose()
    traces = [json.loads(l) for f in (out / "selections").glob("*.jsonl") for l in f.read_text().splitlines()]
    return wall, traces


def summarize(traces, wall):
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    out = dict(decisions=len(traces), wall_seconds=wall, decisions_per_second=len(traces) / wall)
    for key in ("generation_seconds", "selection_seconds", "decision_seconds"):
        xs = [t["timing"][key] for t in traces]
        out[key] = dict(mean=statistics.mean(xs), p50=q(xs, .5), p90=q(xs, .9))
    tel = [t.get("selector_telemetry_full") or {} for t in traces]
    out["selector_input_tokens_mean"] = statistics.mean(x.get("input_tokens") or 0 for x in tel)
    out["fallbacks"] = sum(bool(t.get("fallback")) for t in traces)
    return out


def run_config(name, gpu, states, all_states, out_root, concurrency, warmup):
    actor, graphs, n, selector = CONFIGS[name]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    svc, report = Services(gpu, out), dict(config=name, actor=str(actor), cuda_graphs=graphs, candidates=n, gpu=gpu)
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=gpu_sampler, args=(gpu, stop, peak), daemon=True)
    try:
        port = free_port()
        hooks = selector == "headkv"
        env = {}
        if hooks:
            # capture exports are shared by lanes; rids are unique
            env = dict(ACTORHEAD_DIR=f"/dev/shm/actorhead-bench-{os.getpid()}", ACTORHEAD_HEAD=str(R / "heads/pointwise/head.pt"),
                       ACTORHEAD_TOKENIZER=str(ACTOR4), ACTORHEAD_PARAMS=str(4022468096 + 415347712))
            os.environ.update(env)
        mem = .45 if selector in ("piotr_sglang", "piotr_hf", "kev") else .80
        actor_proc = svc.spawn("actor", actor_cmd(actor, port, graphs, mem, hooks), env)
        mode, endpoint = ("baseline", "") if n == 1 else ("action_prior", "") if selector == "action_prior" else (None, None)
        if selector == "piotr_sglang":
            sport, aport = free_port(), free_port()
            sel_proc = svc.spawn("selector-sglang", [str(RUNTIME / "venv/bin/python"), "-m", "sglang.launch_server",
                                 "--model-path", str(CKPT / "arm/selection-sglang-81b452d"), "--host", "127.0.0.1",
                                 "--port", str(sport), "--dtype", "bfloat16", "--mem-fraction-static", ".30",
                                 "--context-length", "32768", "--grammar-backend", "xgrammar", "--cuda-graph-max-bs", "48"])
            svc.wait(f"http://127.0.0.1:{sport}/health", sel_proc)
            adapter = svc.spawn("selector", [str(RUNTIME / "venv/bin/python"), str(SOURCE / "scripts/serve_arm_sglang.py"), "serve",
                                "--sglang", f"http://127.0.0.1:{sport}", "--port", str(aport)])
            svc.wait(f"http://127.0.0.1:{aport}/health", adapter)
            mode, endpoint = "selection", f"http://127.0.0.1:{aport}"
        elif selector == "piotr_hf":
            aport = free_port()
            adapter = svc.spawn("selector", [str(RUNTIME / "arm-reproduction/venv/bin/python"), str(PIOTR_HF_SOURCE / "scripts/serve_arm.py"),
                                "--mode", "selection", "--model", str(CKPT / "arm/selection/81b452d800d9f859687074f82680dd5257e02d89"),
                                "--base", str(ACTOR4), "--port", str(aport)],
                                dict(CPATH=str(RUNTIME / "src/python-headers/Include") + ":" + str(RUNTIME / "src/python-headers")))
            svc.wait(f"http://127.0.0.1:{aport}/health", adapter)
            mode, endpoint = "selection", f"http://127.0.0.1:{aport}"
        elif selector == "kev":
            aport = free_port()
            adapter = svc.spawn("selector", [str(RUNTIME / "arm-reproduction/venv/bin/python"), str(SOURCE / "scripts/serve_actor_selector.py"),
                                "--mode", "kev", "--model", str(ACTOR4), "--port", str(aport)])
            svc.wait(f"http://127.0.0.1:{aport}/health", adapter)
            mode, endpoint = "actor_scalar", f"http://127.0.0.1:{aport}"
        elif selector == "headkv":
            mode, endpoint = "actor_cache_head", ""
        svc.wait(f"http://127.0.0.1:{port}/health", actor_proc)
        sampler.start()
        if selector in ("piotr_sglang", "piotr_hf"):
            parity = asyncio.run(piotr_parity(endpoint, all_states))
            (out / "piotr-parity.json").write_text(json.dumps(parity, indent=1))
            report["piotr_parity"] = {k: v for k, v in parity.items() if k != "rows"}
        wall, traces = asyncio.run(workload(f"http://127.0.0.1:{port}/generate", (mode, endpoint), states, out, concurrency, warmup))
        report.update(summarize(traces, wall))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        report["peak_gpu_memory_mib"] = peak[0]
        svc.stop()
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("piotr_parity",)}), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--lanes", default="0", help="GPU indices; configurations are distributed across them")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prompts = {json.loads(l)["state_id"]: json.loads(l)["prompt_text"] for l in open(R / "branch124/states.jsonl")}
    states = [dict(p, prompt_text=prompts[p["state_id"]]) for p in map(json.loads, open(R / "branch124/panels.jsonl"))]
    all_states, states = states, states[:args.states]
    queue = list(args.configs.split(","))
    lock, results = threading.Lock(), {}

    def lane(gpu):
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            results[name] = run_config(name, gpu, states, all_states, args.out, args.concurrency, args.warmup)
    threads = [threading.Thread(target=lane, args=(int(g),)) for g in args.lanes.split(",")]
    for t in threads: t.start()
    for t in threads: t.join()
    (args.out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
