#!/usr/bin/env python3
"""One live-site episode of the A18 WebArena harness on Online-Mind2Web, WebVoyager or DeepShop.

    live_episode.py --benchmark online-mind2web|webvoyager|deepshop --index N \
        --method no_search|arbiter|selectionarm --out DIR [--viewport 1280x720] [--usd-cap 5]

Generalizes om2w_episode.py (which stays unchanged and remains the runner of record for the OM2W rows in
tillicum/BASELINES.md section 2). Everything that defines A18 is shared, imported from om2w_episode / wa_episode:
the released WebArbiter policy prompt and parser with the full action space (WA_POLICY_STOP=full), no terminal
detector (WA_DETECTOR=off), Web-Shepherd sampling (WA_SAMPLING=shepherd: 20 GPT-4o samples, five most frequent
distinct actions), the released knockout tournament for WebArbiter-7B, SelectionARM picking from the screenshot
(wa_selectionarm.py), native accessibility tree of the current 1280x720 viewport, the live-browser patches (any
origin through the cluster proxy, desktop Chrome UA, CDP screenshots, DOM-ready navigation), the per-episode USD
cap (wa_episode.Ledger, checked every step) and the calls.jsonl cost ledger. As on OM2W, no_search runs with the
released single sample (WA_SAMPLING unset; jobs/live_cells.sbatch does this per method) and refuses Shepherd sampling.
summary.json adds selector evidence (pool size per step, tournaments won / WebArbiter calls, ARM forward passes and
fallbacks) and splits OpenAI cost into policy / judge / judge_shadow.

Tasks. Rows are read by openwebrl/run_evaluate.py's own load_tasks_from_jsonl (its source is loaded from that file,
so the start URL / intent are exactly what our arm_eval runs give the actor): WebVoyager = metadata.start_url /
metadata.intent of webvoyager_fara.jsonl (595), DeepShop = website / task_name of deepshop.jsonl (150, all Amazon,
start URL with ?language=en_US&currency=USD), OM2W = website / task_name (identical to om2w_episode.py).

Judging, per benchmark, as in our paper tables:
  * online-mind2web: unchanged from om2w_episode.py (AgentTrek prompt, executed actions + final viewport JPEG,
    gpt-5.6-terra, seed 42). Code copied verbatim.
  * webvoyager / deepshop: gpt-4o with our judges. openwebrl/eval/reward_webvoyager.py and reward_deepshop.py are
    executed from their own source (their `_judge` coroutine: prompt text, message layout, seed 42, temperature 0
    + structured DeepShopVerdict for DeepShop, SUCCESS / NOT SUCCESS parsing, 4 attempts, 180 s timeout as in
    arm_eval) with their training-stack imports (slime, openwebrl.reward_browser, ToolParser) replaced by small
    stand-ins: the agent's answer is passed where our actor's `done(response=...)` would be extracted, and the
    OpenAI client goes through the cluster proxy and records every call in the ledger. Like our protocol, an
    episode without an answer, or one that hit the step limit (the released N/A fallback = our
    max_steps_exhausted, status != COMPLETED), scores 0 without a judge call.
  * Screenshots. Every observation is saved as browser/observation-NNN.png (the released capture, viewport-sized
    PNG; observation 000 is the start page, then one per executed browser action; a send_msg_to_user adds none,
    so the last file is the page the agent answered on).
    DEVIATION FROM THE BRIEF, deliberately: the brief says our judge attaches the trajectory (up to 30 images), but
    every WebVoyager / DeepShop record of ours checked (arm_eval and run_evaluate outputs under
    /gpfs/scrubbed/piotrt/owrl/eval/pass*/, including Zixian's ADM runs) carries exactly ONE screenshot in
    metadata.full_image_list: the turn-level generator builds it from the context-filtered messages
    (context_num_screenshots=1), so judge_max_attached_imgs=30 never binds. The primary verdict therefore
    attaches the last observation only (--judge-images 1, comparable to our tables). A second, recorded-only
    verdict with the trajectory (last <=30 observations, --shadow-judge-images 30; 0 disables) is stored as
    judge_trajectory for comparison; it never changes success.
Other deviations from our own actor's eval (inherent to the A18 harness, as on OM2W): the actor is GPT-4o reading
the accessibility tree; its answer is the send_msg_to_user text; report_infeasible becomes the answer "N/A"
(released WebArena convention, wa_protocol.parse_extra_action), whereas our actor would write its reason in done().
"""
import argparse
import ast
import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om2w_episode as om  # noqa: E402  (imports wa_episode: placeholder modules, native_episode, proxy-env scrub)

we, ne, wp = om.we, om.ne, om.wp
PROXY = om.PROXY
REPO = Path('/gpfs/home/piotrt/OpenWebRL')
BENCHMARKS = {  # task file, judge (None = the OM2W judge of om2w_episode.py)
    'online-mind2web': (REPO / 'openwebrl/data/eval/online-mind2web.jsonl', None),
    'webvoyager': (REPO / 'openwebrl/data/eval/webvoyager_fara.jsonl', REPO / 'openwebrl/eval/reward_webvoyager.py'),
    'deepshop': (REPO / 'openwebrl/data/eval/deepshop.jsonl', REPO / 'openwebrl/eval/reward_deepshop.py'),
}
GPT4O_JUDGE = os.environ.get('LIVE_JUDGE', 'gpt-4o')  # arm_eval PROTOCOLS / run_evaluate defaults: "gpt-4o"
we.PRICES.setdefault('gpt-4o', (2.5, 10.0))  # the alias resolves to gpt-4o-2024-08-06 pricing


def load_tasks(task_file):
    """openwebrl.run_evaluate.load_tasks_from_jsonl, executed from its source (the module imports the training stack)."""
    source = (REPO / 'openwebrl/run_evaluate.py').read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'load_tasks_from_jsonl')
    scope = {'json': json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'run_evaluate.py', 'exec'), scope)
    return scope['load_tasks_from_jsonl'](str(task_file))


class _Sample:  # stand-in for slime.utils.types.Sample: only what the judges read
    class Status:
        COMPLETED = 'completed'

    def __init__(self, intent, answer, images):
        self.metadata = dict(intent=intent, full_image_list=images)
        self.response = answer or ''
        self.status = _Sample.Status.COMPLETED


def load_judge_module(path, client_factory):
    """Execute openwebrl/eval/reward_<benchmark>.py from source with its heavy imports stubbed (see docstring)."""
    stubs = {name: types.ModuleType(name) for name in (
        'openwebrl', 'openwebrl.base', 'openwebrl.base.utils', 'openwebrl.eval', 'openwebrl.eval._shared',
        'openwebrl.reward_browser', 'slime', 'slime.utils', 'slime.utils.types')}
    stubs['openwebrl.base.utils'].ToolParser = lambda *a, **k: None
    shared = stubs['openwebrl.eval._shared']
    shared._TOOLS_INFO = []
    # our actor's answer = the `response` argument of its final done() call; here the harness answer is passed as is
    shared._extract_final_answer = lambda _parser, response: (response or '').strip()
    shared.resolve_parser_type = lambda *a, **k: 'qwen25'
    stubs['openwebrl.reward_browser']._get_openai_client = lambda method='served': client_factory()
    stubs['openwebrl.reward_browser']._reward_semaphore = asyncio.Semaphore(1)
    stubs['slime.utils.types'].Sample = _Sample
    saved = {n: sys.modules.get(n) for n in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(f'_live_judge_{path.stem}', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    assert hasattr(module, '_judge') and hasattr(module, 'SYSTEM_PROMPT')
    return module


def selection_stats(out, method):
    """Per-step evidence that the selector actually chose among the pool (not a degenerate / fallback path)."""
    pools = []
    for f in sorted(out.glob('selector-*.json')):
        d = json.loads(f.read_text()); pools.append(len(d.get('candidates') or []))
    stats = dict(selector_steps=len(pools), pool_sizes=pools, steps_pool_ge2=sum(p >= 2 for p in pools))
    if method == 'arbiter':
        tourn = [json.loads(f.read_text()).get('tournament_success') for f in sorted(out.glob('selector-*.json'))]
        calls = out / 'calls.jsonl'
        rows = [json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
        stats.update(tournament_success=sum(t is True for t in tourn),
                     arbiter_calls=sum(r.get('kind') == 'engine' and r.get('role') == 'reward' for r in rows))
    if method == 'selectionarm' and (out / 'selectionarm.jsonl').exists():
        rows = [json.loads(x) for x in (out / 'selectionarm.jsonl').read_text().splitlines() if x.strip()]
        stats.update(arm_calls=sum(r.get('raw') is not None for r in rows),  # ARM forward passes (>= 2 shown)
                     arm_single=sum('single' in (r.get('rule') or '') for r in rows),  # one on-screen candidate
                     arm_fallbacks=sum('first candidate' in (r.get('rule') or '') for r in rows))  # none on screen
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='online-mind2web')
    ap.add_argument('--index', type=int, required=True, help='0-based row of the task file')
    ap.add_argument('--method', choices=['no_search', 'arbiter', 'selectionarm'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--policy', default='gpt-4o-2024-08-06')
    ap.add_argument('--max-steps', type=int, default=30)
    ap.add_argument('--usd-cap', type=float, default=5.0)
    ap.add_argument('--viewport', default=os.environ.get('WA_VIEWPORT') or '1280x720')
    ap.add_argument('--judge-images', type=int, default=1,
                    help='screenshots for the scored gpt-4o verdict (1 = what our paper-table judges received)')
    ap.add_argument('--shadow-judge-images', type=int, default=int(os.environ.get('LIVE_SHADOW_JUDGE_IMAGES', 30)),
                    help='recorded-only second verdict on the last N observations (0 = off)')
    a = ap.parse_args()
    if a.method == 'no_search' and os.environ.get('WA_SAMPLING') == 'shepherd':
        # Shepherd returns 5 BLOCKS, no_search's one-BLOCK check rejects every policy reply, and the released
        # coordinator then quietly ends the episode with a no-candidates terminal (smoke job 500421). A18's no_search
        # row uses the released single sample (OM2W N18, job 422923).
        sys.exit('no_search with WA_SAMPLING=shepherd is not the A18 no-selection row; unset WA_SAMPLING')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    task_file, judge_source = BENCHMARKS[a.benchmark]
    row = load_tasks(task_file)[a.index]
    assert row['index'] == a.index
    intent, start_url = row['metadata']['task'], row['start_url']
    if not intent or not start_url:
        sys.exit(f'task {a.index} has no intent / start url')
    task = dict(sites=[a.benchmark], task_id=a.index, benchmark_task_id=row['task_id'], require_login=False,
                storage_state=None, start_url=start_url, geolocation=None, intent_template='', instantiation_dict={},
                intent=intent, require_reset=False,
                eval=dict(eval_types=[f'{a.benchmark}_terminal_judge'], reference_answers={}, reference_url='', program_html=[]))
    (out / 'task.json').write_text(json.dumps(dict(benchmark=a.benchmark, index=a.index, task_id=row['task_id'],
                                                   start_url=start_url, intent=intent, task_file=str(task_file)), indent=1))
    we.RELAY = PROXY
    price = tuple(float(x) for x in (os.environ.get('OM2W_JUDGE_PRICE') or '0,0').split(','))
    we.PRICES.setdefault(om.JUDGE, price if any(price) else (2.0, 12.0))
    ledger = we.Ledger(out / 'calls.jsonl', a.usd_cap)
    native = ne.load_native()
    we.install_adaptations(native, a.method, ledger, 'greedy')
    if a.method == 'arbiter' and not os.environ.get('OPENAI_API_URL'):
        sys.exit('arbiter needs OPENAI_API_URL (the served WebArbiter-7B)')
    if a.method == 'selectionarm' and not os.environ.get('WA_SELECTOR_URL'):
        sys.exit('selectionarm needs WA_SELECTOR_URL (the served SelectionARM)')
    we.chromium_launch_patch(); om.live_browser_patches()
    Spec = native.tools['engine'].ModelSpec
    try:  # same (empty) domain knowledge as om2w_episode for every benchmark: identical A18 policy prompt
        knowledge = native.history.load_domain_knowledge('online_mind2web') or ''
    except Exception:
        knowledge = ''
    context = native.coordinator.RunContext.create(
        run_id=f'{a.benchmark}{a.index}-{a.method}-{int(started)}', intent=intent, domain='online_mind2web',
        domain_knowledge=knowledge, policy_model=Spec('openai', a.policy, 'chat'),
        reward_model=Spec('openai', we.TOURNAMENT[a.method], 'chat') if a.method in we.TOURNAMENT else None)
    width, height = (int(x) for x in a.viewport.split('x'))
    browser = ne.NativeBrowser(native, out / 'browser', viewport=dict(width=width, height=height),
                               allowed_origins=[], compatibility_geometry=True)
    arm = None
    if a.method == 'selectionarm':
        import wa_selectionarm
        arm = wa_selectionarm.SelectionARM(os.environ['WA_SELECTOR_URL'], dict(width=width, height=height), out / 'selectionarm.jsonl',
                                         crop=int(os.environ.get('WA_ARM_CROP') or 0) or None)

    async def select_control(candidates, inputs, ctx):
        if a.method == 'selectionarm':
            index, trace = arm.select(candidates, browser.observations[-1], intent)
            return index, [trace]
        if len(candidates) != 1:
            raise wp.EvaluationNotReady('no_search must yield exactly one admissible candidate')
        return 0, []
    adapter = wp.v8_decision_adapter('arbiter7b' if a.method in we.TOURNAMENT else a.method, select_control,
                                     lambda step, value: (out / f'selector-{step:03d}.json').write_text(json.dumps(wp.v8_plain(value), default=str)))
    executed = []
    original_execute = browser.execute

    async def execute(action_string):
        executed.append(action_string)
        return await original_execute(action_string)
    browser.execute = execute

    def om2w_evaluate(config, raw_trajectory, page, terminal):  # verbatim om2w_episode.py
        import httpx
        from openai import OpenAI
        prompts = om.judge_prompts()
        acts = list(executed)
        answer = (terminal or {}).get('answer') if isinstance(terminal, dict) else None
        lines = [f'Thought {i}: \nAction {i}: {x}' for i, x in enumerate(acts, 1)]
        n = len(acts) + 1
        lines.append(f'Thought {n}: \nAction {n}: ' + (f'Agent answered: {answer}' if answer else
                                                       f'Agent stopped: {terminal} (this is not a success verdict)'))
        jpeg = om.cdp_jpeg(page); (out / 'judge-final.jpg').write_bytes(jpeg)
        messages = [dict(role='system', content=prompts['SYSTEM_PROMPT']),
                    dict(role='user', content=[
                        dict(type='text', text=prompts['USER_PROMPT'].format(task=intent, thoughts_and_actions='\n\n'.join(lines))),
                        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(), detail='high'))])]
        client = OpenAI(http_client=httpx.Client(proxy=PROXY, timeout=180))
        t0 = time.time()
        r = client.chat.completions.create(model=om.JUDGE, messages=messages, seed=42)
        text = r.choices[0].message.content or ''
        ledger.add(kind='direct_chat', role='judge', model=om.JUDGE, usage=we.usage_dict(r.usage), reply=text,
                   seconds=round(time.time() - t0, 1))
        verdict = om.parse_verdict(text)
        (out / 'om2w-judge.json').write_text(json.dumps(dict(model=r.model, raw=text, parsed=verdict, url=page.url,
                                                             usage=we.usage_dict(r.usage)), indent=1))
        if verdict is None:
            raise wp.EvaluationNotReady('unparseable OM2W judgment')
        return dict(score=verdict, success=bool(verdict))

    def judge_client(role):
        import httpx
        from openai import AsyncOpenAI
        client = AsyncOpenAI(http_client=httpx.AsyncClient(proxy=PROXY, timeout=180))

        def wrap(fn):
            async def call(*args, **kwargs):
                t0 = time.time()
                try:
                    r = await fn(*args, **kwargs)
                except Exception as e:  # the judge module retries (4 attempts); a final failure invalidates below
                    ledger.add(kind='judge_attempt_failed', role=role, model=kwargs.get('model'), error=f'{type(e).__name__}: {e}')
                    raise
                ledger.add(kind='direct_chat', role=role, model=kwargs.get('model'), served_model=r.model,
                           usage=we.usage_dict(r.usage), reply=r.choices[0].message.content,
                           n_images=sum(c.get('type') == 'image_url' for m in kwargs.get('messages', [])
                                        if isinstance(m.get('content'), list) for c in m['content']),
                           seconds=round(time.time() - t0, 1))
                return r
            return call
        client.chat.completions.create = wrap(client.chat.completions.create)
        client.beta.chat.completions.parse = wrap(client.beta.chat.completions.parse)
        return client

    def gpt4o_evaluate(config, raw_trajectory, page, terminal):
        terminal = terminal if isinstance(terminal, dict) else {}
        answer = terminal.get('answer')
        shots = [p for p in sorted((out / 'browser').glob('observation-*.png'))]
        images = [base64.b64encode(p.read_bytes()).decode() for p in shots]
        record = dict(judge_model=GPT4O_JUDGE, source=str(judge_source), answer=answer, url=page.url,
                      observations=len(shots), terminal_step=terminal.get('step'), fallback=bool(terminal.get('fallback')))

        def run(role, n_images):
            if terminal.get('fallback'):  # our protocol: max_steps_exhausted -> status FAILED -> judge not run, 0
                return dict(score=0.0, text='Judge not run: step limit (released N/A fallback)', n_images=0)
            module = load_judge_module(judge_source, lambda: judge_client(role))
            args = types.SimpleNamespace(judge_max_attached_imgs=n_images, judge_api_model=GPT4O_JUDGE,
                                         judge_timeout_secs=180, judge_api_mode='served', log_judge_output=False)
            sample = _Sample(intent, answer, images)
            # own thread: this callback runs on the browser thread, where sync Playwright's loop counts as running
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(1) as pool:
                score, text, timed_out = pool.submit(asyncio.run, module._judge(args, sample, None)).result()
            return dict(score=score, text=text, timed_out=timed_out,
                        n_images=len(images[-n_images:]) if answer else 0,
                        images=[p.name for p in shots[-n_images:]] if answer else [])
        record['primary'] = primary = run('judge', a.judge_images)
        if a.shadow_judge_images > 0:
            try:
                record['trajectory'] = run('judge_shadow', a.shadow_judge_images)
            except Exception as e:
                record['trajectory'] = dict(score=None, text=f'{type(e).__name__}: {e}')
        (out / 'judge.json').write_text(json.dumps(record, indent=1))
        if primary['score'] is None:
            raise wp.EvaluationNotReady(f'gpt-4o judge gave no verdict: {primary["text"][:200]}')
        return dict(score=primary['score'], success=bool(primary['score']),
                    trajectory_score=(record.get('trajectory') or {}).get('score'))

    evaluate = om2w_evaluate if judge_source is None else gpt4o_evaluate
    episode = asyncio.run(ne.run_episode(task=task, context=context, browser=browser, completion_scope=ledger,
                                         evaluate=evaluate, evidence_root=out / 'episode', max_steps=a.max_steps,
                                         native=native, decision_adapter=adapter))
    verdict = episode.get('verdict') or {}
    valid = bool(episode.get('valid')) and ledger.errors == 0 and ledger.calls > 0
    usd = dict(total=0.0)
    calls = out / 'calls.jsonl'
    for line in (calls.read_text().splitlines() if calls.exists() else []):
        r = json.loads(line)
        if r.get('kind') == 'direct_chat':
            k = r.get('role') if r.get('role') in ('judge', 'judge_shadow') else 'policy'
            usd[k] = usd.get(k, 0.0) + (r.get('usd') or 0.0); usd['total'] += r.get('usd') or 0.0
            tok = usd.setdefault(k + '_tokens', [0, 0]); u = r.get('usage') or {}
            tok[0] += u.get('prompt_tokens', 0); tok[1] += u.get('completion_tokens', 0)
    judge = json.loads((out / 'judge.json').read_text()) if (out / 'judge.json').exists() else {}
    summary = dict(benchmark=a.benchmark, index=a.index, task_id=row['task_id'], method=a.method, policy=a.policy,
                   viewport=a.viewport, valid=valid, success=int(bool(verdict.get('success'))) if valid else None,
                   score=verdict.get('score'), trajectory_judge_score=verdict.get('trajectory_score'),
                   status=episode.get('status'), steps=len(episode.get('selected_actions', [])),
                   observations=len(list((out / 'browser').glob('observation-*.png'))),
                   answer=(episode.get('terminal') or {}).get('answer'), error=episode.get('error'),
                   judge_text=((judge.get('primary') or {}).get('text') or '')[:300] or None,
                   **selection_stats(out, a.method),
                   model_calls=ledger.calls, model_call_errors=ledger.errors, model_usd=round(ledger.usd, 4),
                   usd={k: (round(v, 4) if isinstance(v, float) else v) for k, v in usd.items()},
                   seconds=round(time.time() - started, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
