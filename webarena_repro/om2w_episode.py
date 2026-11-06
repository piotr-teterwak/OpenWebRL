#!/usr/bin/env python3
"""One Online-Mind2Web episode with the A18 WebArena harness (wa_episode.py), on live public websites.

    om2w_episode.py --index 0 --method no_search|arbiter|selectionarm --out DIR [--viewport 1280x720]

Everything that defines A18 is wa_episode.py unchanged: the released WebArbiter policy prompt and parser with the full
action space (WA_POLICY_STOP=full), no terminal detector (WA_DETECTOR=off), Web-Shepherd sampling for the selector rows
(WA_SAMPLING=shepherd: 20 samples at T 1.0 / top-p 0.9, five most frequent), the released knockout tournament for
WebArbiter-7B, SelectionARM's tool-call rendering, native accessibility-tree observation of the current viewport.
What changes for live OM2W (as in Zixian's webprm_om2w_table5.py, which wraps the same native environment):
  * task = one row of openwebrl/data/eval/online-mind2web.jsonl (intent = task_name, start = website); no login, no reset
  * the browser may load any origin, through the cluster HTTP proxy (compute nodes have no direct egress), with a
    desktop Chrome user agent (headless UA: 25/84 start pages HTTP 403 in her probe 359515)
  * no programmatic evaluator: the terminal state is scored once by the OM2W/AgentTrek judge (openwebrl/eval/
    reward_online_mind2web.py SYSTEM_PROMPT / USER_PROMPT, executed actions, final viewport screenshot, gpt-5.6-terra,
    seed 42) - the judge our OM2W column uses
Model calls (policy, judge) go to OpenAI through the proxy; served selectors stay on localhost. Cost is the ledger.
"""
import argparse
import ast
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wa_episode as we  # noqa: E402  (module-level setup: placeholder modules, native_episode, proxy-env scrub)

ne, wp = we.ne, we.wp
PROXY = os.environ.get('OM2W_PROXY', 'http://tillicum-dtn01:3128')
TASKS = Path('/gpfs/home/piotrt/OpenWebRL/openwebrl/data/eval/online-mind2web.jsonl')
JUDGE_SOURCE = Path('/gpfs/home/piotrt/OpenWebRL/openwebrl/eval/reward_online_mind2web.py')
JUDGE = os.environ.get('OM2W_JUDGE', 'gpt-5.6-terra')
DESKTOP_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36')


def judge_prompts():
    tree = ast.parse(JUDGE_SOURCE.read_text()); out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ('SYSTEM_PROMPT', 'USER_PROMPT'):
                    out[t.id] = ast.literal_eval(node.value)
    assert set(out) == {'SYSTEM_PROMPT', 'USER_PROMPT'}
    return out


def parse_verdict(text):  # identical to reward_online_mind2web._parse_verdict
    lower = (text or '').lower()
    if 'status:' not in lower:
        return None
    return 1.0 if 'success' in lower.split('status:', 1)[1] else 0.0


def live_browser_patches():
    """Allow every origin and route Chromium through the proxy with a desktop UA (launch kwargs only)."""
    ne.NativeBrowser._route = lambda self, route: route.continue_()
    original = ne._LaunchProxy.launch

    def launch(self, **kwargs):
        kwargs['proxy'] = dict(server=PROXY, bypass='127.0.0.1,localhost')
        kwargs['args'] = list(kwargs.get('args') or []) + [f'--user-agent={DESKTOP_UA}']
        return original(self, **kwargs)
    ne._LaunchProxy.launch = launch
    # Live ad-heavy pages often never fire "load" (flightaware: 30 s timeout at the start page, job 421546 task 1;
    # Zixian saw the same site never settle). Navigations wait for the document instead, unless a caller asks.
    from playwright.sync_api import Page
    goto = Page.goto

    def goto_dom(self, url, **kwargs):
        kwargs.setdefault('wait_until', 'domcontentloaded')
        return goto(self, url, **kwargs)
    Page.goto = goto_dom
    # Playwright's page.screenshot waits for web fonts, which never settle on ad-heavy live pages (flightaware, ign:
    # job 421546 tasks 4/8/9 timed out in the released ImageObservationProcessor, whose fallback then waits for a
    # "load" event that never comes). Capture the viewport through CDP instead, as Zixian's OM2W worker does
    # (webprm_om2w_table5.use_cdp_screenshots). Same pixels; no font wait.
    shot = Page.screenshot

    def screenshot(self, *, type='png', path=None, quality=None, **kwargs):
        if kwargs.get('full_page') or kwargs.get('clip') is not None or kwargs.get('mask'):
            return shot(self, type=type, path=path, quality=quality, **kwargs)
        params = dict(format='jpeg' if type == 'jpeg' else 'png')
        if quality is not None:
            params['quality'] = quality
        s = self.context.new_cdp_session(self)
        try:
            data = base64.b64decode(s.send('Page.captureScreenshot', params)['data'])
        finally:
            s.detach()
        if path is not None:
            Path(path).write_bytes(data)
        return data
    Page.screenshot = screenshot


def cdp_jpeg(page):
    s = page.context.new_cdp_session(page)
    try:
        return base64.b64decode(s.send('Page.captureScreenshot', dict(format='jpeg', quality=80))['data'])
    finally:
        s.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--method', choices=['no_search', 'arbiter', 'selectionarm'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--policy', default='gpt-4o-2024-08-06')
    ap.add_argument('--max-steps', type=int, default=30)
    ap.add_argument('--usd-cap', type=float, default=5.0)
    ap.add_argument('--viewport', default=os.environ.get('WA_VIEWPORT') or '1280x720')
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    row = json.loads(TASKS.read_text().splitlines()[a.index])
    task = dict(sites=['online_mind2web'], task_id=a.index, om2w_task_id=row['task_id'], require_login=False,
                storage_state=None, start_url=row['website'], geolocation=None, intent_template='', instantiation_dict={},
                intent=row['task_name'], require_reset=False,
                eval=dict(eval_types=['om2w_terminal_judge'], reference_answers={}, reference_url='', program_html=[]))
    we.RELAY = PROXY  # OpenAI-bound clients use the cluster proxy directly (no namespace here)
    # gpt-5.6-terra list price, USD per 1M input/output tokens (output includes reasoning tokens); override with
    # OM2W_JUDGE_PRICE="in,out" (a value of 0,0 means "use the default").
    price = tuple(float(x) for x in (os.environ.get('OM2W_JUDGE_PRICE') or '0,0').split(','))
    we.PRICES.setdefault(JUDGE, price if any(price) else (2.0, 12.0))
    ledger = we.Ledger(out / 'calls.jsonl', a.usd_cap)
    native = ne.load_native()
    we.install_adaptations(native, a.method, ledger, 'greedy')
    if a.method == 'arbiter' and not os.environ.get('OPENAI_API_URL'):
        sys.exit('arbiter needs OPENAI_API_URL (the served WebArbiter-7B)')
    we.chromium_launch_patch(); live_browser_patches()
    Spec = native.tools['engine'].ModelSpec
    try:
        knowledge = native.history.load_domain_knowledge('online_mind2web') or ''
    except Exception:
        knowledge = ''
    context = native.coordinator.RunContext.create(
        run_id=f'om2w{a.index}-{a.method}-{int(started)}', intent=task['intent'], domain='online_mind2web',
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
            index, trace = arm.select(candidates, browser.observations[-1], task['intent'])
            return index, [trace]
        if len(candidates) != 1:
            raise wp.EvaluationNotReady('no_search must yield exactly one admissible candidate')
        return 0, []
    adapter = wp.v8_decision_adapter('arbiter7b' if a.method in we.TOURNAMENT else a.method, select_control,
                                     lambda step, value: (out / f'selector-{step:03d}.json').write_text(json.dumps(wp.v8_plain(value), default=str)))
    prompts = judge_prompts()
    executed = []  # every action string the browser executed, in order (judge trajectory)
    original_execute = browser.execute

    async def execute(action_string):
        executed.append(action_string)
        return await original_execute(action_string)
    browser.execute = execute

    def evaluate(config, raw_trajectory, page, terminal):
        import httpx
        from openai import OpenAI
        # executed actions, in order (the released trajectory alternates state, action, state, ...)
        acts = list(executed)
        answer = (terminal or {}).get('answer') if isinstance(terminal, dict) else None
        lines = [f'Thought {i}: \nAction {i}: {x}' for i, x in enumerate(acts, 1)]
        n = len(acts) + 1
        lines.append(f'Thought {n}: \nAction {n}: ' + (f'Agent answered: {answer}' if answer else
                                                       f'Agent stopped: {terminal} (this is not a success verdict)'))
        jpeg = cdp_jpeg(page); (out / 'judge-final.jpg').write_bytes(jpeg)
        messages = [dict(role='system', content=prompts['SYSTEM_PROMPT']),
                    dict(role='user', content=[
                        dict(type='text', text=prompts['USER_PROMPT'].format(task=task['intent'], thoughts_and_actions='\n\n'.join(lines))),
                        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(), detail='high'))])]
        client = OpenAI(http_client=httpx.Client(proxy=PROXY, timeout=180))
        t0 = time.time()
        r = client.chat.completions.create(model=JUDGE, messages=messages, seed=42)
        text = r.choices[0].message.content or ''
        ledger.add(kind='direct_chat', role='judge', model=JUDGE, usage=we.usage_dict(r.usage), reply=text,
                   seconds=round(time.time() - t0, 1))
        verdict = parse_verdict(text)
        (out / 'om2w-judge.json').write_text(json.dumps(dict(model=r.model, raw=text, parsed=verdict, url=page.url,
                                                             usage=we.usage_dict(r.usage)), indent=1))
        if verdict is None:
            raise wp.EvaluationNotReady('unparseable OM2W judgment')
        return dict(score=verdict, success=bool(verdict))

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
            k = 'judge' if r.get('role') == 'judge' else 'policy'
            usd[k] = usd.get(k, 0.0) + (r.get('usd') or 0.0); usd['total'] += r.get('usd') or 0.0
            tok = usd.setdefault(k + '_tokens', [0, 0]); u = r.get('usage') or {}
            tok[0] += u.get('prompt_tokens', 0); tok[1] += u.get('completion_tokens', 0)
    summary = dict(index=a.index, om2w_task_id=row['task_id'], method=a.method, policy=a.policy, viewport=a.viewport,
                   valid=valid, success=int(bool(verdict.get('success'))) if valid else None, score=verdict.get('score'),
                   status=episode.get('status'), steps=len(episode.get('selected_actions', [])),
                   answer=(episode.get('terminal') or {}).get('answer'), error=episode.get('error'),
                   model_calls=ledger.calls, model_call_errors=ledger.errors, model_usd=round(ledger.usd, 4),
                   usd={k: (round(v, 4) if isinstance(v, float) else v) for k, v in usd.items()},
                   seconds=round(time.time() - started, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
