#!/usr/bin/env python3
"""Run one WebArena-Lite task with the released WebArbiter pipeline (one cell = task x method x repeat).

Runs inside the site's network namespace (wa_site.py exec OUT -- python wa_episode.py ...). Model calls leave through
wa_relay.py's proxy at 127.0.0.1:3128; the browser gets no proxy.

    wa_episode.py --task-id 0 --method no_search --out DIR [--policy gpt-4o-2024-08-06] [--max-steps 30]

Released code (YaoZ720/WebArbiterCode @ 764ec26) does the work: CoordinatorAgent.execute_step (sample, order by
confidence, tournament, stop detector), ScriptBrowserEnv observation/actions, state diff, the 30-step N/A fallback,
auto_login cookies and evaluation_harness. Through Zixian's native_episode.py wrapper (vendor/). Adaptations, each
listed in PROTOCOL.md: model egress proxy; fenced-JSON replies unwrapped; no-search prompt edited to one BLOCK
(Zixian's envelope v2); retired fuzzy judge gpt-4-1106-preview -> gpt-4.1-2025-04-14; Chromium path injected.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'vendor'), str(HERE)]
for k in [k for k in os.environ if k.lower() in ('http_proxy', 'https_proxy', 'all_proxy')]:
    del os.environ[k]  # the browser must reach only the namespace's sites; model clients get the relay explicitly
os.environ['DATASET'] = 'webarena'
for k, v in dict(SHOPPING='http://localhost:7770', SHOPPING_ADMIN='http://localhost:7780/admin',
                 REDDIT='http://localhost:9999', GITLAB='http://localhost:8023', MAP='http://localhost:3000',
                 WIKIPEDIA='http://localhost:8888', HOMEPAGE='http://localhost:4399').items():
    os.environ.setdefault(k, v)  # the released defaults (server/utils/url_mappings.py), made explicit

PYDEPS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/pydeps')  # nltk 3.9.1 + punkt_tab (string_match tokenization)
sys.path.append(str(PYDEPS)); os.environ.setdefault('NLTK_DATA', str(PYDEPS / 'nltk_data'))


class _Unavailable:
    """Placeholder for SDKs the released code imports but our OpenAI-only path never calls."""
    def __init__(self, name): self._name = name

    def __call__(self, *a, **k):
        if '()' in self._name:  # constructing a client at import is allowed; using it is not
            raise RuntimeError(f'{self._name} is not installed (unused provider/evaluator)')
        return _Unavailable(self._name + '()')

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _Unavailable(f'{self._name}.{attr}')


def _placeholder_modules():
    import importlib, types
    for name in ('dashscope', 'anthropic', 'text_generation', 'evaluate', 'google', 'google.auth',
                 'google.auth.transport', 'google.auth.transport.requests', 'google.oauth2',
                 'google.oauth2.service_account', 'google.api_core', 'google.api_core.exceptions', 'vertexai',
                 'vertexai.preview', 'vertexai.preview.generative_models'):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name); continue
        except ImportError:
            pass
        module = types.ModuleType(name); module.__path__ = []
        module.__getattr__ = (lambda attr, _n=name: (_ for _ in ()).throw(AttributeError(attr))
                              if attr.startswith('__') else _Unavailable(f'{_n}.{attr}'))
        sys.modules[name] = module


_placeholder_modules()
import native_episode as ne  # noqa: E402
import wa_protocol as wp  # noqa: E402

RELAY = 'http://127.0.0.1:3128'
# no_search: the released stop-detector override (D14). no_search_conf: the detector's stop competes with the
# policy's single action and the higher stated confidence executes (released `confidence` ordering; ties keep
# the policy action, which comes first), mirroring how the selector rows put the stop in their pool (D8).
SINGLE_ACTION = ('no_search', 'no_search_conf')
# The authors' own Table 5 rows: released 5-BLOCK policy + released knockout tournament; only the judge differs.
# 'WebArbiter' is the released routing name (openai_utils: model == "WebArbiter" -> OPENAI_API_URL, a local server).
TOURNAMENT = {'mini_prm': 'gpt-4o-mini', 'arbiter': 'WebArbiter'}
REWARD_SYSTEM = 'You are a skilled expert at evaluating assistant responses.'
# Replaces retired gpt-4-1106-preview (PROTOCOL D11): gpt-4-turbo-2024-04-09 is the same GPT-4 Turbo family; the
# no-search pilot (jobs 411181-411196) used gpt-4.1-2025-04-14 (Zixian's re-score choice, ablation A10).
JUDGE = os.environ.get('WA_JUDGE', 'gpt-4.1-2025-04-14')
# OpenAI retired the whole GPT-4 Turbo family mid-step-1 (2026-11-02: gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4-0613 all
# 404 "deprecated"; none of step 1's turbo calls ever succeeded), so jobs submitted with it fall back to gpt-4.1.
if JUDGE in ('gpt-4-turbo-2024-04-09', 'gpt-4-turbo', 'gpt-4-0613'):
    JUDGE = 'gpt-4.1-2025-04-14'
PRICES = {'WebArbiter': (0.0, 0.0), 'gpt-4o-2024-08-06': (2.5, 10.0), 'gpt-4o-mini': (0.15, 0.6), 'gpt-4o-mini-2024-07-18': (0.15, 0.6),
          'gpt-4.1-2025-04-14': (2.0, 8.0), 'gpt-4-turbo-2024-04-09': (10.0, 30.0)}  # USD per 1M input/output tokens
TASKS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26/viewer/config_files/wa/test_webarena_lite.raw.json')


class Ledger:
    """Every model call, appended to calls.jsonl; a per-episode USD cap enforced at each step."""
    def __init__(self, path, cap):
        self.path, self.cap, self.usd, self.calls, self.errors = path, cap, 0.0, 0, 0

    def error(self, **row):
        self.errors += 1
        with open(self.path, 'a') as f:
            f.write(json.dumps(dict(kind='error', **row), default=str) + '\n')

    def add(self, **row):
        # Cost and call counts come from the actual API calls (direct_chat); engine rows label what each call was.
        if row.get('kind') == 'direct_chat':
            u = row.get('usage') or {}
            pin, pout = PRICES.get(row.get('model'), (0.0, 0.0))
            row['usd'] = (u.get('prompt_tokens', 0) * pin + u.get('completion_tokens', 0) * pout) / 1e6
            self.usd += row['usd']; self.calls += 1
        with open(self.path, 'a') as f:
            f.write(json.dumps(row, default=str) + '\n')

    def assert_healthy(self):
        if self.usd > self.cap:
            raise RuntimeError(f'Episode model spend ${self.usd:.2f} exceeded the ${self.cap} cap')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def usage_dict(usage):
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {k: getattr(usage, k, 0) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def install_adaptations(native, method, ledger, reward_decoding='greedy'):
    from webarbiter.src.llms.providers import openai_utils
    import httpx
    # 1. Egress: give OpenAI-bound clients the relay proxy; local endpoints (served critics) stay direct.
    for name in ('OpenAI', 'AsyncOpenAI'):
        original = getattr(openai_utils, name)

        def factory(*a, _original=original, _async=(name == 'AsyncOpenAI'), **k):
            base = k.get('base_url') or ''
            if not base.startswith(('http://127.0.0.1', 'http://localhost')):
                k['http_client'] = (httpx.AsyncClient if _async else httpx.Client)(proxy=RELAY, timeout=180)
            return _original(*a, **k)
        setattr(openai_utils, name, factory)
    # 2. Retired fuzzy judge -> JUDGE, and a ledger row for every direct chat completion (evaluator calls).
    released_chat = openai_utils.generate_from_openai_chat_completion

    def chat(messages, model, *a, **k):
        swapped = JUDGE if model == 'gpt-4-1106-preview' else model
        try:
            answer, usage = released_chat(messages, swapped, *a, **k)
        except Exception as e:
            ledger.error(where='direct_chat', model=swapped, error=f'{type(e).__name__}: {e}'); raise
        ledger.add(kind='direct_chat', model=swapped, requested_model=model, messages=messages, reply=answer,
                   usage=usage_dict(usage))
        return answer, usage
    openai_utils.generate_from_openai_chat_completion = chat
    # 3. LMEngine: log every agent call; unwrap a whole-reply ```json fence (released parser needs bare JSON);
    #    no-search edits the policy prompt to one BLOCK (Zixian envelope v2).
    Engine = native.tools['engine'].LMEngine
    released_sync = Engine.generate_text  # the released generate_text_async runs this in an executor

    def adapt(self, system_text, user_text, model):
        role = ('policy' if 'Generate 5 diverse, PARALLEL action candidates' in system_text else
                'reward' if system_text.lstrip().startswith(REWARD_SYSTEM) else 'helper')
        if role == 'policy' and (method in SINGLE_ACTION or os.environ.get('WA_SAMPLING') == 'shepherd'):  # A17: one action per sample
            system_text = wp.single_action_system_prompt(system_text, version=wp.ENVELOPE_NO_SEARCH_PROMPT)
        if role == 'policy' and os.environ.get('WA_POLICY_STOP') in ('1', 'full'):  # ablation A12, see PROTOCOL 2026-11-03
            system_text = wp.policy_actions_system_prompt(system_text, 'full' if os.environ.get('WA_POLICY_STOP') == 'full' else 'stop')
        spec = model or self._default_model
        if role == 'reward' and reward_decoding == 'greedy' and spec is not None:  # PROTOCOL D6 (ablation A6: released)
            spec = type(spec)(provider=spec.provider, model=spec.model, mode=spec.mode,
                              gen_config=dict(temperature=0.0, top_p=1.0, max_tokens=2048, context_length=4096))
            model = spec
        return role, system_text, getattr(spec, 'model', 'gpt-4o-mini'), model

    def record(role, model_name, system_text, user_text, raw, usage):
        text = wp.strip_json_fence(raw)
        if role == 'policy' and os.environ.get('WA_TYPE_UNQUOTE') == '1':  # ablation A13
            text = wp.unwrap_type_content(text)
        ledger.add(kind='engine', role=role, model=model_name, system=system_text, user=user_text, reply=raw,
                   unwrapped_fence=(text != raw), usage=usage_dict(usage))
        if role == 'policy' and method in SINGLE_ACTION:
            wp.validate_single_action_response(text)
        return text

    def shepherd_sample(system_text, user_text, name, spec):
        """A17 (Web-Shepherd App. F.2): 20 single-action samples from one chat call (n=20, so the ~6k-token input is
        billed once) at the released policy decoding (T 1.0, top-p 0.9, 2,048 tokens), aggregated by
        wp.shepherd_candidates into the released BLOCKS format; the released parser, element filter, confidence rerank
        and tournament then run unchanged on the 5 most frequent actions."""
        gen = getattr(spec, 'gen_config', None) or {}
        messages = [{'role': 'system', 'content': system_text}, {'role': 'user', 'content': user_text}]
        client = openai_utils.OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # relay-proxied (adaptation 1)
        t0 = time.time()
        try:
            response = client.chat.completions.create(model=name, messages=messages, n=wp.SHEPHERD_SAMPLES,
                                                      temperature=gen.get('temperature', 1.0), top_p=gen.get('top_p', 0.9),
                                                      max_tokens=gen.get('max_tokens', 2048))
        except Exception as e:
            ledger.error(where='direct_chat', model=name, error=f'{type(e).__name__}: {e}'); raise
        replies = [c.message.content or '' for c in response.choices]
        usage = usage_dict(response.usage)
        raw, stats = wp.shepherd_candidates(replies)
        ledger.add(kind='direct_chat', model=name, requested_model=name, messages=messages, reply=replies, usage=usage,
                   n=wp.SHEPHERD_SAMPLES, shepherd=stats, seconds=round(time.time() - t0, 1))
        return raw, usage

    def generate_text(self, system_text, user_text, model=None):
        role, system_text, name, model = adapt(self, system_text, user_text, model)
        if role == 'policy' and os.environ.get('WA_SAMPLING') == 'shepherd':  # ablation A17
            raw, usage = shepherd_sample(system_text, user_text, name, model)
            return record(role, name, system_text, user_text, raw, usage), usage
        if role == 'reward' and method == 'arbiter' and os.environ.get('WA_ARBITER_FORMAT') == 'browsergym':
            user_text = wp.to_browsergym(user_text)  # ablation A11: WebArbiter reads its training (WebPRMBench) format
        if role == 'reward' and method == 'arbiter' and os.environ.get('WA_ARBITER_FORMAT') == 'training':
            system_text, user_text = wp.training_judge_prompt(system_text, user_text)  # ablation A14: paper Appendix D prompt
        try:
            raw, usage = released_sync(self, system_text, user_text, model)
        except Exception as e:
            ledger.error(where='engine', role=role, model=name, error=f'{type(e).__name__}: {e}'); raise
        return record(role, name, system_text, user_text, raw, usage), usage

    Engine.generate_text = generate_text
    # 4. A12 'full': released id-based parser extended with report_infeasible / select_option / noop (wa_protocol).
    if os.environ.get('WA_POLICY_STOP') == 'full':
        released_parse = native.actions.create_id_based_action

        def parse(action_str):
            extra = wp.parse_extra_action(action_str, native.actions)
            return extra if extra is not None else released_parse(action_str)
        native.actions.create_id_based_action = parse


def chromium_launch_patch():
    """Playwright 1.44 expects chromium-1117; inject the pinned chromium-1208 path for the released auto_login."""
    from playwright.sync_api._generated import BrowserType
    original = BrowserType.launch

    def launch(self, *a, **k):
        k.setdefault('executable_path', str(ne.CHROMIUM))
        return original(self, *a, **k)
    BrowserType.launch = launch


def authenticate(sites, folder):
    """Released browser_env/auto_login.renew_comb, run here inside the namespace."""
    folder.mkdir(parents=True, exist_ok=True)
    from browser_env import auto_login
    auto_login.renew_comb(list(sites), auth_folder=str(folder))
    return folder / f"{'.'.join(sites)}_state.json"


def resolve(raw, storage_state):
    from webarbiter.src.server.utils.url_mappings import replace_url_placeholders
    task = deepcopy(raw)
    task['start_url'] = replace_url_placeholders(task['start_url'])
    task['storage_state'] = str(storage_state) if storage_state else None
    return task


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--method', choices=list(SINGLE_ACTION) + list(TOURNAMENT) + ['selectionarm'], required=True)
    ap.add_argument('--reward-decoding', choices=['greedy', 'released'], default='greedy')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--policy', default='gpt-4o-2024-08-06')
    ap.add_argument('--max-steps', type=int, default=30)
    ap.add_argument('--usd-cap', type=float, default=3.0)
    ap.add_argument('--viewport', default=os.environ.get('WA_VIEWPORT') or '1024x2490')  # released service default (run_management.py:172); WA_VIEWPORT=1280x720 for the SelectionARM rows
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    raw = next(t for t in json.loads(TASKS.read_text()) if t['task_id'] == a.task_id)
    # Released Lite config bug: task 81 (WebArena 349) has must_include "yjlou" as a bare string, and the released
    # StringEvaluator asserts a list, so every episode crashed at evaluation. Original WebArena has ["yjlou"].
    ref = raw['eval'].get('reference_answers') or {}
    for k in ('must_include', 'must_exclude'):
        if isinstance(ref.get(k), str):
            ref[k] = [ref[k]]
    sites = raw['sites']
    if len(sites) != 1:
        sys.exit('single-site tasks only')
    ledger = Ledger(out / 'calls.jsonl', a.usd_cap)
    native = ne.load_native()
    install_adaptations(native, a.method, ledger, a.reward_decoding)
    if a.method == 'arbiter' and not os.environ.get('OPENAI_API_URL'):
        sys.exit('arbiter needs OPENAI_API_URL (the served WebArbiter-7B, reachable inside the namespace)')
    chromium_launch_patch()
    storage = authenticate(sites, out / 'auth') if raw.get('require_login') else None
    task = resolve(raw, storage)
    Spec = native.tools['engine'].ModelSpec
    context = native.coordinator.RunContext.create(
        run_id=f'{a.task_id}-{a.method}-{int(started)}', intent=task['intent'], domain=sites[0],
        domain_knowledge=native.history.load_domain_knowledge(sites[0]) or '',
        policy_model=Spec('openai', a.policy, 'chat'),
        reward_model=Spec('openai', TOURNAMENT[a.method], 'chat') if a.method in TOURNAMENT else None)
    width, height = (int(x) for x in a.viewport.split('x'))
    origin = os.environ[{'shopping': 'SHOPPING', 'shopping_admin': 'SHOPPING_ADMIN', 'reddit': 'REDDIT',
                         'gitlab': 'GITLAB', 'map': 'MAP'}[sites[0]]].split('/admin')[0]
    browser = ne.NativeBrowser(native, out / 'browser', viewport=dict(width=width, height=height),
                               allowed_origins=[origin], compatibility_geometry=True)

    arm = None
    if a.method == 'selectionarm':  # Piotr's SelectionARM picks among the released pool (wa_selectionarm.py)
        import wa_selectionarm
        if not os.environ.get('WA_SELECTOR_URL'):
            sys.exit('selectionarm needs WA_SELECTOR_URL (the served SelectionARM, reachable inside the namespace)')
        arm = wa_selectionarm.SelectionARM(os.environ['WA_SELECTOR_URL'], dict(width=width, height=height), out / 'selectionarm.jsonl',
                                         crop=int(os.environ.get('WA_ARM_CROP') or 0) or None)

    async def select_control(candidates, inputs, ctx):
        if a.method == 'selectionarm':
            index, trace = arm.select(candidates, browser.observations[-1], task['intent'])
            return index, [trace]
        if a.method == 'no_search':
            if len(candidates) != 1:
                raise wp.EvaluationNotReady('no_search must yield exactly one admissible candidate')
            return 0, []
        # no_search_conf: pool = [policy action] (+ detector stop); stable max over stated confidence
        conf = [float(c.get('confidence') or 0.0) for c in candidates]
        best = max(range(len(candidates)), key=lambda i: (conf[i], -i))
        return best, [dict(confidences=conf, rule='max stated confidence; ties to the policy action')]
    # Tournament rows: the stop-detector action joins the released tournament's pool (PROTOCOL D8).
    adapter = wp.v8_decision_adapter('arbiter7b' if a.method in TOURNAMENT else a.method, select_control,
                                     lambda step, value: (out / f'selector-{step:03d}.json').write_text(json.dumps(wp.v8_plain(value), default=str)))

    def evaluate(config, raw_trajectory, page, terminal):
        from webarbiter.src.evaluation_harness import evaluators, helper_functions
        from webarbiter.src.evaluation_harness.evaluators import evaluate_trajectory
        # Released evaluators.py imports only PseudoPage/llm_fuzzy_match/llm_ua_match, so program_html checks that
        # eval() a helper (reddit_get_post_url, shopping_get_latest_order_url, gitlab_get_project_memeber_role; 14
        # tasks) raise NameError. Restore WebArena's original import of the released helper_functions module.
        for name in dir(helper_functions):
            if not name.startswith('_') and callable(getattr(helper_functions, name)) and not hasattr(evaluators, name):
                setattr(evaluators, name, getattr(helper_functions, name))
        # The released StringEvaluator.clean_answer dropped WebArena's leading `answer.strip()`, so an exact_match
        # against page text with surrounding whitespace (e.g. task 100's `.cover-status` lastChild, " Enjoying life")
        # fails even when the task was done. Restore the original WebArena semantics (2026-11-03; cells before this
        # date were graded without the strip).
        def clean_answer(answer):
            answer = str(answer).strip()
            if (answer.startswith("'") and answer.endswith("'")) or (answer.startswith('"') and answer.endswith('"')):
                answer = answer[1:-1]
            return answer.lower()
        evaluators.StringEvaluator.clean_answer = staticmethod(clean_answer)
        # Released must_include tokenizes whenever the reference is a single word; the original only when it is a
        # single character (so that e.g. "0" cannot match inside "10"). Restore the original rule (offline re-grade:
        # 4 step-1 cells on CMS task 32 differ).
        def must_include(ref, pred, tokenize=None):
            from nltk.tokenize import word_tokenize
            clean_ref, clean_pred = clean_answer(ref), clean_answer(pred)
            if len(clean_ref) == 1:
                return float(clean_ref in word_tokenize(clean_pred))
            return float(clean_ref in clean_pred)
        evaluators.StringEvaluator.must_include = staticmethod(must_include)
        # The released detailed-verdict path reads self.url_service on EvaluatorComb, which never sets it, so every
        # program_html verdict carried "'EvaluatorComb' object has no attribute 'url_service'" instead of the value the
        # grader read. Give it the module's service so actual_value is recorded (scores were never affected).
        from webarbiter.src.evaluation_harness import url_service as _us
        if not hasattr(evaluators.EvaluatorComb, 'url_service'):
            evaluators.EvaluatorComb.url_service = _us.get_url_service()
        trajectory = wp.normalize_trajectory(raw_trajectory, native.actions.ActionTypes)
        result = evaluate_trajectory(trajectory, config, page, detailed=True)
        return dict(score=result.score, success=bool(result.success),
                    details=wp.v8_plain(getattr(result, 'evaluator_details', None)))

    episode = asyncio.run(ne.run_episode(task=task, context=context, browser=browser, completion_scope=ledger,
                                         evaluate=evaluate, evidence_root=out / 'episode', max_steps=a.max_steps,
                                         native=native, decision_adapter=adapter))
    verdict = episode.get('verdict') or {}
    # Infrastructure failures (any failed model call, or none made) never count as task failures.
    valid = bool(episode.get('valid')) and ledger.errors == 0 and ledger.calls > 0
    summary = dict(task_id=a.task_id, site=sites[0], method=a.method, policy=a.policy, valid=valid,
                   reward_model=TOURNAMENT.get(a.method), reward_decoding=a.reward_decoding if a.method in TOURNAMENT else None,
                   invalid_reason=None if valid else ('model_call_errors' if ledger.errors else 'no_model_calls'
                                                      if ledger.calls == 0 else 'episode_invalid'),
                   success=int(bool(verdict.get('success'))) if valid else None, score=verdict.get('score'),
                   status=episode.get('status'), steps=len(episode.get('selected_actions', [])),
                   answer=(episode.get('terminal') or {}).get('answer'), error=episode.get('error'),
                   model_calls=ledger.calls, model_call_errors=ledger.errors, model_usd=round(ledger.usd, 4), seconds=round(time.time() - started, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
