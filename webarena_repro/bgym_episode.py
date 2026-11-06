#!/usr/bin/env python3
"""One WebArena-Lite task in the BrowserGym harness Web-Shepherd used (arXiv 2505.15277 §6.2, App. F.2).

    bgym_episode.py --task-id 0 --method arbiter|no_search --out DIR [--policy gpt-4o-2024-08-06] [--max-steps 30]

Runs inside the site's network namespace exactly like wa_episode.py (sites, fresh copy per task, auto-login cookies,
model egress through the relay), but the agent side is BrowserGym's, not the released WebArbiter code:
  * environment: browsergym-core 0.14.3 BrowserEnv (default 1280x720 viewport); a thin LiteTask opens the Lite task's
    start URL with the released auto-login storage state; BrowserGym itself puts the intent into the chat on reset;
  * policy: Web-Shepherd's released demo BrowserAgent prompt, verbatim (sources/shepherd-demo.py, loaded through
    Zixian's load_shepherd_actor): chat, open tabs, visible accessibility tree, action space description
    (HighLevelActionSet chat+tab+nav+bid+infeas, single action), past responses, last action error;
  * no_search: one response per step with the demo's default sampling (the paper's no-search baseline);
  * arbiter (= A18 in this harness): 20 samples in one n=20 call at T 1.0 / top-p 0.9, actions grouped by their code
    string, the 5 most frequent kept (frequency order), judged by WebArbiter-7B in its native WebPRMBench prompt (the
    benchmark's own template, BrowserGym action syntax) with the released knockout bracket (consecutive pairs, odd one
    carries over, unparseable verdict -> first response wins), greedy; WA_MERGE_STOP=1 merges stop/infeasible actions
    into one candidate (A20);
  * stop: the policy's send_msg_to_user / report_infeasible ends the episode (infeasible -> answer N/A); 30 steps, then
    answer N/A (as A18);
  * grading: the released WebArena-Lite evaluation harness on the live BrowserGym page, with the same restorations as
    wa_episode.py (original clean_answer / must_include, helper imports, gpt-4.1 fuzzy judge).
Writes summary.json in wa_episode.py's format (so wa_report.py / wa_site_status.py work), steps.jsonl and calls.jsonl.
"""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'vendor'), str(HERE)]
import wa_episode as we  # noqa: E402  (environment, released-code loading, auth, ledger, grading pieces)

ne, wp = we.ne, we.wp
SHEPHERD = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009')
BENCH = SHEPHERD / 'WebPRMBench.jsonl'
CODE = re.compile(r'```(?:python)?\s*(.*?)```', re.S)
STOP = ('send_msg_to_user', 'report_infeasible')


def judge_template():
    """WebArbiter's native prompt pieces, cut from the first WebPRMBench WebArena row (no reconstruction)."""
    for line in BENCH.open():
        if '"webarena"' in line:
            p = json.loads(line)['prompt']; break
    head = p[:p.index('#### Intent ####')]
    tail = p[p.index('### Output Instructions ###'):]
    note = lambda sec: re.search(rf'#### {sec} ####\n(Note:[^\n]*|The URL[^\n]*)\n', p).group(1)
    return head, tail, note('AXTREE'), note('Trajectory'), note('current url')


def judge_prompt(T, intent, axtree, traj, start, url, r1, r2):
    head, tail, n_ax, n_tr, n_url = T
    resp = lambda k, r: f"[The Begin of Response {k}]\n\nTHOUGHT:\n{r['thought']}\n\nACTION:\n{r['action']}\n\n[The End of Response {k}]"
    return (f"{head}#### Intent ####\n{intent}\n\n#### AXTREE ####\n{n_ax}\n{axtree}\n\n#### Trajectory ####\n{n_tr}\n"
            f"{json.dumps(traj, indent=2, ensure_ascii=False)}\n\n#### start url ####\n{start}\n\n#### current url ####\n{n_url}\n{url}\n\n"
            f"#### Assistant Responses ####\n{resp(1, r1)}\n\n{resp(2, r2)}\n\n\n{tail}")


def split(text):
    blocks = CODE.findall(text or '')
    code = blocks[-1].strip() if blocks else (text or '').strip()
    thought = CODE.sub('', text or '').strip() if blocks else ''
    return thought, code


def stop_answer(code):
    m = re.match(r'\s*(send_msg_to_user|report_infeasible)\((.*)\)\s*$', code, re.S)
    if not m:
        return None
    if m.group(1) == 'report_infeasible':
        return 'N/A'
    try:
        return str(ast.literal_eval(m.group(2).strip()))
    except Exception:
        return m.group(2).strip().strip('"\'')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task-id', type=int, required=True); ap.add_argument('--method', choices=['arbiter', 'no_search'], required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--policy', default='gpt-4o-2024-08-06')
    ap.add_argument('--max-steps', type=int, default=30); ap.add_argument('--usd-cap', type=float, default=4.0)
    ap.add_argument('--samples', type=int, default=20); ap.add_argument('--keep', type=int, default=5)
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False); started = time.time()
    raw = next(t for t in json.loads(we.TASKS.read_text()) if t['task_id'] == a.task_id)
    ref = raw['eval'].get('reference_answers') or {}
    for k in ('must_include', 'must_exclude'):
        if isinstance(ref.get(k), str):
            ref[k] = [ref[k]]
    sites = raw['sites']
    ledger = we.Ledger(out / 'calls.jsonl', a.usd_cap)
    native = ne.load_native(); we.install_adaptations(native, 'no_search', ledger)  # relay + fuzzy-judge swap for grading
    we.chromium_launch_patch()
    storage = we.authenticate(sites, out / 'auth') if raw.get('require_login') else None
    task = we.resolve(raw, storage)

    import httpx, openai
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import AbstractBrowserTask
    sys.path.insert(0, str(SHEPHERD / 'code-v2'))
    from openwebrl.webprm_shepherd_harness import load_shepherd_actor
    action_set = HighLevelActionSet(subsets=['chat', 'tab', 'nav', 'bid', 'infeas'], strict=False, multiaction=False, demo_mode='off')
    actor = load_shepherd_actor(SHEPHERD / 'sources/shepherd-demo.py', action_set)

    class LiteTask(AbstractBrowserTask):
        def __init__(self, seed, start_url, intent):
            super().__init__(seed); self.start_url, self.intent = start_url, intent

        @classmethod
        def get_task_id(cls):
            return 'webarena_lite'

        def setup(self, page):
            page.goto(self.start_url, timeout=120000); return self.intent, {}

        def teardown(self):
            pass

        def validate(self, page, chat_messages):
            return 0.0, False, '', {}

    env = BrowserEnv(task_entrypoint=LiteTask, task_kwargs=dict(start_url=task['start_url'], intent=task['intent']), headless=True,
                     pw_chromium_kwargs=dict(executable_path=str(ne.CHROMIUM)),
                     pw_context_kwargs=(dict(storage_state=str(storage)) if storage else {}), action_mapping=action_set.to_python_code)
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'], http_client=httpx.Client(proxy=we.RELAY, timeout=300))
    arb = httpx.Client(timeout=600, trust_env=False); arb_url = os.environ.get('OPENAI_API_URL', '')
    T = judge_template() if a.method == 'arbiter' else None
    merge = os.environ.get('WA_MERGE_STOP') == '1'
    history, traj, steps_log = [], [], []
    answer, status, error = None, 'turn_limit', None
    try:
        obs, info = env.reset(seed=0)
        for step in range(a.max_steps):
            ledger.assert_healthy()
            pre = actor.preprocess(obs)
            msgs = actor.messages(pre, history, model=a.policy)
            kw = dict(model=a.policy, messages=msgs)
            if a.method == 'arbiter':
                kw.update(n=a.samples, temperature=1.0, top_p=0.9)
            try:
                resp = client.chat.completions.create(**kw)
            except Exception as e:
                ledger.error(where='policy', error=f'{type(e).__name__}: {e}'); raise
            u = we.usage_dict(resp.usage)
            ledger.add(kind='direct_chat', model=a.policy, requested_model=a.policy, usage=u, n=kw.get('n', 1), role='policy')
            texts = [c.message.content or '' for c in resp.choices]
            groups, order = {}, []
            for t in texts:
                th, code = split(t)
                key = 'STOP' if (merge and code.startswith(STOP)) else code
                if key not in groups:
                    groups[key] = dict(text=t, thought=th, action=code, freq=0, variants={}); order.append(key)
                groups[key]['freq'] += 1; groups[key]['variants'][code] = groups[key]['variants'].get(code, 0) + 1
            if merge and 'STOP' in groups:  # merged stop is worded as its most frequent variant
                g = groups['STOP']; best = max(g['variants'], key=g['variants'].get)
                g.update(action=best, text=next(t for t in texts if split(t)[1] == best), thought=split(next(t for t in texts if split(t)[1] == best))[0])
            ranked = sorted(order, key=lambda k: (-groups[k]['freq'], order.index(k)))[:a.keep]
            cands = [groups[k] for k in ranked]
            comparisons = []
            if a.method == 'arbiter' and len(cands) > 1:
                q = list(range(len(cands)))
                while len(q) > 1:
                    nxt = []
                    for i in range(0, len(q), 2):
                        if i + 1 >= len(q):
                            nxt.append(q[i]); continue
                        x, y = cands[q[i]], cands[q[i + 1]]
                        prompt = judge_prompt(T, task['intent'], pre['axtree_txt'], traj, task['start_url'], obs['url'], x, y)
                        r = arb.post(f'{arb_url}/completions', json=dict(model='WebArbiter', prompt=prompt, max_tokens=2048, temperature=0.0, top_p=1.0)).json()
                        text = r['choices'][0]['text']; m = re.search(r'<Answer>\s*Response\s*([12])\s*</Answer>', text)
                        win = q[i + 1] if (m and m.group(1) == '2') else q[i]
                        comparisons.append(dict(a=q[i], b=q[i + 1], verdict=m.group(1) if m else None, winner=win,
                                                prompt_tokens=(r.get('usage') or {}).get('prompt_tokens')))
                        ledger.add(kind='engine', role='reward', model='WebArbiter', usage=r.get('usage') or {})
                        nxt.append(win)
                    q = nxt
                pick = q[0]
            else:
                pick = 0
            chosen = cands[pick]
            steps_log.append(dict(step=step, url=obs['url'], candidates=[(c['action'], c['freq']) for c in cands], pick=pick,
                                  comparisons=comparisons, samples=len(texts)))
            with (out / 'steps.jsonl').open('a') as f:
                f.write(json.dumps(steps_log[-1], ensure_ascii=False) + '\n')
            obs, reward, terminated, truncated, info = env.step(chosen['text'])
            history.append(chosen['text']); traj.append(dict(thought=chosen['thought'], action=chosen['action']))
            ans = stop_answer(chosen['action'])
            if ans is not None:
                answer, status = ans, 'terminal'; break
        if answer is None:
            answer, status = 'N/A', 'max_steps_reached_na_fallback'
        # grade on the live page with the released Lite harness (as wa_episode.py)
        from webarbiter.src.evaluation_harness import evaluators, helper_functions, url_service as _us
        from webarbiter.src.evaluation_harness.evaluators import evaluate_trajectory
        for name in dir(helper_functions):
            if not name.startswith('_') and callable(getattr(helper_functions, name)) and not hasattr(evaluators, name):
                setattr(evaluators, name, getattr(helper_functions, name))

        def clean_answer(x):
            x = str(x).strip()
            if (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
                x = x[1:-1]
            return x.lower()
        evaluators.StringEvaluator.clean_answer = staticmethod(clean_answer)

        def must_include(r, p, tokenize=None):
            from nltk.tokenize import word_tokenize
            cr, cp = clean_answer(r), clean_answer(p)
            return float(cr in word_tokenize(cp)) if len(cr) == 1 else float(cr in cp)
        evaluators.StringEvaluator.must_include = staticmethod(must_include)
        if not hasattr(evaluators.EvaluatorComb, 'url_service'):
            evaluators.EvaluatorComb.url_service = _us.get_url_service()
        act = native.actions.create_send_message_to_user_action(answer)
        trajectory = wp.normalize_trajectory([dict(observation={'text': ''}, info=dict(fail_error='', observation_metadata={})), act],
                                             native.actions.ActionTypes)
        result = evaluate_trajectory(trajectory, task, env.page, detailed=True)
        verdict = dict(score=result.score, success=bool(result.success))
    except Exception as e:
        error, verdict = f'{type(e).__name__}: {e}', {}
        import traceback; traceback.print_exc()
    finally:
        try:
            env.close()
        except Exception:
            pass
    valid = error is None and ledger.errors == 0 and ledger.calls > 0
    summary = dict(task_id=a.task_id, site=sites[0], method=a.method, policy=a.policy, valid=valid, harness='browsergym-0.14.3',
                   reward_model='WebArbiter' if a.method == 'arbiter' else None, reward_decoding='greedy' if a.method == 'arbiter' else None,
                   invalid_reason=None if valid else ('model_call_errors' if ledger.errors else 'episode_error' if error else 'no_model_calls'),
                   success=int(bool(verdict.get('success'))) if valid else None, score=verdict.get('score'), status=status,
                   steps=len(steps_log), answer=answer, error=error, model_calls=ledger.calls, model_call_errors=ledger.errors,
                   model_usd=round(ledger.usd, 4), seconds=round(time.time() - started, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == '__main__':
    main()
