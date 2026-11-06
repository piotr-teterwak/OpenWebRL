#!/usr/bin/env python3
"""PRO-CUA (yifei-he/PRO-CUA @ 83f60cf) as a standalone agent, judged exactly like our other evals.

    pro_cua_eval.py --bench deepshop --model-url http://127.0.0.1:PORT/v1 --model-name NAME --out DIR [--limit N]
                    [--exclude allrecipe github] [--parallel 8]

Agent loop: their eval-time loop (WebVoyager/run_qwen3_og_parallel_playwright_data_collection_train.py, flags of
verl/scripts/pro_cua_4b.sh:208-223): system prompt prompts.QWEN3_SYSTEM_PROMPT; first user turn "Now given a task:
{ques}  Please interact with {web} and get the answer. \\nObservation: please analyze the attached screenshot and
give the Thought and Action. " + screenshot; later turns the observation text + screenshot; history clipped by their
utils.clip_message_and_obs(max_img_num=1); 1000x1000 viewport; greedy; 30 model calls; their parser, 0-1000 -> x/1000*992
coordinate map (smart_resize of 1000x1000 at factor 32), action executors and sleeps; error turns verbatim.
Judging: our benchmark judges (openwebrl.eval.reward_<bench>, settings from arm_eval.PROTOCOLS), given PRO-CUA's
answer text as a `done` call plus the episode's screenshots, so scores are comparable with our other evals.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import ast
import sys
import time

PRO = Path('/gpfs/scrubbed/piotrt/owrl/pro-cua/WebVoyager')
REPO = Path('/gpfs/home/piotrt/OpenWebRL')
sys.path[:0] = [str(PRO), str(REPO)]
import prompts as pro_prompts  # noqa: E402  (their QWEN3_SYSTEM_PROMPT, verbatim)
from cua_utils import CUA_KEY_TO_PLAYWRIGHT_KEY  # noqa: E402
from utils import clip_message_and_obs  # noqa: E402

CHROMIUM = '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'
TASK_FILES = {'deepshop': 'openwebrl/data/eval/deepshop.jsonl', 'webvoyager': 'openwebrl/data/eval/webvoyager_fara.jsonl',
              'online-mind2web': 'openwebrl/data/eval/online-mind2web.jsonl'}
WINDOW = 1000          # --window_width/--window_height 1000
RESIZED = 992          # smart_resize(1000, 1000, factor=32) -> 992 (their coordinate scale, RUN:446-458, 516-518)
MAX_ITER = 30          # --max_iter 30


# ---- verbatim from RUN:61-86 (parse_model_action) ----
def parse_model_action(model_res):
    if '<tool_call>' in model_res and '</tool_call>' in model_res:
        tool_call = model_res.split('<tool_call>', 1)[1].split('</tool_call>', 1)[0].strip()
        return json.loads(tool_call)
    action_match = re.search(r'Action:\s*(.+)', model_res, re.IGNORECASE | re.DOTALL)
    if not action_match:
        raise ValueError("No Action line found in model response.")
    action_text = action_match.group(1).strip()
    action_line = action_text.splitlines()[0].strip()
    type_match = re.match(r'type\((.*)\)\s*$', action_line, re.IGNORECASE)
    if type_match:
        text = ast.literal_eval(type_match.group(1).strip())
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}
    answer_match = re.match(r'(answer|finished)\((.*)\)\s*$', action_line, re.IGNORECASE)
    if answer_match:
        return {"name": "computer_use", "arguments": {"action": "answer"}}
    simple_action = action_line.lower().strip()
    if simple_action in {"wait", "goback"}:
        return {"name": "computer_use", "arguments": {"action": simple_action}}
    raise ValueError(f"Unsupported action format: {action_line}")


# ---- verbatim behaviour of RUN:288-361 executors ----
async def exec_action_click(x, y, page):
    await page.evaluate("""([x, y]) => { const elem = document.elementFromPoint(x, y);
        if (elem && 'target' in elem) { elem.setAttribute('target', '_self'); } }""", [x, y])
    await page.mouse.click(x, y)
    await asyncio.sleep(3)


async def exec_action_type(text, page):
    await page.keyboard.type(text)
    await page.keyboard.press('Enter')
    await asyncio.sleep(3)


async def perform_hotkey(page, keys):
    mapped = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(k, k) for k in keys]
    for k in mapped:
        await page.keyboard.down(k); await asyncio.sleep(1)
    for k in reversed(mapped):
        await page.keyboard.up(k); await asyncio.sleep(1)


def format_msg(it, init_msg, b64):
    if it == 1:
        return {'role': 'user', 'content': [{'type': 'text', 'text': init_msg + 'Please proceed with your Thought and Action.'},
                                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}
    return {'role': 'user', 'content': [{'type': 'text', 'text': 'Observation: please analyze the attached screenshot and give the Thought and Action. '},
                                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}


def answer_text(model_res, action):
    text = (action or {}).get('arguments', {}).get('text') if isinstance(action, dict) else None
    if text:
        return text
    m = re.search(r'(?:answer|finished)\((.*)\)\s*$', model_res, re.I | re.M)  # Action: answer('...') text form
    if m:
        try:
            return str(ast.literal_eval(m.group(1).strip()))
        except Exception:
            return m.group(1).strip()
    return ''


async def run_task(task, args, client, out):
    from playwright.async_api import async_playwright
    d = out / 'tasks' / re.sub(r'[^A-Za-z0-9_.-]', '_', str(task['task_id'])); d.mkdir(parents=True, exist_ok=True)
    rec = dict(task_id=task['task_id'], intent=task['prompt'], start_url=task['start_url'], steps=0, answer='',
               terminated='', errors=[], screenshots=[], started=time.time())
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=CHROMIUM,
                                           args=[f'--window-size={WINDOW},{WINDOW}', '--disable-extensions', '--disable-file-system'])
        try:
            context = await browser.new_context(viewport={'width': WINDOW, 'height': WINDOW}, device_scale_factor=1)
            page = await context.new_page()
            try:
                await page.goto(task['start_url'], timeout=180000); await asyncio.sleep(5)
            except Exception as e:
                rec.update(terminated='goto_failed', errors=[f'goto: {type(e).__name__}: {e}'[:300]]); return rec
            messages = [{'role': 'system', 'content': pro_prompts.QWEN3_SYSTEM_PROMPT}]
            init_msg = (f"Now given a task: {task['prompt']}  Please interact with {task['start_url']} and get the answer. \n"
                        "Observation: please analyze the attached screenshot and give the Thought and Action. ")
            fail_obs, it, coord = '', 0, None
            while it < MAX_ITER:
                it += 1
                path = d / f'screenshot{it}.png'
                try:
                    await page.screenshot(path=str(path), timeout=60000)
                except Exception as e:
                    rec['errors'].append(f'screenshot: {e}'[:200]); fail_obs = 'Internal error: Failed to capture screenshot.'; continue
                b64 = base64.b64encode(path.read_bytes()).decode(); rec['screenshots'].append(str(path))
                if not fail_obs:
                    messages = clip_message_and_obs(messages + [format_msg(it, init_msg, b64)], max_img_num=1)
                else:
                    messages.append({'role': 'user', 'content': fail_obs})
                messages = clip_message_and_obs(messages, 1)
                try:
                    resp = await client.chat.completions.create(model=args.model_name, messages=messages, temperature=0)
                    model_res = resp.choices[0].message.content
                except Exception as e:
                    rec['errors'].append(f'api: {type(e).__name__}: {e}'[:300]); continue
                rec['steps'] = it
                try:
                    action = parse_model_action(model_res)
                    if 'coordinate' in action['arguments']:
                        c = action['arguments']['coordinate']; coord = [c[0] / 1000 * RESIZED, c[1] / 1000 * RESIZED]
                    key = action['arguments']['action']
                except Exception as e:
                    messages.append({'role': 'assistant', 'content': model_res})
                    fail_obs = ("Format ERROR: The Action format is not correct, please follow the format: "
                                "Action: <action_type>(<action_inputs>)\n\n" f"Original model output:\n{model_res}")
                    continue
                messages.append({'role': 'assistant', 'content': model_res})
                if '"action": "answer"' in model_res:
                    rec.update(answer=answer_text(model_res, action) or model_res.strip(), terminated='answer',
                               answer_is_final_message=not answer_text(model_res, action)); break
                fail_obs = ''
                try:
                    if key in ('left_click', 'right_click', 'middle_click', 'double_click', 'triple_click'):
                        await exec_action_click(coord[0], coord[1], page)
                    elif key == 'mouse_move':
                        await page.mouse.move(coord[0], coord[1])
                    elif key == 'type':
                        await exec_action_type(action['arguments']['text'], page)
                    elif key == 'key':
                        await perform_hotkey(page, action['arguments']['keys'])
                    elif key == 'scroll':
                        await page.evaluate(f"window.scrollBy(0, {-action['arguments']['pixels']})"); await asyncio.sleep(3)
                    elif key == 'hscroll':
                        await page.evaluate(f"window.scrollBy({-action['arguments']['pixels']}, 0)"); await asyncio.sleep(3)
                    elif key == 'left_click_drag':  # their bug kept: start == end
                        await page.mouse.move(coord[0], coord[1]); await page.mouse.down(); await page.mouse.move(coord[0], coord[1]); await page.mouse.up()
                    elif key == 'wait':
                        await asyncio.sleep(10)
                    elif key == 'goback':
                        await page.go_back(); await asyncio.sleep(3)
                    elif key in ('answer', 'terminate'):
                        # no answer text (e.g. terminate status=success): their judge gets the whole final message
                        # (auto_eval_qwen.py:68-87), so pass it as the result response rather than an empty answer
                        rec.update(answer=answer_text(model_res, action) or model_res.strip(), terminated=key,
                                   answer_is_final_message=not answer_text(model_res, action)); break
                    await asyncio.sleep(3)
                except Exception as e:
                    rec['errors'].append(f'exec {key}: {type(e).__name__}: {e}'[:200])
                    fail_obs = 'The action cannot be executed. Please revise.'; await asyncio.sleep(3)
            else:
                rec['terminated'] = rec['terminated'] or 'max_iter'
            (d / 'messages.json').write_text(json.dumps([m if isinstance(m.get('content'), str) else
                {'role': m['role'], 'content': [x if x['type'] == 'text' else {'type': 'image_url', 'image_url': '<stripped>'} for x in m['content']]}
                for m in messages], indent=1))
        finally:
            await browser.close()
    rec['seconds'] = round(time.time() - rec.pop('started'), 1)
    return rec


async def judge(rec, args):
    """Our benchmark judge, unchanged: PRO-CUA's answer as a `done` call + the episode screenshots."""
    from slime.utils.types import Sample
    from openwebrl.arm_eval import PROTOCOLS
    from openwebrl.run_evaluate import _load_reward_func
    protocol = args.bench.replace('-', '_')
    reward_name, judge_model, judge_imgs, _ = PROTOCOLS[protocol]
    reward = _load_reward_func(reward_name)
    s = Sample(prompt='', response='<tool_call>\n' + json.dumps({'name': 'done', 'arguments': {'response': rec['answer']}}) + '\n</tool_call>')
    s.status = Sample.Status.COMPLETED
    s.metadata = dict(intent=rec['intent'], task_id=rec['task_id'], is_last_turn=True, turn_index=0,
                      full_image_list=[base64.b64encode(Path(p).read_bytes()).decode() for p in rec['screenshots']])
    ns = argparse.Namespace(judge_api_model=judge_model, judge_max_attached_imgs=judge_imgs, judge_api_mode='served',
                            judge_timeout_secs=180, judge_prompt_variant='default', hf_checkpoint=args.tokenizer)
    score = await reward(ns, [s])
    return score[0] if isinstance(score, list) else score, s.metadata.get('reward')


async def rejudge(a):
    rows = [json.loads(l) for l in (a.out / 'results.jsonl').open()]
    for r in rows:
        if r.get('judge_error') and r['terminated'] != 'goto_failed':
            d = a.out / 'tasks' / re.sub(r'[^A-Za-z0-9_.-]', '_', str(r['task_id']))
            r['screenshots'] = [str(d / f'screenshot{i}.png') for i in range(1, 99) if (d / f'screenshot{i}.png').exists()][:r['n_screenshots']]
            try:
                r['score'], r['judge'] = await judge(r, a); r.pop('judge_error')
            except Exception as e:
                r['judge_error'] = f'{type(e).__name__}: {e}'[:300]
            r.pop('screenshots'); r['valid'] = r.get('score') is not None
            print(json.dumps({k: r.get(k) for k in ('task_id', 'steps', 'terminated', 'score', 'judge_error')}), flush=True)
    (a.out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', choices=list(TASK_FILES), required=True)
    ap.add_argument('--model-url', required=True)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--tokenizer', default='/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT',
                    help='only used by our judge to parse the `done` call')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--exclude', nargs='*', default=[], help='task_id substrings to drop (their eval drops allrecipe, github)')
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--rejudge', action='store_true', help='only re-judge recorded rows whose judge call failed (saved screenshots)')
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(REPO / '.env', override=False)
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')  # as arm_eval
    from openwebrl.run_evaluate import load_tasks_from_jsonl
    import httpx
    from openai import AsyncOpenAI
    tasks = load_tasks_from_jsonl(str(REPO / TASK_FILES[a.bench]))
    tasks = [t for t in tasks if not any(x.lower() in str(t['task_id']).lower() for x in a.exclude)][:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['task_id'] for l in (a.out / 'results.jsonl').open()} if (a.out / 'results.jsonl').exists() else set()
    client = AsyncOpenAI(base_url=a.model_url, api_key='EMPTY', timeout=600, http_client=httpx.AsyncClient(trust_env=False, timeout=600))
    gate = asyncio.Semaphore(a.parallel)

    async def one(t):
        async with gate:
            rec = await run_task(t, a, client, a.out)
            try:
                rec['score'], rec['judge'] = await judge(rec, a) if rec['terminated'] != 'goto_failed' else (None, None)
            except Exception as e:
                rec['score'], rec['judge_error'] = None, f'{type(e).__name__}: {e}'[:300]
            rec['valid'] = rec['score'] is not None
            with (a.out / 'results.jsonl').open('a') as f:
                f.write(json.dumps({k: v for k, v in rec.items() if k != 'screenshots'} | dict(n_screenshots=len(rec['screenshots']))) + '\n')
            print(json.dumps({k: rec.get(k) for k in ('task_id', 'steps', 'terminated', 'score')}), flush=True)
    if a.rejudge:
        await rejudge(a)
    else:
        await asyncio.gather(*(one(t) for t in tasks if t['task_id'] not in done))
    rows = [json.loads(l) for l in (a.out / 'results.jsonl').open()]
    v = [r for r in rows if r['valid']]
    summary = dict(bench=a.bench, scheduled=len(tasks), recorded=len(rows), valid=len(v), successes=sum(r['score'] >= 1 for r in v),
                   success_rate_valid=(sum(r['score'] >= 1 for r in v) / len(v)) if v else None,
                   success_rate_all_scheduled=(sum(r['score'] >= 1 for r in v) / len(tasks)) if tasks else None)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == '__main__':
    asyncio.run(main())
