#!/usr/bin/env python3
"""WebDreamer (Gu et al., arXiv 2411.06559) as an arm_eval `scalar` selector over our SFT actor's candidates.

    serve_webdreamer_selector.py --dreamer-url http://127.0.0.1:P1/v1 --dreamer-name Dreamer-7B \
        --value-url http://127.0.0.1:P2/v1 --value-name Qwen3-VL-4B-Instruct --port 19101 [--log calls.jsonl] \
        [--n 10] [--sims 1]

Per candidate action (OSU-NLP-Group/WebDreamer @ main, prompts verbatim from world_model.py / simulation_scoring.py):
  1. action description (option 1, user 2026-11-05): the action in words from our tool call plus the actor's own
     reasoning, which names the target ("click — <thought>"); WebDreamer's world model expects "operation type and the
     element to operate", and our actor's tool calls carry only a screen point
  2. world model: Dreamer-7B (osunlp/Dreamer-7B, Qwen2-VL-7B) with WebWorldModel.state_change_prediction_in_website,
     format "change" (system prompt + " Please predict the changes after {description}." + current screenshot),
     temperature 1.0 as on the model card; the imagination string is multiple_step_change_prediction's with k = 0
     ("Proposed New Action: ... Predicted new webpage: ...": one predicted step, no imagined follow-up actions)
  3. value: evaluate_simulation_inner's system prompt and user message (intent, action history, URL, current screenshot,
     simulated steps) on Qwen3-VL-4B-Instruct, n samples (max 256 tokens, top-p 1.0); each reply scores 1 (Status
     success), 0.5 (failure but "On the right track to success: yes") or 0, and the candidate's score is the mean
  4. with --sims > 1 the world model is sampled that many times and the scores averaged (evaluate_simulation's num_of_sim)
arm_eval executes the argmax (ties -> lower index, i.e. the actor's earlier sample).
"""
import argparse
import asyncio
import base64
import io
import json
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)

WM_SYSTEM = ("You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage "
             "and an operation to perform on the webpage. You are required to predict the changes that will occur on the "
             "webpage after the operation is performed, such as the appearance of new elements, the disappearance of existing "
             "elements, or changes in the content of existing elements. The operation type and the element to operate will be "
             "provided in the prompt. Directly output 'State changes: ...' and don't output anything else. Try to be as "
             "comprehensive and detailed as possible.")
VALUE_SYSTEM = """
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the current state of the webpage, your goal is to decide **whether the simulated steps by the agent indicate a successful execution of the user intent**. In particular, if the predicted state (i.e., the current state represented by the last image plus all the predicted changes so far) corresponds to a successful final state. If it is a failure but it looks like the simulated steps are on the right track towards success, you should also output as such. Note that, in the simulated steps, all the state changes are predicted by the agent's world model, and they may not actually be faithful to the real website interactions (e.g., some proposed actions may not be avaiable in a realistic website). You should also account for this in your evaluation (e.g., if the predicted state changes are not reasonable then it's probably a failure).

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""


class SelectionRequest(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str


def describe(action, thought):
    """Option 1: our tool call in words + the actor's reasoning, which names the element the point is on."""
    m = TOOL.search(action or '')
    try:
        call = json.loads(m.group(1)) if m else {}
    except json.JSONDecodeError:
        call = {}
    name, args = call.get('name', ''), call.get('arguments') or {}
    if name == 'click':
        op = 'click'
    elif name == 'hover':
        op = 'hover over'
    elif name == 'write':
        op = f"type '{args.get('message', '')}'"
    elif name == 'scroll':
        op = f"scroll {args.get('direction', 'down')}"
    elif name == 'press_keys':
        keys = args.get('keys'); op = 'press ' + ('+'.join(keys) if isinstance(keys, list) else str(keys))
    elif name == 'goto_url':
        op = f"go to {args.get('url', '')}"
    elif name == 'go_back':
        op = 'go back'
    elif name == 'done':
        op = f"stop and answer: {args.get('response', '')}"
    else:
        op = name or 'act'
    thought = ' '.join((thought or '').replace('<think>', ' ').replace('</think>', ' ').split())
    if len(thought) > 400:
        thought = '...' + thought[-400:]
    return f'{op} ({thought})' if thought else op


def parse_score(text):  # simulation_scoring.evaluate_simulation_inner, unchanged
    try:
        pred = re.search(r'Status: "?(.+)"?', text).group(1)
        if 'success' in pred.lower():
            return 1.0
        on_path = re.search(r'On the right track to success: "?(.+)"?', text).group(1)
        return 0.5 if 'yes' in on_path.lower() else 0.0
    except Exception:
        return 0.0


def make_app(a):
    app = FastAPI(); client = httpx.AsyncClient(timeout=600, trust_env=False); log = Path(a.log) if a.log else None
    # Bound in-flight model requests: each value request carries n samples, and with 12 parallel tasks x 5 candidates
    # x n=10 the value server ran out of request slots and died (job 423033, alloc_req_slots runs out of memory).
    value_gate, dreamer_gate = asyncio.Semaphore(a.value_concurrency), asyncio.Semaphore(a.dreamer_concurrency)

    async def chat(url, payload):
        for attempt in range(3):
            try:
                r = await client.post(f'{url}/chat/completions', json=payload); r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    async def simulate(task, url, history, cand, img_url):
        desc = describe(cand.get('action', ''), cand.get('thought', ''))
        async with dreamer_gate:
          wm = await chat(a.dreamer_url, dict(model=a.dreamer_name, temperature=1.0, max_tokens=1024, messages=[
            {'role': 'system', 'content': WM_SYSTEM},
            {'role': 'user', 'content': [{'type': 'text', 'text': f' Please predict the changes after {desc}.'},
                                         {'type': 'image_url', 'image_url': {'url': img_url, 'detail': 'high'}}]}]))
        change = wm['choices'][0]['message']['content'] or ''
        imagination = 'Proposed New Action: \n' + desc + '\n' + 'Predicted new webpage: \n' + change + '\n'
        text = (f'User Intent: {task}\nAction History: ' + '\n'.join(history) + f'\nCurrent URL: {url}\n'
                'The last 1 snapshots of the agent\'s trajectory are shown in the 1 images. The LAST IMAGE represents the '
                f'current state of the webpage.\nSimulated steps: {imagination}\n')
        async with value_gate:
          val = await chat(a.value_url, dict(model=a.value_name, n=a.n, max_tokens=256, top_p=1.0, temperature=1.0, messages=[
            {'role': 'system', 'content': VALUE_SYSTEM},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img_url, 'detail': 'high'}},
                                         {'type': 'text', 'text': text}]}]))
        replies = [c['message']['content'] or '' for c in val['choices']]
        scores = [parse_score(r) for r in replies]
        usage = dict(dreamer=wm.get('usage'), value=val.get('usage'))
        return sum(scores) / max(1, len(scores)), dict(description=desc, change=change, scores=scores,
                                                        replies=[r[:600] for r in replies[:3]], usage=usage)

    @app.get('/health')
    def health():
        return dict(mode='scalar', selector='webdreamer', dreamer=a.dreamer_name, value=a.value_name, n=a.n, sims=a.sims,
                    description='option 1: tool call in words + actor reasoning', imagination_steps=0)

    @app.post('/select')
    async def select(req: SelectionRequest):
        t0 = time.monotonic()
        img = Image.open(io.BytesIO(base64.b64decode(req.screenshot))).convert('RGB')
        buf = io.BytesIO(); img.save(buf, format='PNG'); img_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        history = [(h.get('action') or '').replace('<|im_end|>', '').strip() for h in req.history]
        jobs = [simulate(req.task, req.url, history, c, img_url) for c in req.candidates for _ in range(a.sims)]
        out = await asyncio.gather(*jobs)
        scores = [sum(s for s, _ in out[i * a.sims:(i + 1) * a.sims]) / a.sims for i in range(len(req.candidates))]
        if log:
            with log.open('a') as f:
                f.write(json.dumps(dict(task=req.task[:120], url=req.url, scores=scores, sims=[d for _, d in out],
                                        seconds=round(time.monotonic() - t0, 2))) + '\n')
        return {'scores': scores, 'seconds': time.monotonic() - t0}
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dreamer-url', required=True); ap.add_argument('--dreamer-name', required=True)
    ap.add_argument('--value-url', required=True); ap.add_argument('--value-name', required=True)
    ap.add_argument('--n', type=int, default=10); ap.add_argument('--sims', type=int, default=1)
    ap.add_argument('--value-concurrency', type=int, default=4); ap.add_argument('--dreamer-concurrency', type=int, default=16)
    ap.add_argument('--port', type=int, default=19101); ap.add_argument('--log')
    a = ap.parse_args()
    uvicorn.run(make_app(a), host='127.0.0.1', port=a.port)


if __name__ == '__main__':
    main()
