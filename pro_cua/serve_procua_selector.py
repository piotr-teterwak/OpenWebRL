#!/usr/bin/env python3
"""PRO-CUA's frozen-VLM process reward model as an arm_eval `scalar` selector.

    serve_procua_selector.py --vlm-url http://127.0.0.1:PORT/v1 --vlm-name NAME --port 19101 [--log calls.jsonl]

PRO-CUA (yifei-he/PRO-CUA, verl/examples/data_preprocess/webvoyager_reward_new.py, used by verl/scripts/pro_cua_4b.sh)
scores a proposed step with a frozen Qwen3-VL-4B-Instruct: PRM_PROMPT (verbatim, imported from their file) with the task
instruction, the history of executed actions (json list), "Step {i}: {thought + action}", and the screenshot BEFORE the
action annotated by their own data_visualization.actions_visual (red point at a click target, arrow for scroll/drag,
action name and typed text overlaid); the reply ends in a JSON block {"is_correct": true|false}. Their RL reward is the
binary verdict (512 response tokens, the model's default sampling). Here every candidate is judged with greedy decoding
and the same 512-token budget, and its score is the model's probability of `true` at the is_correct value token
(renormalized over true/false from the top logprobs); the selector executes the argmax (arm_eval: ties -> lower index).
A reply with no parseable verdict scores 0, as their reward does.

Our actor's tool calls are rendered into their action-string grammar before annotation (point_2d is on a 0-1000 grid
of the screenshot): click/hover -> click(point=...), scroll -> scroll(direction=...), drag -> drag(start_point, end_point),
write -> type(content=...), press_keys -> hotkey(key=...), done -> finished(content=...); other actions are shown by name.
"""
import argparse
import asyncio
import base64
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

PRO = Path('/gpfs/scrubbed/piotrt/owrl/pro-cua/verl/examples/data_preprocess')
sys.path.insert(0, str(PRO))
from data_visualization import actions_visual  # noqa: E402  (their annotation code, unchanged)

_src = (PRO / 'webvoyager_reward_new.py').read_text()
PRM_PROMPT = re.search(r'PRM_PROMPT = """(.*?)"""', _src, re.S).group(1)  # verbatim; the module itself imports verl
TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
VERDICT = re.compile(r'(?:"|\')?is_correct(?:"|\')?\s*:\s*(true|false)', re.I)


class SelectionRequest(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str


def to_action_string(action, w, h):
    """Our tool call -> PRO-CUA's rendered action string (data_visualization.parse_action_string grammar)."""
    m = TOOL.search(action or '')
    try:
        call = json.loads(m.group(1)) if m else {}
    except json.JSONDecodeError:
        call = {}
    name, args = call.get('name', ''), call.get('arguments') or {}
    px = lambda p: (round(p[0] / 1000 * w), round(p[1] / 1000 * h))
    esc = lambda s: str(s).replace("'", '’')
    if name in ('click', 'hover') and isinstance(args.get('point_2d'), list):
        x, y = px(args['point_2d']); return f"click(point='<point>{x} {y}</point>')"
    if name == 'scroll':
        return f"scroll(direction='{args.get('direction', 'down')}')"
    if name == 'drag' and isinstance(args.get('start_point_2d'), list) and isinstance(args.get('end_point_2d'), list):
        (x1, y1), (x2, y2) = px(args['start_point_2d']), px(args['end_point_2d'])
        return f"drag(start_point='<point>{x1} {y1}</point>', end_point='<point>{x2} {y2}</point>')"
    if name == 'write':
        return f"type(content='{esc(args.get('message', ''))}')"
    if name == 'press_keys':
        keys = args.get('keys'); return f"hotkey(key='{esc(' '.join(keys) if isinstance(keys, list) else keys)}')"
    if name == 'done':
        return f"finished(content='{esc(args.get('response', ''))}')"
    return f"{name or 'unknown'}()"


def annotate(img, action):
    try:
        out, _ = actions_visual([to_action_string(action, img.width, img.height)], img.copy(), '')
        return out
    except Exception:
        return img.copy()  # their builder also falls back to the raw screenshot


def p_true(content, logprobs):
    """P(true) at the is_correct value token, from that token's top logprobs (all spellings of true / false summed, then
    renormalized); None if the reply has no verdict. The token is found by walking the generated tokens to the last
    true/false token that directly follows `is_correct":` (SGLang's token strings do not concatenate to the reply text
    character for character, so character offsets cannot be used; smoke job 416019 read the wrong token that way)."""
    verdicts = VERDICT.findall(content or '')
    if not verdicts:
        return None
    joined, hit = '', None
    for tok in logprobs or []:
        t = tok['token']
        if t.strip().lower() in ('true', 'false') and re.search(r'is_correct[\'"]?\s*:\s*$', joined[-40:], re.I):
            hit = tok
        joined += t
    if hit is None:
        return 1.0 if verdicts[-1].lower() == 'true' else 0.0
    pt = sum(math.exp(x['logprob']) for x in hit.get('top_logprobs') or [] if x['token'].strip().lower() == 'true')
    pf = sum(math.exp(x['logprob']) for x in hit.get('top_logprobs') or [] if x['token'].strip().lower() == 'false')
    if pt + pf == 0:
        return 1.0 if hit['token'].strip().lower() == 'true' else 0.0
    return pt / (pt + pf)


def make_app(a):
    app = FastAPI(); client = httpx.AsyncClient(timeout=300, trust_env=False); log = Path(a.log) if a.log else None

    async def judge(task, history, step, cand, img):
        action_code = (f"{cand.get('thought', '').strip()}\n{cand.get('action', '').strip()}").strip()
        text = (PRM_PROMPT.replace('{instruction}', task).replace('{step_index}', str(step))
                .replace('{action_code}', action_code).replace('{history_actions}', json.dumps(history)))
        buf = io.BytesIO(); annotate(img, cand.get('action', '')).save(buf, format='PNG')
        msg = [{'role': 'user', 'content': [
            {'type': 'text', 'text': text}, {'type': 'text', 'text': 'Screenshot BEFORE last action'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}},
            {'type': 'text', 'text': 'Your judgement:\n'}]}]
        for attempt in range(3):
            try:
                r = await client.post(f'{a.vlm_url}/chat/completions', json=dict(
                    model=a.vlm_name, messages=msg, max_tokens=512, temperature=0.0, logprobs=True, top_logprobs=5))
                r.raise_for_status(); ch = r.json()['choices'][0]
                content = ch['message']['content'] or ''
                s = p_true(content, (ch.get('logprobs') or {}).get('content'))
                return (0.0 if s is None else s), content
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    @app.get('/health')
    def health():
        return dict(mode='scalar', selector='pro_cua_prm', vlm=a.vlm_name, prompt='PRO-CUA PRM_PROMPT (webvoyager_reward_new.py)',
                    score='P(is_correct=true), greedy, 512 tokens')

    @app.post('/select')
    async def select(req: SelectionRequest):
        t0 = time.monotonic()
        img = Image.open(io.BytesIO(base64.b64decode(req.screenshot))).convert('RGB')
        history = [(h.get('action') or '').replace('<|im_end|>', '').strip() for h in req.history]
        out = await asyncio.gather(*(judge(req.task, history, len(history) + 1, c, img) for c in req.candidates))
        scores = [s for s, _ in out]
        if log:
            with log.open('a') as f:
                f.write(json.dumps(dict(task=req.task[:120], url=req.url, scores=scores,
                                        actions=[(c.get('action') or '')[:160] for c in req.candidates],
                                        replies=[c[:1500] for _, c in out], seconds=round(time.monotonic() - t0, 2))) + '\n')
        return {'scores': scores, 'seconds': time.monotonic() - t0}
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vlm-url', required=True); ap.add_argument('--vlm-name', required=True)
    ap.add_argument('--port', type=int, default=19101); ap.add_argument('--log')
    a = ap.parse_args()
    uvicorn.run(make_app(a), host='127.0.0.1', port=a.port)


if __name__ == '__main__':
    main()
