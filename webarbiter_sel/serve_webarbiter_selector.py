#!/usr/bin/env python3
"""WebArbiter-7B (Zhang et al., arXiv 2601.21872) as an arm_eval `scalar` selector over our SFT actor's candidates.

    serve_webarbiter_selector.py --arbiter-url http://127.0.0.1:P/v1 --arbiter-name WebArbiter --port 19101 \
        [--log calls.jsonl] [--diff-model gpt-4o-mini] [--max-obs-chars 60000]

Needs arm_eval --selector-page-tree: each /select request then carries the env's WebArena accessibility tree of the
current viewport (openwebrl/env/webarena_axtree.py, a port of the released TextObervationProcessor), task_id and turn.

Per step, as the released pipeline (YaoZ720/WebArbiterCode @ 764ec26) and our WebArena reproduction (webarena_repro/,
A18 harness: tillicum/BASELINES.md §1-2) run it, with prompts loaded verbatim from viewer/src/prompts/:
  1. candidates: our actor's 5 iid samples, each rendered into WebArbiter's bracket action grammar (sample_p.py, plus
     the send_msg_to_user and noop verbs of WebArbiter-Data that the reproduction adds, PROTOCOL A12):
       click / hover point_2d          -> click [id] / hover [id]   (id = the tree element under the point, see 2)
       click p + write m (+ Enter)     -> type [id] [m] [0]  (type [id] [m] when the actor also pressed Enter, the
                                          released default press_enter_after=1)
       write m (+ Enter)               -> type [id] [m] [0] / type [id] [m]  (id = the deepest element marked focused:
                                          True other than RootWebArea, which Chrome also marks; else the element of this
                                          task's last executed click if it is still in the tree; else `type [] [m]`,
                                          counted unmapped)
       scroll d -> scroll [d];  press_keys k -> press [k1+k2];  goto_url u -> goto [u];  go_back -> go_back;
       new_tab / close_tab -> new_tab / close_tab;  switch_tab i -> tab_focus [i];  done r -> send_msg_to_user [r];
       wait -> noop;  drag s, e -> drag [id_s] [id_e] (no drag verb in the grammar);  call_user q -> send_msg_to_user [q]
     other multi-call responses are rendered call by call, one grammar action per line
  2. coordinate -> element: the point (0-1000 grid of the viewport screenshot) is mapped to CSS px and matched against
     the union_bound boxes of the elements printed in the tree (obs_nodes_info of the same observation the candidates
     were proposed from): the smallest box containing the point among interactive roles (link, button, textbox, ...),
     else the smallest box of any printed element (RootWebArea excluded). FALLBACK when nothing contains the point:
     the target is rendered as its CSS pixel position, e.g. `click [(640, 360)]` (an id WebArbiter cannot find in the
     tree), and the candidate is counted as unmapped (selector_meta / log: targets_mapped, targets_unmapped)
  3. dedup: identical rendered actions are one entrant (released sampling dedup by action string), the first sample's
     thought kept; entrants ordered by frequency, ties by sample order (the reproduction's confidence = frequency order;
     our samples carry no confidence)
  4. state difference: the released compute_state_diff prompt (observation_p.system_text_diff / user_text_diff) on
     gpt-4o-mini (released default model and decoding: T 1.0, top-p 0.9, max 2,048) over the previous and current
     obs_nodes_info lines, computed before the tournament (the released service races it in a thread); "(None)" on error
  5. knockout: released TournamentRunner bracket (FIFO consecutive pairs, odd entrant carries over; 5 entrants -> 4
     comparisons in 3 rounds), released RewardPromptBuilder prompt (reward_p role + evaluation_summarized_v3 as system,
     context_rft_v3 as user: intent, AXTREE, trajectory, start url = current url, current url, Prior Knowledge "No prior
     knowledge available" (live sites), state difference with the released first-step / no-change wording, the two
     THOUGHT/ACTION responses), WebArbiter-7B greedy, max 2,048 tokens (PROTOCOL D6, as the reproduction), released
     parser (<Answer>Response 1|2</Answer>; anything else -> Response 1 wins, counted as default_a; a failed call drops
     both, as released). If every comparison fails, the first entrant wins (counted as tournament_failed). Before
     the bracket, a max_tokens=1 call measures the longest comparison prompt; a tree whose prompt would not leave room
     for the 2,048-token reply in WebArbiter's 32k context is cut from the end until it does (counted: fit_cuts)
  6. trajectory: the released get_trajectory_text format ("Step i:\\nTHOUGHT: ...\\nACTION: ...", 0-based), each earlier
     executed action in its rendering at its own step, enriched with that step's element text as enrich_action_with_nodes
  7. scores for arm_eval: every sample rendered as the winning entrant 1.0 (the lowest index executes), every other
     0.25 x its comparisons won (< 1); arm_eval executes the argmax
Deviations from the released WebArbiter setup, beyond those of the reproduction (PROTOCOL.md): the actor is our
OpenWebRL-4B-SFT (screenshot-only, coordinate tool calls, 5 iid samples at T 1.0 instead of GPT-4o's id actions); the
coordinate -> element mapping and fallback above; drag and multi-call renderings outside the grammar; no terminal
detector (our actor stops with done, as A18's detector-off regime); trees longer than --max-obs-chars, or too long
for the 32k context, are cut (counted); viewport 1280x720 (A18 OM2W setting; the paper's WebArena service used 1024x2490).
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import runpy
import time
from collections import OrderedDict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RELEASED = Path('/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26/viewer/src/prompts')
_reward = runpy.run_path(str(RELEASED / 'reward_p.py'))       # pure string constants, used verbatim
_observation = runpy.run_path(str(RELEASED / 'observation_p.py'))
SYSTEM_TEXT = _reward['role'] + _reward['evaluation_summarized_v3']  # RewardPromptBuilder: f"{reward_role}{criteria}"
CONTEXT = _reward['context_rft_v3']
DIFF_SYSTEM, DIFF_USER = _observation['system_text_diff'], _observation['user_text_diff']
ANSWER = re.compile(r"<Answer>\s*(Response\s*[12])\s*</Answer>", re.I)  # RewardGenerator._parse_reward_response
TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
LINE_ID = re.compile(r'^\s*\[([^\]\s]+)\]', re.M)
INTERACTIVE = {'link', 'button', 'textbox', 'searchbox', 'combobox', 'checkbox', 'radio', 'menuitem', 'menuitemcheckbox',
               'menuitemradio', 'tab', 'option', 'switch', 'slider', 'spinbutton', 'listbox', 'treeitem',
               'MenuListOption', 'MenuListPopup', 'PopUpButton', 'ToggleButton', 'textField', 'DisclosureTriangle'}


class SelectionRequest(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str
    page_tree: dict | None = None
    task_id: str | None = None
    turn: int | None = None


class Page:
    """The elements WebArbiter sees (printed in the cleaned tree) with their boxes, and the point -> element rule."""

    def __init__(self, tree):
        tree = tree or {}
        self.text = tree.get('text') or ''
        self.all_nodes = tree.get('nodes') or {}
        shown = set(LINE_ID.findall(self.text))
        self.nodes = {k: v for k, v in self.all_nodes.items() if k in shown}
        self.vw, self.vh = (tree.get('viewport') or [1280, 720])[:2]
        # Chrome also marks the document's RootWebArea focused: True, so take the last (deepest) other focused element
        # (smoke job 500425 typed into RootWebArea 11 times when the first focused node was used)
        focused = [k for k, v in self.nodes.items() if ' focused: True' in (v.get('text') or '')]
        focused = [k for k in focused if not (self.nodes[k].get('text') or '').startswith(f'[{k}] RootWebArea')]
        self.focused = focused[-1] if focused else None

    def role(self, nid):
        parts = (self.nodes[nid].get('text') or '').split(' ', 2)
        return parts[1] if len(parts) > 1 else ''

    def px(self, p):
        return p[0] / 1000 * self.vw, p[1] / 1000 * self.vh

    def at(self, p):
        x, y = self.px(p)
        hits = []
        for k, v in self.nodes.items():
            b = v.get('union_bound')
            if not b or len(b) != 4 or self.role(k) == 'RootWebArea':
                continue
            if b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]:
                hits.append((b[2] * b[3], k))
        inter = [h for h in hits if self.role(h[1]) in INTERACTIVE]
        pool = inter or hits
        return min(pool)[1] if pool else None


def tool_calls(action):
    calls = []
    for m in TOOL.finditer(action or ''):
        try:
            c = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(c, dict):
            calls.append((c.get('name') or '', c.get('arguments') or {}))
    return calls


def _is_enter(call):
    if not call or call[0] != 'press_keys':
        return False
    k = call[1].get('keys')
    k = k if isinstance(k, list) else [k]
    return len(k) == 1 and str(k[0]).lower() in ('enter', 'return')


def render(action, page, last_click=None):
    """Our tool call(s) -> (WebArbiter grammar action, [(kind, mapped)] for each element target)."""
    calls, out, targets = tool_calls(action), [], []

    def el(p, kind):
        if not (isinstance(p, list) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:2])):
            targets.append((kind, False)); return '?'
        nid = page.at(p)
        targets.append((kind, nid is not None))
        if nid is not None:
            return nid
        x, y = page.px(p)
        return f'({round(x)}, {round(y)})'

    def focus():
        nid = page.focused or (last_click if last_click in page.nodes else None)
        targets.append(('focus', nid is not None))
        return nid or ''

    i = 0
    while i < len(calls):
        name, a = calls[i]
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        if name == 'click' and nxt and nxt[0] == 'write':
            target, msg = el(a.get('point_2d'), 'click'), nxt[1].get('message', '')
            i += 2; enter = i < len(calls) and _is_enter(calls[i]); i += enter
            out.append(f'type [{target}] [{msg}]' + ('' if enter else ' [0]')); continue
        if name == 'write':
            target, msg = focus(), a.get('message', '')
            i += 1; enter = i < len(calls) and _is_enter(calls[i]); i += enter
            out.append(f'type [{target}] [{msg}]' + ('' if enter else ' [0]')); continue
        i += 1
        if name in ('click', 'hover'):
            out.append(f"{name} [{el(a.get('point_2d'), name)}]")
        elif name == 'scroll':
            out.append(f"scroll [{a.get('direction', 'down')}]")
        elif name == 'press_keys':
            k = a.get('keys'); out.append(f"press [{'+'.join(map(str, k)) if isinstance(k, list) else k}]")
        elif name == 'goto_url':
            out.append(f"goto [{a.get('url', '')}]")
        elif name in ('go_back', 'go_forward', 'new_tab', 'close_tab'):
            out.append(name)
        elif name == 'switch_tab':
            out.append(f"tab_focus [{a.get('tab_index', 0)}]")
        elif name == 'done':
            out.append(f"send_msg_to_user [{a.get('response', '')}]")
        elif name == 'call_user':
            out.append(f"send_msg_to_user [{a.get('question', '')}]")
        elif name == 'wait':
            out.append('noop')
        elif name == 'drag':
            s = el(a.get('start_point_2d'), 'drag')
            e = el(a.get('end_point_2d'), 'drag') if a.get('end_point_2d') is not None else None
            out.append(f'drag [{s}]' + (f' [{e}]' if e is not None else ''))
        else:
            out.append(name or 'none')
    return ('\n'.join(out) or 'none'), targets


def enrich(action, nodes):
    """trajectory_utils.enrich_action_with_nodes: [id] -> [id] "<obs_nodes_info text[:50]>"."""
    out = action
    for nid in re.findall(r"\[(\d+)\]", action):
        text = ((nodes.get(nid) or {}).get('text') or '').strip()
        if text:
            out = out.replace(f'[{nid}]', f'[{nid}] "{text[:50]}"', 1)
    return out


def node_lines(nodes):
    """convert_observation_dict_to_text({'obs_nodes_info': ...}): obs_nodes_info has no 'type', so 'Element'."""
    return '\n'.join(f'[{k}] Element {v.get("text")}' if v.get('text') else f'[{k}] Element' for k, v in nodes.items())


def reward_user_text(intent, observation, url, trajectory, a, b, state_diff):
    """RewardPromptBuilder.build_reward_prompt's user text, unchanged."""
    if not trajectory or len(trajectory.strip()) < 50:
        sd = 'N/A (first step - no previous state to compare)'
    elif not state_diff or not state_diff.strip():
        sd = 'No state changes detected'
    else:
        sd = state_diff
    return CONTEXT.format(intent=intent or '', observation=observation or '', trajectory=trajectory or '',
                          start_url=url or '', current_url=url or '', domain_knowledge='No prior knowledge available',
                          state_diff=sd, thought1=a['thought'] or '', action1=a['action'] or '',
                          thought2=b['thought'] or '', action2=b['action'] or '')


def make_app(a):
    app = FastAPI()
    local = httpx.AsyncClient(timeout=600, trust_env=False)
    remote = httpx.AsyncClient(timeout=120, trust_env=True)  # OpenAI through the cluster proxy
    log = Path(a.log) if a.log else None
    gate = asyncio.Semaphore(a.arbiter_concurrency)
    tasks = OrderedDict()  # task_id -> {turn: {by_action, nodes, url, target}}; LRU, earlier turns feed history/diff

    async def arbiter(system, user):
        async with gate:
            for attempt in range(3):  # released client: max_retries=2 on transport errors
                try:
                    r = await local.post(f'{a.arbiter_url}/chat/completions', json=dict(
                        model=a.arbiter_name, temperature=0.0, top_p=1.0, max_tokens=2048,
                        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))
                    if r.status_code >= 400:
                        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:400]}')
                    j = r.json()
                    return j['choices'][0]['message']['content'] or '', j.get('usage') or {}
                except (httpx.TransportError, RuntimeError) as e:
                    if attempt == 2 or (isinstance(e, RuntimeError) and 'HTTP 4' in str(e)):
                        raise
                    await asyncio.sleep(2)

    async def fit(observation, build):
        """Cut the tree until the longest comparison prompt + the 2,048-token reply fits WebArbiter's context. One
        max_tokens=1 call per try reads the exact prompt length (and warms the radix cache the comparisons share)."""
        cuts = 0
        for _ in range(5):
            try:
                async with gate:
                    r = await local.post(f'{a.arbiter_url}/chat/completions', json=dict(
                        model=a.arbiter_name, temperature=0.0, max_tokens=1,
                        messages=[{'role': 'system', 'content': SYSTEM_TEXT}, {'role': 'user', 'content': build(observation)}]))
            except httpx.TransportError:
                return observation, cuts, None
            if r.status_code == 400:
                observation = observation[:int(len(observation) * 0.6)]; cuts += 1; continue
            if r.status_code >= 400:
                return observation, cuts, None
            pt = (r.json().get('usage') or {}).get('prompt_tokens') or 0
            budget = a.arbiter_context - 2048 - 16
            if pt <= budget:
                return observation, cuts, pt
            observation = observation[:int(len(observation) * budget / pt * 0.97)]; cuts += 1
        return observation, cuts, None

    async def state_diff(prev_nodes, cur_nodes):
        user = DIFF_USER.format(prev_text=node_lines(prev_nodes)[:200000], cur_text=node_lines(cur_nodes)[:200000])
        try:
            r = await remote.post(os.environ.get('JUDGE_API_BASE', 'https://api.openai.com/v1').rstrip('/') + '/chat/completions',
                                  headers={'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', '')},
                                  json=dict(model=a.diff_model, temperature=1.0, top_p=0.9, max_tokens=2048, messages=[
                                      {'role': 'system', 'content': DIFF_SYSTEM}, {'role': 'user', 'content': user}]))
            r.raise_for_status(); j = r.json()
            text = (j['choices'][0]['message']['content'] or '').strip()
            return (text or '(None)'), j.get('usage') or {}, None
        except Exception as e:  # released StateDiffGenerator: "(None)" on any error
            return '(None)', {}, f'{type(e).__name__}: {str(e)[:200]}'

    @app.get('/health')
    def health():
        return dict(mode='scalar', selector='webarbiter_knockout', arbiter=a.arbiter_name, decoding='greedy, 2048',
                    diff_model=a.diff_model, prompts='WebArbiterCode@764ec26 reward_p.role+evaluation_summarized_v3 / '
                    'context_rft_v3, observation_p diff', system_sha256=hashlib.sha256(SYSTEM_TEXT.encode()).hexdigest(),
                    max_obs_chars=a.max_obs_chars, needs='arm_eval --selector-page-tree')

    @app.post('/select')
    async def select(req: SelectionRequest):
        t0 = time.monotonic()
        if req.page_tree is None or req.task_id is None or req.turn is None:
            raise HTTPException(400, 'WebArbiter selector needs arm_eval --selector-page-tree (page_tree, task_id, turn)')
        page = Page(req.page_tree)
        if req.turn == 0:
            tasks.pop(req.task_id, None)
        state = tasks.setdefault(req.task_id, {}); tasks.move_to_end(req.task_id)
        while len(tasks) > 512:
            tasks.popitem(last=False)
        # trajectory: earlier executed actions as rendered at their own step
        steps, last_click, history_misses = [], None, 0
        for i, h in enumerate(req.history):
            act = (h.get('action') or '').strip(); thought = (h.get('thought') or '').strip()
            ent = (state.get(i) or {}).get('by_action', {}).get(act)
            if ent is None:
                history_misses += 1; shown = render(act, Page({}))[0]
            else:
                shown = enrich(ent['grammar'], state[i]['nodes']); last_click = ent['click'] or last_click
            steps.append(f'Step {i}:\nTHOUGHT: {thought}\nACTION: {shown}' if thought else f'Step {i}: ACTION: {shown}')
        trajectory = '\n'.join(steps)
        # candidates
        rendered = []
        for c in req.candidates:
            g, targets = render(c.get('action', ''), page, last_click)
            ids = [t for t in re.findall(r'^(?:click|type|hover) \[([^\]]+)\]', g, re.M) if t in page.nodes]
            rendered.append(dict(grammar=g, targets=targets, click=ids[-1] if ids else None))
        state[req.turn] = dict(nodes=page.all_nodes, url=req.url,
                               by_action={(c.get('action') or '').strip(): r for c, r in zip(req.candidates, rendered)})
        entrants = []
        for j, (c, r) in enumerate(zip(req.candidates, rendered)):
            e = next((e for e in entrants if e['action'] == r['grammar']), None)
            if e is None:
                entrants.append(dict(action=r['grammar'], thought=(c.get('thought') or '').strip(), idx=[j], wins=0))
            else:
                e['idx'].append(j)
        entrants.sort(key=lambda e: (-len(e['idx']), e['idx'][0]))
        for k, e in enumerate(entrants):
            e['entrant'] = k
        observation = page.text if page.text else f"(no accessibility tree: {(req.page_tree or {}).get('error', 'empty')})"
        truncated = len(observation) > a.max_obs_chars
        if truncated:
            observation = observation[:a.max_obs_chars]
        prompt_tokens, fit_cuts = None, 0
        if len(entrants) > 1:  # longest pair (by candidate text) bounds every comparison prompt of this step
            lx, ly = sorted(entrants, key=lambda e: -len(e['thought']) - len(e['action']))[:2]
        diff, diff_usage, diff_error = '', {}, None
        prev = state.get(req.turn - 1)
        if req.turn > 0 and len(entrants) > 1:
            if prev is not None:
                diff, diff_usage, diff_error = await state_diff(prev['nodes'], page.all_nodes)
            else:
                diff, diff_error = '(None)', 'previous step not cached'
        if len(entrants) > 1:
            observation, fit_cuts, prompt_tokens = await fit(
                observation, lambda o: reward_user_text(req.task, o, req.url, trajectory, lx, ly, diff))
            truncated = truncated or fit_cuts > 0
        # knockout (TournamentRunner.__call__)
        queue, comparisons, rnd, failed = deque(entrants), [], 1, False
        while len(queue) > 1:
            pairs = []
            while len(queue) >= 2:
                pairs.append((queue.popleft(), queue.popleft()))
            users = [reward_user_text(req.task, observation, req.url, trajectory, x, y, diff) for x, y in pairs]
            results = await asyncio.gather(*(arbiter(SYSTEM_TEXT, u) for u in users), return_exceptions=True)
            for (x, y), u, res in zip(pairs, users, results):
                row = dict(round=rnd, a=x['entrant'], b=y['entrant'], prompt_sha256=hashlib.sha256(u.encode()).hexdigest(),
                           prompt_chars=len(u))
                if isinstance(res, Exception):
                    row.update(error=f'{type(res).__name__}: {str(res)[:300]}'); comparisons.append(row); continue
                raw, usage = res
                m = ANSWER.search(raw)
                b_wins = bool(m) and ('response 2' in m.group(1).lower() or 'response2' in m.group(1).lower())
                w = y if b_wins else x
                w['wins'] += 1; queue.append(w)
                row.update(reply=raw, parsed=m.group(1) if m else None, default_a=not m, winner=w['entrant'], usage=usage)
                comparisons.append(row)
            rnd += 1
        if not queue:
            failed, winner = True, entrants[0]
        else:
            winner = queue[0]
        scores = [0.0] * len(req.candidates)
        for e in entrants:
            for j in e['idx']:
                scores[j] = 1.0 if e is winner else 0.25 * e['wins']
        flat = [t for r in rendered for t in r['targets']]
        meta = dict(entrants=len(entrants), winner_entrant=winner['entrant'], winner_index=min(winner['idx']),
                    comparisons=len(comparisons), comparison_errors=sum('error' in c for c in comparisons),
                    default_a=sum(bool(c.get('default_a')) for c in comparisons), tournament_failed=failed,
                    targets_mapped=sum(m for _, m in flat), targets_unmapped=sum(not m for _, m in flat),
                    obs_truncated=truncated, fit_cuts=fit_cuts, prompt_tokens=prompt_tokens, history_misses=history_misses, diff_error=diff_error,
                    seconds=round(time.monotonic() - t0, 2))
        if log:
            with log.open('a') as f:
                f.write(json.dumps(dict(
                    task_id=req.task_id, turn=req.turn, task=req.task[:200], url=req.url, meta=meta, scores=scores,
                    candidates=[dict(action=(c.get('action') or '')[:400], grammar=r['grammar'], targets=r['targets'])
                                for c, r in zip(req.candidates, rendered)],
                    entrants=[dict(entrant=e['entrant'], action=e['action'], idx=e['idx'], wins=e['wins'], thought=e['thought'])
                              for e in entrants],
                    comparisons=comparisons, state_diff=diff, diff_usage=diff_usage, trajectory=trajectory,
                    observation=observation, tree_error=(req.page_tree or {}).get('error')), ensure_ascii=False) + '\n')
        return {'scores': scores, 'seconds': time.monotonic() - t0, 'selector_meta': meta}
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arbiter-url', required=True); ap.add_argument('--arbiter-name', default='WebArbiter')
    ap.add_argument('--diff-model', default='gpt-4o-mini')
    ap.add_argument('--max-obs-chars', type=int, default=60000, help='cut longer trees (WebArbiter context 32k tokens)')
    ap.add_argument('--arbiter-concurrency', type=int, default=32)
    ap.add_argument('--arbiter-context', type=int, default=32768, help='WebArbiter server --context-length')
    ap.add_argument('--port', type=int, default=19101); ap.add_argument('--log')
    a = ap.parse_args()
    uvicorn.run(make_app(a), host='127.0.0.1', port=a.port)


if __name__ == '__main__':
    main()
