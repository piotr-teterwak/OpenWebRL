"""Piotr's SelectionARM as the WebArena selector (wa_episode.py --method selectionarm).

The ARM (PTeterwak/OpenWebRL-4B-SelectionARM @ 81b452d, scripts/serve_arm.py --mode selection) was trained on the
OpenWebRL-4B-SFT actor's candidates: a screenshot plus tool calls on a 0-1000 point grid. It never sees an accessibility
tree. Each released-harness candidate (WebArena bracket action on element ids) is therefore rendered as the tool call
the actor would have written, grounded with that element's bounding box in the current observation (as Zixian's v11
grounder does, but in the actor's own syntax instead of a descriptor dict):

  click/hover [id]        -> click/hover point_2d = element centre / viewport * 1000
  type [id] text [0|1]    -> click point_2d, then write message (and press_keys Enter unless the flag is 0)
  clear [id] / select_option [id] [opt] -> click point_2d (+ write the option text for select_option)
  scroll [down|up]        -> scroll direction, amount 0.5
  press [keys]            -> press_keys keys
  goto [url] / go_back / go_forward / new_tab / tab_focus [i] / close_tab -> goto_url / go_back / ... / switch_tab / close_tab
  send_msg_to_user [ans]  -> done response ans;  report_infeasible [why] -> done response "N/A (why)";  noop -> wait 1 s
A candidate whose element is missing from the observation is not shown to the ARM; one whose centre lies outside the
viewport is shown with its off-grid point (e.g. y > 1000 below the fold). If none survives, the first candidate executes. History = the earlier
executed candidates, rendered the same way at their own step. The screenshot is the observation the candidates were
proposed from (byte for byte, viewport-sized).
"""
import base64
import json
import os
from pathlib import Path
import re
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # openwebrl.arm_inference (selection parser)

ID = re.compile(r'\[(\d+)\]')
ARGS = re.compile(r'\[([^\]]*)\]')


def _tool(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def target_y(action, nodes):
    """Vertical centre (page px) of the element an id action targets, or None."""
    args = ARGS.findall((action or '').strip())
    n = nodes.get(str(args[0])) if args else None
    b = (n or {}).get('union_bound')
    if not b or len(b) != 4 or b[2] <= 0 or b[3] <= 0:
        return None
    return b[1] + b[3] / 2


def render(action, nodes, viewport, y0=0, height=None):
    """WebArena bracket action -> the SFT actor's tool-call string, or None if its target is not on screen.

    y0/height: the vertical window of the screenshot the ARM sees (crop mode); default the whole viewport."""
    a = (action or '').strip(); verb = a.split(' ', 1)[0].split('[', 1)[0].strip().lower(); args = ARGS.findall(a)
    height = height or viewport['height']

    def point(eid):
        n = nodes.get(str(eid))
        b = (n or {}).get('union_bound')
        if not b or len(b) != 4 or b[2] <= 0 or b[3] <= 0:
            return None
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        # Off-screen targets are kept (user, 2026-11-04): their point simply falls outside 0-1000 on the screenshot grid.
        return [round(1000 * cx / viewport['width']), round(1000 * (cy - y0) / height)]
    if verb in ('click', 'hover', 'clear', 'select_option', 'type'):
        if not args:
            return None
        p = point(args[0])
        if p is None:
            return None
        if verb == 'hover':
            return _tool('hover', point_2d=p)
        out = [_tool('click', point_2d=p)]
        if verb == 'type':
            text = args[1] if len(args) > 1 else a.split(']', 1)[1].strip()
            out.append(_tool('write', message=text.strip().strip('"\'')))
            if not (len(args) > 2 and args[2].strip() == '0'):
                out.append(_tool('press_keys', keys=['Enter']))
        if verb == 'select_option' and len(args) > 1:
            out.append(_tool('write', message=args[1]))
        return '\n'.join(out)
    if verb == 'scroll':
        return _tool('scroll', direction=(args[0] if args else 'down').strip().lower(), amount=0.5)
    if verb == 'press':
        return _tool('press_keys', keys=(args[0] if args else '').split('+'))
    if verb == 'goto':
        return _tool('goto_url', url=args[0] if args else '')
    if verb in ('go_back', 'go_forward', 'new_tab', 'close_tab'):
        return _tool(verb)
    if verb == 'tab_focus':
        return _tool('switch_tab', index=int(args[0]) if args and args[0].isdigit() else 0)
    if verb == 'send_msg_to_user':
        return _tool('done', response=args[0] if args else a[len('send_msg_to_user'):].strip())
    if verb == 'report_infeasible':
        return _tool('done', response='N/A' + (f' ({args[0]})' if args else ''))
    if verb in ('noop', 'none'):
        return _tool('wait', seconds=1)
    return None


class SelectionARM:
    def __init__(self, endpoint, viewport, journal, crop=None):
        self.endpoint, self.viewport, self.journal = endpoint.rstrip('/'), viewport, journal
        # crop: show the ARM a viewport-wide window this many px tall (its 1280x720 training input) instead of the whole
        # tall screenshot; the window is centred on the candidates' targets (option 2, 2026-11-05). None = whole screenshot.
        self.crop = crop if crop and crop < viewport['height'] else None
        self.history = []; self.client = httpx.Client(timeout=300, trust_env=False)

    def window(self, candidates, nodes):
        if not self.crop:
            return 0, None
        ys = [target_y(c.get('action'), nodes) for c in candidates]  # pool order = sampling frequency rank
        if all(y is None for y in ys):
            return 0, self.crop  # no element targets (scroll / stop / goto): the top of the page
        if os.environ.get('WA_ARM_WINDOW') != 'cover':
            pts = sorted(y for y in ys if y is not None)
            mid = (pts[0] + pts[-1]) / 2  # option 2 (S18crop): centre of the band the targets span
            return int(min(max(0, mid - self.crop / 2), self.viewport['height'] - self.crop)), self.crop
        # option 2b (S18cover): the window showing the most candidate targets (10 px inside its edges); ties go to the
        # window that shows the highest-ranked targeted candidate, then to the one nearest the top of the page.
        top = next(y for y in ys if y is not None)
        best = None
        for s in range(0, max(1, self.viewport['height'] - self.crop + 1), 10):
            inside = lambda y: y is not None and s + 10 <= y <= s + self.crop - 10
            key = (sum(map(inside, ys)), inside(top), -s)
            if best is None or key > best[0]:
                best = (key, s)
        return best[1], self.crop

    def select(self, candidates, row, task):
        """candidates: released pool dicts (thought, action); row: the current native observation record."""
        nodes, shown = row.get('obs_nodes_info') or {}, []
        y0, height = self.window(candidates, nodes)
        for i, c in enumerate(candidates):
            r = render(c.get('action'), nodes, self.viewport, y0, height)
            if r is not None:
                shown.append((i, dict(thought=c.get('thought') or '', action=r)))
        trace = dict(pool=[c.get('action') for c in candidates], shown=[i for i, _ in shown], rendered=[s['action'] for _, s in shown],
                     window=[y0, height])
        if not shown:
            index = 0; trace['rule'] = 'no candidate on screen: first candidate'
        elif len(shown) == 1:
            index = shown[0][0]; trace['rule'] = 'single on-screen candidate'
        else:
            png = open(row['image']['path'] if isinstance(row.get('image'), dict) else row['image'], 'rb').read()
            if height:
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(png)); buf = io.BytesIO()
                im.crop((0, y0, im.width, y0 + height)).save(buf, format='PNG'); png = buf.getvalue()
            payload = dict(mode='selection', task=task, url=row.get('url', ''), history=self.history,
                           candidates=[s for _, s in shown], screenshot=base64.b64encode(png).decode())
            r = self.client.post(f'{self.endpoint}/select', json=payload); r.raise_for_status(); out = r.json()
            from openwebrl.arm_inference import parse_selection
            k = parse_selection(out.get('raw') or '', len(shown))
            index = shown[k][0]; trace.update(raw=out.get('raw'), probs=out.get('probs'), selection=k + 1)
        chosen = render(candidates[index].get('action'), nodes, self.viewport, y0, height) or (candidates[index].get('action') or '')
        self.history.append(dict(thought=candidates[index].get('thought') or '', action=chosen))
        with open(self.journal, 'a') as f:
            f.write(json.dumps(dict(trace, index=index, url=row.get('url')), default=str) + '\n')
        return index, trace
