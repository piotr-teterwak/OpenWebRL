#!/usr/bin/env python3
"""Put each branch candidate's own reasoning back into its action text (Best-of-Q embeds "action type and associated
reasoning", arXiv 2601.22701 §3.1; Zixian's panels store the tool call with the reasoning stripped).

    branch_thoughts.py --rows sim/rows.jsonl --out sim-thought/rows.jsonl

For every state the branch attempt directory (two levels above the saved screenshot) holds the actor's sampled
proposals at the branch turn as `candidate_draw` events. Each panel candidate is matched to the first draw whose tool
call is the same JSON call; its action text becomes "<thought>\n<tool call>", the form the actor emitted it in. Rows
keep their order (so the state embeddings of the source rows still apply); unmatched candidates keep the bare tool call
and are counted. Stdlib only.
"""
import argparse
import json
import os
import re
from pathlib import Path

TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def call_key(text):
    m = TOOL.search(text or '')
    try:
        return json.dumps(json.loads(m.group(1)), sort_keys=True) if m else None
    except json.JSONDecodeError:
        return None


def draws(attempt):
    ev = Path(attempt) / 'events'; out = []
    for f in sorted(ev.glob('*.json')):
        d = json.loads(f.read_text())
        if d.get('event') == 'candidate_draw':
            o = d['record'].get('output'); text = o[0] if isinstance(o, list) else o
            if isinstance(text, str):
                thought, _, rest = text.partition('</think>') if '</think>' in text else ('', '', text)
                out.append((call_key(text), thought.replace('<think>', '').strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}; matched = unmatched = empty = 0
    with a.out.open('w') as f:
        for line in a.rows.open():
            r = json.loads(line)
            attempt = os.path.dirname(os.path.dirname(r['screenshot']))
            if attempt not in cache:
                cache[attempt] = draws(attempt)
            key = call_key(r['action']); hit = next((t for k, t in cache[attempt] if k == key), None)
            if hit is None:
                unmatched += 1
            elif not hit:
                empty += 1
            else:
                matched += 1
                r = dict(r, tool_call=r['action'], action=f"{hit}\n{r['action']}", candidates=[f"{hit}\n{r['action']}"])
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(rows=matched + unmatched + empty, with_thought=matched, empty_thought=empty, unmatched=unmatched,
                          attempts=len(cache))))


if __name__ == '__main__':
    main()
