#!/usr/bin/env python3
"""Offline round-2 probe: does a trained actor keep candidate diversity, and does the ARM still discriminate?

Replays stored C2 states (prompt + screenshot + executed history, no browser) against
a served actor and the served selection ARM:

  replay   send the stored base-actor candidates to the ARM and check it reproduces the
           stored pick -- validates that the request is rebuilt faithfully. Run once.
  sample   draw 5 fresh candidates per state from --actor (same sampling as collection),
           then ask the ARM to pick. Run once per actor, base included as the control.

Writes one JSONL row per state to --out and prints a summary: share of states with
>1 distinct action, mean distinct actions, ARM pick-position histogram, parse fallbacks.
"""
import argparse
import asyncio
import base64
import collections
import json
from pathlib import Path
import random
import re

from openwebrl.arm_inference import candidate_seed, parse_selection, split_response


def norm(action): return re.sub(r'\s+', ' ', action.strip())


def load_states(c2, n, seed):
    rows = [json.loads(line) for line in (c2 / 'training.jsonl').read_text().splitlines()]
    intents = {}
    for line in (c2 / 'tasks.jsonl').read_text().splitlines():
        task = json.loads(line)
        intents[task['task_id']] = task['metadata'].get('intent') or task['prompt']
    states = []
    for row in random.Random(seed).sample(rows, n):
        turn = json.loads(Path(row['source']).read_text())
        attempt = Path(row['source']).parent
        history = []
        for t in range(turn['turn']):
            prior = json.loads((attempt / f'turn-{t:04d}.json').read_text())
            history.append(prior['raw_candidates'][prior['selected_index']]['response'])
        png = Path(turn['images'][0]['path']).read_bytes()
        states.append(dict(task_id=turn['task_id'], turn=turn['turn'], prompt=turn['prompt'], url=turn['url'],
                           task=intents[turn['task_id']], history=history, png=base64.b64encode(png).decode(),
                           stored_candidates=turn['candidates'], stored_index=turn['selected_index']))
    return states


async def select(client, port, state, candidates):
    payload = dict(mode='selection', task=state['task'], url=state['url'], candidates=candidates,
                   history=[split_response(r) for r in state['history']], screenshot=state['png'])
    response = await client.post(f'http://127.0.0.1:{port}/select', json=payload)
    response.raise_for_status()
    raw = response.json().get('raw') or ''
    try:
        return parse_selection(raw, len(candidates)), None
    except ValueError as exc:
        return 0, str(exc)  # collection's own fallback: execute candidate 1


async def sample(client, port, state, k, seed):
    async def one(j):
        params = dict(temperature=.7, top_p=.9, max_new_tokens=1024,
                      sampling_seed=candidate_seed(seed, state['task_id'], state['turn'], j))
        response = await client.post(f'http://127.0.0.1:{port}/generate', json=dict(
            text=state['prompt'], sampling_params=params, image_data=[f"data:image/png;base64,{state['png']}"]))
        response.raise_for_status()
        out = response.json()
        # Same normalisation as generate_browser._ensure_im_end_w_new_line, so the ARM
        # sees candidates exactly as it did during collection.
        text = out['text'].rstrip()
        text = (text if text.endswith('<|im_end|>') else text + '<|im_end|>') + '\n'
        return text, out.get('meta_info', {}).get('finish_reason', {}).get('type', 'stop')
    return await asyncio.gather(*(one(j) for j in range(k)))


def summarize(rows, label):
    distinct = [len({norm(c['action']) for c in r['candidates']}) for r in rows]
    picks = collections.Counter(r['picked'] for r in rows)
    finishes = [f for r in rows for f in (r.get('finish_types') or [])]
    calls = [c['action'].count('<tool_call>') for r in rows for c in r['candidates']]
    out = dict(label=label, states=len(rows), share_multi_distinct=sum(d > 1 for d in distinct) / len(rows),
               # Generation health: base hits the 1024-token cap ~1% of the time in browser evals.
               length_cap_rate=(sum(f == 'length' for f in finishes) / len(finishes)) if finishes else None,
               mean_tool_calls=sum(calls) / len(calls),
               mean_distinct=sum(distinct) / len(rows), pick_position=[picks[i] / len(rows) for i in range(5)],
               fallbacks=sum(r['fallback'] is not None for r in rows))
    if 'stored_index' in rows[0] and rows[0].get('mode') == 'replay':
        out['replay_agreement'] = sum(r['picked'] == r['stored_index'] for r in rows) / len(rows)
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--c2', type=Path, required=True, help='C2 collection run holding training.jsonl + tasks.jsonl')
    ap.add_argument('--mode', choices=['replay', 'sample'], required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--actor-port', type=int)
    ap.add_argument('--selector-port', type=int, required=True)
    ap.add_argument('--states', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--parallel', type=int, default=16)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import httpx
    states = load_states(args.c2, args.states, args.seed)
    gate = asyncio.Semaphore(args.parallel)
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        async def one(state):
            async with gate:
                if args.mode == 'replay':
                    candidates, finish = state['stored_candidates'], None
                else:
                    outputs = await sample(client, args.actor_port, state, 5, args.seed)
                    candidates, finish = [split_response(text) for text, _ in outputs], [f for _, f in outputs]
                picked, fallback = await select(client, args.selector_port, state, candidates)
                return dict(mode=args.mode, label=args.label, task_id=state['task_id'], turn=state['turn'],
                            candidates=candidates, finish_types=finish, picked=picked, fallback=fallback,
                            stored_index=state['stored_index'],
                            same_action_as_stored_pick=norm(candidates[picked]['action'])
                                == norm(state['stored_candidates'][state['stored_index']]['action']))
        rows = await asyncio.gather(*(one(s) for s in states))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = summarize(rows, args.label)
    summary['same_action_as_stored_base_pick'] = sum(r['same_action_as_stored_pick'] for r in rows) / len(rows)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
