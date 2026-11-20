#!/usr/bin/env python3
"""Re-pick the preferred candidate at every stored collection state with another selector.

Reads a collection dataset (arm_c2 build's training.jsonl: one row per executed turn of a successful
episode), rebuilds the exact selection request the collection ARM saw -- task, url, executed history,
screenshot and the five stored candidates -- posts it to a running selector service (/select, e.g.
scripts/serve_sol_selector.py --model gpt-6-luna), and writes one JSON line per state:
{"id": "C2:<task_id>:<turn>", "selected_index": <0-based>, "collection_index": <ARM pick>}.
Feed the result to prepare_arm_joint_data.py --relabel. Resumable: finished ids are skipped.
--panels <scripts/build_mix_panels.py output> instead of --dataset/--tasks: the request's candidates are each panel's
(any count; the selector must accept it, e.g. serve_sol_selector.py --max-candidates 6), task/url/history/screenshot
as stored in the panel line; selected_index is then a panel position.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path

from openwebrl.arm_inference import parse_selection, split_response


def load_states(dataset, tasks_file):
    intents = {}
    for line in Path(tasks_file).read_text().splitlines():
        task = json.loads(line)
        intents[task['metadata']['task_id']] = task['metadata'].get('intent') or task['prompt']
    for line in Path(dataset).read_text().splitlines():
        row = json.loads(line)
        turn = json.loads(Path(row['source']).read_text())
        attempt = Path(row['source']).parent
        history = []
        for t in range(turn['turn']):
            prior = json.loads((attempt / f'turn-{t:04d}.json').read_text())
            history.append(prior['raw_candidates'][prior['selected_index']]['response'])
        yield dict(id=f"C2:{row['task_id']}:{row['turn']}", task=intents[row['task_id']], url=turn['url'],
                   history=history, candidates=turn['candidates'], collection_index=row['selected_index'],
                   png=Path(turn['images'][0]['path']).read_bytes())


def load_panel_states(panels):
    for line in Path(panels).open():
        p = json.loads(line)
        png = Path(p['screenshot']).read_bytes()
        if hashlib.sha256(png).hexdigest() != p['screenshot_sha256']: raise ValueError(f"{p['id']}: screenshot hash")
        yield dict(id=p['id'], task=p['task'], url=p['url'], history=p['history'], candidates=p['candidates'],
                   collection_index=p['collection_index'], png=png)


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', help='collection training.jsonl')
    ap.add_argument('--tasks', help='task file the collection used')
    ap.add_argument('--panels', help='candidate panels (scripts/build_mix_panels.py) instead of --dataset/--tasks')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parallel', type=int, default=16)
    args = ap.parse_args()
    if not args.panels and not (args.dataset and args.tasks): ap.error('--dataset and --tasks (or --panels) are required')
    import httpx
    done = set()
    if args.out.exists():
        done = {json.loads(l)['id'] for l in args.out.read_text().splitlines()}
    source = load_panel_states(args.panels) if args.panels else load_states(args.dataset, args.tasks)
    states = [s for s in source if s['id'] not in done]
    print(f'{len(done)} already labelled, {len(states)} to go', flush=True)
    gate = asyncio.Semaphore(args.parallel)
    failures = 0
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        async def one(s, stream):
            nonlocal failures
            async with gate:
                payload = dict(mode='selection', task=s['task'], url=s['url'], candidates=s['candidates'],
                               history=[split_response(r) for r in s['history']],
                               screenshot=base64.b64encode(s['png']).decode())
                try:
                    response = await client.post(f'http://127.0.0.1:{args.port}/select', json=payload)
                    response.raise_for_status()
                    body = response.json()
                    index = parse_selection(body.get('raw') or '', len(s['candidates']))
                    scores = body.get('scores')
                except Exception as exc:  # a failed state is left unlabelled, never guessed
                    failures += 1
                    print('FAIL', s['id'], type(exc).__name__, str(exc)[:120], flush=True)
                    return
                stream.write(json.dumps(dict(id=s['id'], selected_index=index, collection_index=s['collection_index'],
                                             scores=scores)) + '\n')
                stream.flush()
        with args.out.open('a') as stream:
            await asyncio.gather(*(one(s, stream) for s in states))
    rows = [json.loads(l) for l in args.out.read_text().splitlines()]
    agree = sum(r['selected_index'] == r['collection_index'] for r in rows)
    print(json.dumps(dict(labelled=len(rows), failures=failures, agrees_with_collection_pick=agree / max(1, len(rows)))), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
