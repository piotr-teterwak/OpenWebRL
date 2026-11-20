#!/usr/bin/env python3
"""Mixed candidate panels at stored collection states (mixed-pool Round 2, 2026-11-19).

    build_mix_panels.py --dataset <NAME-all-shard1/training.jsonl> <...shard2...> --sft sft-s42.jsonl \
        --tasks <collection task file> --out panels-s42.jsonl [--r1-indices 0,1,2] [--shuffle-seed 0]

Each state's panel is the collection's own candidates at --r1-indices (default 0,1,2: the collection samples its 5
candidates iid with per-index seeds, so the first three are an unbiased subset; the executed Sol pick is among them
at ~3/5 of the states) plus every fresh candidate from scripts/sample_sft_candidates.py, shuffled with a fixed
per-state seed, sha256('<shuffle-seed>:<id>'). `sources` records each position's origin ('r1:<i>' collection
candidate i, 'sft:<j>' fresh sample j). States whose panel has fewer than 2 distinct actions (whitespace-normalised,
as ActionSelector's skip_identical) are dropped and counted. Each line carries what a selector request needs (task,
url, executed history, screenshot path) and the raw candidates prepare_arm_joint_data.py --panels builds pairs from:
{id, task_id, turn, source, task, url, history, screenshot, screenshot_sha256, collection_index, sources,
 candidates: [{thought, action}], raw_candidates: [{response, response_token_ids, finish_type}]}
collection_index is the panel position of the collection's executed pick, or null when it is not in the panel.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from openwebrl.arm_inference import split_response


def sha(data): return hashlib.sha256(data).hexdigest()
def action_key(c): return ' '.join((c.get('action') or '').split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', nargs='+', type=Path, required=True)
    ap.add_argument('--sft', type=Path, required=True, help='scripts/sample_sft_candidates.py output')
    ap.add_argument('--tasks', type=Path, required=True, help='task file the collection used (intents)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--r1-indices', default='0,1,2')
    ap.add_argument('--shuffle-seed', default='0')
    a = ap.parse_args()
    keep_r1 = [int(x) for x in a.r1_indices.split(',')]
    intents = {}
    for line in a.tasks.read_text().splitlines():
        t = json.loads(line); intents[t['metadata']['task_id']] = t['metadata'].get('intent') or t['prompt']
    sft = {}
    for line in a.sft.open():
        r = json.loads(line); sft[r['id']] = r
    stats = Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w') as out:
        for dataset in a.dataset:
            for line in dataset.read_text().splitlines():
                row = json.loads(line); key = f"C2:{row['task_id']}:{row['turn']}"; stats['states'] += 1
                if key not in sft: stats['no_sft_samples'] += 1; continue
                s = sft[key]
                if s['source'] != row['source']: raise ValueError(f'{key}: SFT samples are for another turn record')
                raw = Path(row['source']).read_bytes()
                if sha(raw) != row['source_sha256']: raise ValueError(f'{key}: turn record hash')
                turn = json.loads(raw)
                if s['prompt_sha256'] != turn['prompt_sha256']: raise ValueError(f'{key}: prompt differs from the sampled one')
                attempt = Path(row['source']).parent
                history = []   # executed responses, as relabel_with_selector.py and the live ActionSelector send them
                for t in range(turn['turn']):
                    prior = json.loads((attempt / f'turn-{t:04d}.json').read_text())
                    history.append(prior['raw_candidates'][prior['selected_index']]['response'])
                pool = []
                for i in keep_r1:
                    c = turn['raw_candidates'][i]
                    if split_response(c['response']) != turn['candidates'][i]: raise ValueError(f'{key}: stored candidate {i} split differs')
                    pool.append((f'r1:{i}', dict(response=c['response'], response_token_ids=c['response_token_ids'], finish_type=c['finish_type'])))
                for j, c in enumerate(s['candidates']):
                    pool.append((f'sft:{j}', dict(response=c['response'], response_token_ids=c['response_token_ids'], finish_type=c['finish_type'])))
                random.Random(int(sha(f'{a.shuffle_seed}:{key}'.encode()), 16)).shuffle(pool)
                split = [split_response(c['response']) for _, c in pool]
                if len({action_key(c) for c in split}) < 2: stats['fewer_than_2_distinct_actions'] += 1; continue
                sources = [src for src, _ in pool]
                executed = f"r1:{row['selected_index']}"
                out.write(json.dumps(dict(id=key, task_id=row['task_id'], turn=row['turn'], source=row['source'],
                    task=intents[row['task_id']], url=turn['url'], history=history, screenshot=turn['images'][0]['path'],
                    screenshot_sha256=turn['images'][0]['sha256'],
                    collection_index=sources.index(executed) if executed in sources else None, sources=sources,
                    candidates=split, raw_candidates=[c for _, c in pool]), ensure_ascii=False) + '\n')
                stats['panels'] += 1
                stats['r1_distinct_actions_ge2'] += len({action_key(turn['candidates'][i]) for i in keep_r1}) >= 2
    print(json.dumps(dict(stats, out=str(a.out))))


if __name__ == '__main__':
    main()
