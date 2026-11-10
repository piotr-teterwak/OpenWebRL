#!/usr/bin/env python3
"""Split the C2 training-task pool into two disjoint halves for iterative on-policy rounds.

    split_task_pool.py --tasks <c2 tasks.jsonl> --validation <joint-data/matched/joint_pairs.validation.jsonl> --out DIR [--seed 0]

Drops the tasks behind the C2 validation pairs (so collected pairs never land in validation; 2,091 -> 1,978 tasks,
David's training pool), then stratifies by website: each host's tasks are shuffled (seeded) and dealt alternately to
halves A and B, with the starting half alternating between hosts, so both halves cover the same sites with different
tasks and differ in size by at most one. Writes tasks-A.jsonl, tasks-B.jsonl, tasks-AB.jsonl (both halves, for the
single-round data-matched control) and split-report.json, in the C2 tasks.jsonl format c2_collect reads.
"""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import urlsplit


def host(task):
    url = (task.get('metadata') or {}).get('start_url') or ''
    return (urlsplit(url if '//' in url else 'https://' + url).netloc or '').lower().removeprefix('www.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', type=Path, required=True); ap.add_argument('--validation', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    lines = a.tasks.read_text().splitlines(); tasks = [json.loads(l) for l in lines]
    held = {json.loads(l)['episode_id'].split(':', 1)[1] for l in a.validation.read_text().splitlines()
            if json.loads(l)['id'].startswith('C2:')}
    keep = [(t, l) for t, l in zip(tasks, lines) if t['task_id'] not in held]
    by_host = collections.defaultdict(list)
    for t, l in keep:
        by_host[host(t)].append((t, l))
    rng = random.Random(a.seed); halves = {'A': [], 'B': []}; start = 0
    for h in sorted(by_host):
        items = by_host[h]; rng.shuffle(items)
        for i, item in enumerate(items):
            halves['AB'[(i + start) % 2]].append(item)
        start ^= len(items) % 2  # an odd host leaves the next host starting on the other half
    for name, items in halves.items():
        (a.out / f'tasks-{name}.jsonl').write_text(''.join(l + '\n' for _, l in items))
    (a.out / 'tasks-AB.jsonl').write_text(''.join(l + '\n' for _, l in halves['A'] + halves['B']))
    rep = dict(source=str(a.tasks), source_sha256=hashlib.sha256(a.tasks.read_bytes()).hexdigest(), source_tasks=len(tasks),
               validation_tasks_dropped=len(tasks) - len(keep), kept=len(keep), hosts=len(by_host), seed=a.seed,
               A=len(halves['A']), B=len(halves['B']),
               hosts_in_both=len({host(t) for t, _ in halves['A']} & {host(t) for t, _ in halves['B']}),
               task_overlap=len({t['task_id'] for t, _ in halves['A']} & {t['task_id'] for t, _ in halves['B']}))
    (a.out / 'split-report.json').write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep))


if __name__ == '__main__':
    main()
