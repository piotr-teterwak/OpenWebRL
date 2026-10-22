#!/usr/bin/env python3
"""Host-grouped train/held-out split of the C2 1A dataset, for the SFT learning-rate grid.

Holds out whole websites (start_url host) until ~--holdout-frac of the 8,394 turns are
held out, so held-out states come from sites the student never trained on. Writes:

  arm/      training.jsonl + dataset-audit.json  train split, ARM-picked targets (1A)
  random/   training.jsonl + dataset-audit.json  train split, uniform random usable candidate
  heldout.jsonl                                    held-out rows (ARM pick kept), for scoring
  split.json                                       held-out hosts and counts

Both training sets keep 1A's row order and differ only in selected_index, as in
build_c2_random_control.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from openwebrl.arm_c2 import hostname


def sha(data): return hashlib.sha256(data).hexdigest()


def write_set(directory, rows, source_audit, label, extra):
    directory.mkdir(parents=True, exist_ok=False)
    dataset = directory / 'training.jsonl'
    dataset.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    audit = dict(source_audit, dataset=str(dataset), dataset_sha256=sha(dataset.read_bytes()),
                 retained_turns=len(rows), ablation=label, source_dataset_sha256=source_audit['dataset_sha256'], **extra)
    (directory / 'dataset-audit.json').write_text(json.dumps(audit, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    source_audit = json.loads((args.c2 / 'dataset-audit.json').read_text())
    source = Path(source_audit['dataset'])
    if sha(source.read_bytes()) != source_audit['dataset_sha256']: raise ValueError('Source dataset checksum mismatch')
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    host_of = {}
    for line in (args.c2 / 'tasks.jsonl').read_text().splitlines():
        task = json.loads(line)
        host_of[task['task_id']] = hostname(task['metadata']['start_url'])
    turns_per_host = {}
    for row in rows: turns_per_host[host_of[row['task_id']]] = turns_per_host.get(host_of[row['task_id']], 0) + 1
    hosts = sorted(turns_per_host)
    random.Random(args.seed).shuffle(hosts)
    held, count = set(), 0
    for host in hosts:
        if count >= args.holdout_frac * len(rows): break
        held.add(host); count += turns_per_host[host]

    train = [r for r in rows if host_of[r['task_id']] not in held]
    heldout = [r for r in rows if host_of[r['task_id']] in held]
    rng = random.Random(args.seed)
    randomized, same = [], 0
    for row in train:
        captured = json.loads(Path(row['source']).read_text())
        usable = [i for i, f in enumerate(captured['finish_types']) if f == 'stop']
        if row['selected_index'] not in usable: raise ValueError(f"ARM pick unusable: {row['source']}")
        pick = rng.choice(usable); same += pick == row['selected_index']
        randomized.append(dict(row, selected_index=pick, arm_selected_index=row['selected_index']))

    split = dict(seed=args.seed, holdout_frac=args.holdout_frac, heldout_hosts=sorted(held),
                 train_turns=len(train), heldout_turns=len(heldout),
                 train_tasks=len({r['task_id'] for r in train}), heldout_tasks=len({r['task_id'] for r in heldout}),
                 random_index_equals_arm_pick=same)
    args.output.mkdir(parents=True, exist_ok=False)
    write_set(args.output / 'arm', train, source_audit, '1A-heldout-split', dict(split='train'))
    write_set(args.output / 'random', randomized, source_audit, 'random-candidate-heldout-split', dict(split='train', seed=args.seed))
    (args.output / 'heldout.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in heldout))
    (args.output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    print(json.dumps({k: v for k, v in split.items() if k != 'heldout_hosts'}))


if __name__ == '__main__':
    main()
