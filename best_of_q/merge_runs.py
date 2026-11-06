#!/usr/bin/env python3
"""Concatenate per-run Best-of-Q transitions and embeddings into one training set for train_q.py.

    merge_runs.py --out DIR [--max-states N --seed S] RUN_DIR [RUN_DIR ...]

Each RUN_DIR holds boq/rows.jsonl (extract_boq.py) and boq/emb/ (embed.py states + texts on those rows). Writes
DIR/rows.jsonl and DIR/emb/{states-0000000-NNNNNNN.npy, actions.json/.npy, tasks.json/.npy} with rows and state
vectors in the same order; text embeddings are the union over runs (identical strings embed identically).
--max-states keeps a seeded random subset of whole episodes until it holds at least N executed states (a
data-matched Q-function, e.g. N = 1,380 = the branch-state count of Zixian's selection-head study).
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-states', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('runs', type=Path, nargs='+')
    a = ap.parse_args()
    (a.out / 'emb').mkdir(parents=True, exist_ok=True)
    states, texts, n = [], {'actions': {}, 'tasks': {}}, 0
    keep = None
    if a.max_states:
        import random
        eps = {}
        for run in a.runs:
            for i, line in enumerate((run / 'boq' / 'rows.jsonl').read_text().splitlines()):
                eps.setdefault((str(run), json.loads(line)['episode']), []).append(i)
        order = sorted(eps); random.Random(a.seed).shuffle(order); keep, total = set(), 0
        for e in order:
            if total >= a.max_states:
                break
            keep.add(e); total += len(eps[e])
    with (a.out / 'rows.jsonl').open('w') as f:
        for run in a.runs:
            rows = (run / 'boq' / 'rows.jsonl').read_text().splitlines()
            emb = run / 'boq' / 'emb'
            s = np.concatenate([np.load(p) for p in sorted(emb.glob('states-*.npy'))])
            if len(s) != len(rows):
                raise SystemExit(f'{run}: {len(s)} state vectors for {len(rows)} rows')
            if keep is not None:
                idx = [i for i, line in enumerate(rows) if (str(run), json.loads(line)['episode']) in keep]
                rows, s = [rows[i] for i in idx], s[idx]
            if not rows:
                continue
            states.append(s); f.write('\n'.join(rows) + '\n'); n += len(rows)
            for name in texts:
                keys = json.loads((emb / f'{name}.json').read_text()); vecs = np.load(emb / f'{name}.npy')
                for k, v in zip(keys, vecs):
                    texts[name].setdefault(k, v)
    np.save(a.out / 'emb' / f'states-0000000-{n:07d}.npy', np.concatenate(states))
    for name, d in texts.items():
        keys = sorted(d)
        (a.out / 'emb' / f'{name}.json').write_text(json.dumps(keys))
        np.save(a.out / 'emb' / f'{name}.npy', np.stack([d[k] for k in keys]))
    print(json.dumps(dict(runs=len(a.runs), rows=n, episodes=len(keep) if keep is not None else None, actions=len(texts['actions']), tasks=len(texts['tasks']), out=str(a.out))))


if __name__ == '__main__':
    main()
