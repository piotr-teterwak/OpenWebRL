#!/usr/bin/env python3
"""Label-file subsets for Selection-ARM data scaling, fed to action-reward-models' unchanged
data_generation/openwebrl_actor/build_selection_sft.py (run with --holdout 0).

Input is Piotr's OpenWebRL actor release (states_full / candidates_merged / labels_drawlevel). The
split is joint-data-v2's task-grouped split, not the builder's 2%-of-draws holdout (draws of one
state can land on both sides of that): validation task groups are the test set, and states that
joint-data-v2 excluded or never assigned are dropped from both sides.

Two independent axes over the train pool, each nested by a seeded hash:
  states-F   the fraction F of train task groups, with every labelled draw of their states
  draws-K    every train state, at most K draws per state (draw rank by seeded hash)
states-1 and draws-16 are both the full pool.

Also writes states_local.jsonl: states_full with screenshot paths pointed at this cluster's copy
(the release stores the author's /projectnb paths; the builder copies images[-1]).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

RELEASE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor')
SPLIT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/split-manifest.jsonl')


def rank(value):
    return hashlib.sha256(('arm-scaling:42:' + value).encode()).hexdigest()


def unit(value):
    return int(rank(value)[:15], 16) / 16 ** 15


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', type=Path, default=RELEASE)
    ap.add_argument('--split', type=Path, default=SPLIT)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fractions', default='0.0625,0.125,0.25,0.5,1')
    ap.add_argument('--draws', default='1,2,4,8,16')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with (a.release / 'states_full.jsonl').open() as src, (a.out / 'states_local.jsonl').open('w') as dst:
        for line in src:
            s = json.loads(line)
            s['images'] = [str(a.release / 'state_images' / Path(p).name) for p in s['images']]
            assert Path(s['images'][-1]).exists(), s['images'][-1]
            dst.write(json.dumps(s, ensure_ascii=False) + '\n')

    split = {}
    for line in a.split.open():
        r = json.loads(line)
        if r['source'] == 'Piotr' and not r['excluded']:
            split[r['id'].removeprefix('Piotr:')] = (r['split'], r['task_group'])
    labels = [(l, json.loads(l)) for l in (a.release / 'labels_drawlevel.jsonl').open()]
    pool, test, dropped = defaultdict(list), [], Counter()
    for line, lab in labels:
        base = lab['state_id'].split('#')[0]
        if lab.get('selection') is None:
            dropped['no_selection'] += 1
        elif base not in split:
            dropped['excluded_or_unassigned'] += 1
        elif split[base][0] == 'validation':
            test.append(line)
        else:
            pool[base].append((rank(lab['state_id']), line))
    for draws in pool.values():
        draws.sort()
    (a.out / 'labels_test.jsonl').write_text(''.join(test))

    points = {}
    for f in [float(x) for x in a.fractions.split(',')]:
        keep = [b for b in pool if unit(split[b][1]) < f]
        points[f'states-{f:g}'] = [line for b in keep for _, line in pool[b]]
    for k in [int(x) for x in a.draws.split(',')]:
        points[f'draws-{k}'] = [line for b in pool for _, line in pool[b][:k]]
    summary = dict(test_draws=len(test), test_states=len({json.loads(l)['state_id'].split('#')[0] for l in test}),
                   pool_states=len(pool), dropped=dict(dropped), points={})
    for name, lines in points.items():
        (a.out / f'labels_{name}.jsonl').write_text(''.join(lines))
        states = {json.loads(l)['state_id'].split('#')[0] for l in lines}
        summary['points'][name] = dict(draws=len(lines), states=len(states))
    (a.out / 'subsets.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
