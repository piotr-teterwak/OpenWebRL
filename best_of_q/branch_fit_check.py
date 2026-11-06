#!/usr/bin/env python3
"""Can the branch-simulation Q-functions fit their own training states? (check 2 after the flat held-out results)

    branch_fit_check.py --rows sim/rows.jsonl --emb sim/emb --q sim/q [--rows2 ... --emb2 ... --q2 ...]

For every saved round q1..q5, argmax Q over each fold-0 (training) state's candidates is scored three ways:
  fit      the picked candidate's pooled fit-draw success rate (draws 0-2, where every training target was drawn from)
  holdout  the picked candidate's held-out success rate on the SAME training states (draws 3-7, never used in training)
next to uniform choice, the actor's first proposal, and the oracle under each rate. Fold-4 numbers repeat the held-out
table for reference. CPU only.
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import torch

from train_q import Tower, load


def report(label, rows_p, emb_p, q_dir):
    rows, S, A, T, a_index, t_index = load(rows_p, emb_p)
    S, A, T = (torch.tensor(x, dtype=torch.float32) for x in (S, A, T))
    act = torch.tensor([a_index[r['action']] for r in rows]); task = torch.tensor([t_index[r['task']] for r in rows])
    states = collections.defaultdict(list)
    for i, r in enumerate(rows):
        states[(r['fold'], r['state_id'])].append(i)
    groups = {f: [ix for (fold, _), ix in states.items() if fold == f] for f in (0, 4)}
    fit = lambda i: rows[i]['fit_successes'] / rows[i]['fit_valid']
    hold = lambda i: rows[i]['holdout_rate']
    def ref(g, rate):
        return dict(uniform=np.mean([np.mean([rate(i) for i in ix]) for ix in g]),
                    actor_first=np.mean([rate(next(i for i in ix if rows[i]['actor_first'])) for ix in g]),
                    oracle=np.mean([max(rate(i) for i in ix) for ix in g]))
    print(f'\n### {label}')
    for name, g, rate in (('train states, fit rate', groups[0], fit), ('train states, held-out draws', groups[0], hold), ('held-out states', groups[4], hold)):
        r = ref(g, rate); print(f'  {name:30} uniform {100*r["uniform"]:.1f}  actor-first {100*r["actor_first"]:.1f}  oracle {100*r["oracle"]:.1f}')
    for k in range(1, 6):
        p = Path(q_dir) / f'q{k}.pt'
        if not p.exists():
            continue
        ck = torch.load(p, map_location='cpu'); q = Tower(ck['dim'], 3); q.load_state_dict(ck['q']); q.eval()
        with torch.no_grad():
            pred = q(S, A[act], T[task]).numpy()
        pick = lambda ix: max(ix, key=lambda i: (pred[i], rows[i]['actor_first']))
        out = [f'{100*np.mean([rate(pick(ix)) for ix in g]):.1f}' for g, rate in ((groups[0], fit), (groups[0], hold), (groups[4], hold))]
        # how well do Q's scores track the fit rates within a training state (Spearman-like: fraction of ordered pairs agreeing)?
        agree = tot = 0
        for ix in groups[0]:
            for a_ in ix:
                for b_ in ix:
                    if fit(a_) > fit(b_):
                        tot += 1; agree += pred[a_] > pred[b_]
        print(f'  q{k}: pick on train (fit) {out[0]} | train (held-out draws) {out[1]} | held-out states {out[2]} | within-state pair order agrees {100*agree/max(1,tot):.1f}% of {tot}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', nargs=4, action='append', metavar=('LABEL', 'ROWS', 'EMB', 'QDIR'), required=True)
    a = ap.parse_args()
    for label, rows_p, emb_p, q_dir in a.set:
        report(label, Path(rows_p), Path(emb_p), q_dir)


if __name__ == '__main__':
    main()
