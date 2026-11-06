#!/usr/bin/env python3
"""Where does the branch-simulation Q-function fail on unseen states?

    branch_failure_modes.py ROWS EMB QDIR

On fold-4 (held-out) states and, for contrast, fold-0 (training) states, for the final round q5:
  * spread: Q variance between states vs within states (does Q mostly score the state, not the action?)
  * pair accuracy: fraction of within-state candidate pairs with different held-out rates that Q orders correctly
  * action types: what Q picks vs the oracle-best candidate (click / scroll / type / done / other), and each type's
    held-out success when chosen
  * a type-prior baseline: pick by the training states' mean fit rate of the candidate's action type
  * task leakage: on training states, does Q's state-mean track the task's training success rate (memorised task value)?
CPU only.
"""
import collections, json, re, sys
from pathlib import Path

import numpy as np
import torch

from train_q import Tower, load

TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def kind(action):
    m = TOOL.search(action or '')
    try:
        name = json.loads(m.group(1)).get('name') if m else None
    except json.JSONDecodeError:
        name = None
    return {'click': 'click', 'scroll': 'scroll', 'write': 'type', 'done': 'done', 'goto_url': 'goto', 'go_back': 'back'}.get(name, 'other')


def main():
    rows_p, emb_p, q_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows, S, A, T, a_index, t_index = load(rows_p, emb_p)
    S, A, T = (torch.tensor(x, dtype=torch.float32) for x in (S, A, T))
    act = torch.tensor([a_index[r['action']] for r in rows]); task = torch.tensor([t_index[r['task']] for r in rows])
    ck = torch.load(q_dir / 'q5.pt', map_location='cpu'); q = Tower(ck['dim'], 3); q.load_state_dict(ck['q']); q.eval()
    with torch.no_grad():
        pred = q(S, A[act], T[task]).numpy()
    states = collections.defaultdict(list)
    for i, r in enumerate(rows):
        states[(r['fold'], r['state_id'])].append(i)
    k = [kind(r.get('tool_call') or r['action']) for r in rows]
    hold = lambda i: rows[i]['holdout_rate']; fit = lambda i: rows[i]['fit_successes'] / rows[i]['fit_valid']
    train_type = collections.defaultdict(list)
    for (fold, _), ix in states.items():
        if fold == 0:
            for i in ix: train_type[k[i]].append(fit(i))
    prior = {t: float(np.mean(v)) for t, v in train_type.items()}
    print('training type prior (mean fit rate):', {t: round(v, 3) for t, v in sorted(prior.items(), key=lambda x: -x[1])},
          'counts', {t: len(v) for t, v in train_type.items()})
    for fold, name in ((0, 'TRAINING states'), (4, 'HELD-OUT states')):
        g = [ix for (f, _), ix in states.items() if f == fold]
        means = np.array([np.mean(pred[ix]) for ix in g]); within = np.mean([np.var(pred[ix]) for ix in g])
        pairs = [(a, b) for ix in g for a in ix for b in ix if hold(a) > hold(b)]
        acc = np.mean([pred[a] > pred[b] for a, b in pairs])
        pick = [max(ix, key=lambda i: (pred[i], rows[i]['actor_first'])) for ix in g]
        best = [max(ix, key=lambda i: hold(i)) for ix in g]
        tpick = [max(ix, key=lambda i: (prior.get(k[i], 0), rows[i]['actor_first'])) for ix in g]
        print(f'\n## {name} ({len(g)}): Q variance between states {np.var(means):.4f} vs within {within:.4f} | '
              f'pair order correct {100*acc:.1f}% of {len(pairs)} (chance 50)')
        print(f'   held-out success: Q pick {100*np.mean([hold(i) for i in pick]):.1f} | type-prior pick {100*np.mean([hold(i) for i in tpick]):.1f} | '
              f'uniform {100*np.mean([np.mean([hold(i) for i in ix]) for ix in g]):.1f} | oracle {100*np.mean([hold(i) for i in best]):.1f}')
        cp, cb = collections.Counter(k[i] for i in pick), collections.Counter(k[i] for i in best)
        avail = collections.Counter(k[i] for ix in g for i in ix)
        print('   action type   available  Q-picks  oracle-best  held-out success when Q picks it')
        for t in sorted(avail, key=lambda t: -avail[t]):
            sel = [hold(i) for i in pick if k[i] == t]
            print(f'   {t:12} {avail[t]:9} {cp[t]:8} {cb[t]:12}  {100*np.mean(sel):.1f}' if sel else f'   {t:12} {avail[t]:9} {cp[t]:8} {cb[t]:12}  -')
        if fold == 0:
            by_task = collections.defaultdict(list)
            for ix in g: by_task[rows[ix[0]]['task_id']].append((np.mean(pred[ix]), np.mean([fit(i) for i in ix])))
            x = [np.mean([a for a, _ in v]) for v in by_task.values()]; y = [np.mean([b for _, b in v]) for v in by_task.values()]
            print(f'   task-level: corr(Q state-mean, task mean fit rate) = {np.corrcoef(x, y)[0, 1]:.2f} over {len(x)} tasks')
        else:
            g4 = collections.defaultdict(list)
            for ix in g: g4[rows[ix[0]]['task_id']].append((np.mean(pred[ix]), np.mean([hold(i) for i in ix])))
            x = [np.mean([a for a, _ in v]) for v in g4.values()]; y = [np.mean([b for _, b in v]) for v in g4.values()]
            print(f'   task-level: corr(Q state-mean, task mean held-out rate) = {np.corrcoef(x, y)[0, 1]:.2f} over {len(x)} tasks')


if __name__ == '__main__':
    main()
