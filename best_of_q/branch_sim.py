#!/usr/bin/env python3
"""Offline simulation of Best-of-Q's iterative training (arXiv 2601.22701 App. A.2) on Zixian's branch states.

    branch_sim.py --rows sim/rows.jsonl --emb sim/emb --out DIR [--cycles 4] [--epsilon 0.5] [--seed 0]

Each fold-0 state (1,367, 347 tasks) has 2-5 distinct candidates that were all executed and continued; every candidate
carries a pool of recorded continuation outcomes on fit draws 0-2 (fit_valid, fit_successes). "Collecting" at a state =
executing ONE candidate and observing ONE continuation outcome, drawn without replacement from that candidate's pool.

  round 0  epsilon-greedy on the actor's ranking: the actor's first proposal with probability 1 - eps, else a random
           other candidate (paper's seed exploration, eps 0.5); one outcome each -> data D; train q1 on D.
  round k  fully greedy, as in the paper: at every state execute argmax_a q_k(s, a) (ties -> actor's first proposal), draw
           one more outcome for it, add to D (all earlier rows kept), retrain q_{k+1} from scratch on D.

A state whose chosen candidate has no outcome left is skipped that round (counted). q_k is train_branch_q.py's model and
loss (frozen Qwen2.5-VL-3B embeddings, Best-of-Q tower, MSE onto the 0/1 outcome). After every round the held-out fold-4
states (342, 85 tasks, never collected on) score each Q by the held-out continuation success (draws 3-7) of the
candidate it picks, next to the actor's first proposal, uniform choice and the oracle. The final q is saved in
serve_q_selector.py's format.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from train_q import Tower, load


def fit_q(S, A, T, act, task, idx, y, d, epochs, seed, dev):
    torch.manual_seed(seed)
    q = Tower(d, 3).to(dev)
    opt = torch.optim.Adam(q.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = 128; steps = epochs * max(1, len(idx) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    idx = torch.tensor(idx, device=dev); y = torch.tensor(y, device=dev, dtype=torch.float32)
    for _ in range(epochs):
        q.train(); perm = torch.randperm(len(idx), device=dev)
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]; i = idx[j]
            loss = ((q(S[i], A[act[i]], T[task[i]]) - y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step(); sched.step()
    q.eval()
    with torch.no_grad():
        return q, q(S, A[act], T[task]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', type=Path, required=True); ap.add_argument('--emb', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--cycles', type=int, default=4)
    ap.add_argument('--epsilon', type=float, default=0.5); ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    rows, S, A, T, a_index, t_index = load(a.rows, a.emb)
    dev = 'cuda'
    S, A, T = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (S, A, T))
    act = torch.tensor([a_index[r['action']] for r in rows], device=dev)
    task = torch.tensor([t_index[r['task']] for r in rows], device=dev)
    states = collections.defaultdict(list)
    for i, r in enumerate(rows):
        states[(r['fold'], r['state_id'])].append(i)
    train = {k[1]: v for k, v in states.items() if k[0] == 0}; held = [v for k, v in states.items() if k[0] == 4]
    pool = {i: [1.0] * r['fit_successes'] + [0.0] * (r['fit_valid'] - r['fit_successes']) for i, r in enumerate(rows)}
    for v in pool.values():
        rng.shuffle(v)
    first = {sid: next(i for i in ix if rows[i]['actor_first']) for sid, ix in train.items()}

    def heldout(pred):
        def rate(i): return rows[i]['holdout_rate']
        pick = [rate(max(ix, key=lambda i: (pred[i], rows[i]['actor_first']))) for ix in held] if pred is not None else None
        return dict(q_pick=float(np.mean(pick)) if pick else None, actor_first=float(np.mean([rate(next(i for i in ix if rows[i]['actor_first'])) for ix in held])),
                    uniform=float(np.mean([np.mean([rate(i) for i in ix]) for ix in held])), oracle=float(np.mean([max(rate(i) for i in ix) for ix in held])))

    D_idx, D_y, log = [], [], []
    print(json.dumps(dict(round='reference', held_states=len(held), **heldout(None))), flush=True)
    pred = None
    for rnd in range(a.cycles + 1):
        added = skipped = 0
        for sid, ix in train.items():
            if rnd == 0:
                others = [i for i in ix if i != first[sid]]
                i = first[sid] if (not others or rng.random() < 1 - a.epsilon) else rng.choice(others)
            else:
                i = max(ix, key=lambda j: (pred[j], rows[j]['actor_first']))
            if not pool[i]:
                skipped += 1; continue
            D_idx.append(i); D_y.append(pool[i].pop()); added += 1
        q, pred = fit_q(S, A, T, act, task, D_idx, D_y, S.shape[1], a.epochs, a.seed + rnd, dev)
        rec = dict(round=rnd, q=f'q{rnd + 1}', policy='eps-greedy on actor' if rnd == 0 else f'argmax q{rnd}', added=added,
                   skipped_no_outcome_left=skipped, data_rows=len(D_idx), data_success_rate=float(np.mean(D_y)), **heldout(pred))
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save(dict(q=q.state_dict(), dim=S.shape[1], embedder=os.environ.get('BOQ_EMBEDDER'), step=rnd,
                        val_v0_success_auc=None, source=f'branch_sim round {rnd}'), a.out / f'q{rnd + 1}.pt')
    (a.out / 'sim-log.json').write_text(json.dumps(log, indent=1) + '\n')
    import shutil; shutil.copy(a.out / f'q{a.cycles + 1}.pt', a.out / 'q_function_best.pt')


if __name__ == '__main__':
    main()
