#!/usr/bin/env python3
"""Best-of-Q's Q-network trained on Zixian's branch states with one sampled continuation outcome per candidate.

    train_branch_q.py --rows rows.jsonl --emb DIR --out DIR [--epochs 200] [--seed 0] [--epsilon 0.5]

Same tower as train_q.py (state, action and task embeddings from the frozen Qwen2.5-VL-3B, each projected to 1024, MLP
[1024, 1024, 512, 256, 128, 64] -> 1, dropout 0.1), same optimizer (Adam 3e-4 cosine, wd 1e-4, clip 1.0, batch 128).
The target is direct: every row is an executed candidate whose continuation outcome is known, so Q(s, a) is regressed
(MSE) onto that one 0/1 outcome - the Monte-Carlo return IQL would bootstrap towards, without needing next states.
10% of task groups are held out; the checkpoint with the best held-out within-state pick (does argmax Q choose a
candidate whose pooled success rate is the state's best?) is kept as q_function_best.pt, in the format
serve_q_selector.py loads.

--epsilon e simulates Best-of-Q's epsilon-greedy collection on these states: each training state keeps ONE candidate,
the actor's own first proposal (actor_first_index.json, from her panels) with probability 1 - e, else a uniformly
random other candidate, with that candidate's one sampled outcome. Held-out evaluation still uses all candidates.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', type=Path, required=True); ap.add_argument('--emb', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows, S, A, T, a_index, t_index = load(a.rows, a.emb)
    dev = 'cuda'
    S, A, T = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (S, A, T))
    act = torch.tensor([a_index[r['action']] for r in rows], device=dev)
    task = torch.tensor([t_index[r['task']] for r in rows], device=dev)
    y = torch.tensor([r['target'] for r in rows], device=dev)
    groups = sorted({r['task_id'] for r in rows})
    held = set(random.Random(a.seed).sample(groups, k=max(1, len(groups) // 10)))
    is_val = torch.tensor([r['task_id'] in held for r in rows], device=dev)
    tr = torch.nonzero(~is_val).squeeze(1)
    if a.epsilon is not None:
        first = json.loads((a.rows.parent / 'actor_first_index.json').read_text()); rng = random.Random(a.seed + 1)
        per = collections.defaultdict(list)
        for i in tr.tolist():
            per[rows[i]['state_id']].append(i)
        keep = []
        for sid, ix in per.items():
            g = [i for i in ix if rows[i]['candidate_index'] == first[sid]]
            others = [i for i in ix if i not in g]
            keep.append(g[0] if g and (not others or rng.random() < 1 - a.epsilon) else rng.choice(others or ix))
        tr = torch.tensor(sorted(keep), device=dev)
        print(json.dumps(dict(epsilon=a.epsilon, train_states=len(per), train_rows=len(tr))), flush=True)
    by_state = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['task_id'] in held:
            by_state[r['state_id']].append(i)
    val_states = [ix for ix in by_state.values() if len(ix) > 1 and len({rows[i]['pooled'] for i in ix}) > 1]
    d = S.shape[1]; q = Tower(d, 3).to(dev)
    opt = torch.optim.Adam(q.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = a.epochs * max(1, len(tr) // a.batch); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    a.out.mkdir(parents=True, exist_ok=True); log = (a.out / 'train-log.jsonl').open('w'); best, step = -1.0, 0
    for epoch in range(1, a.epochs + 1):
        q.train(); perm = tr[torch.randperm(len(tr), device=dev)]
        for b in range(0, len(perm) - a.batch + 1, a.batch):
            i = perm[b:b + a.batch]
            loss = ((q(S[i], A[act[i]], T[task[i]]) - y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if epoch % 5 == 0 or epoch == a.epochs:
            q.eval()
            with torch.no_grad():
                pred = q(S, A[act], T[task]).cpu().numpy()
            hits = [max(rows[i]['pooled'] for i in ix) == rows[max(ix, key=lambda i: pred[i])]['pooled'] for ix in val_states]
            uniform = [sum(rows[i]['pooled'] == max(rows[j]['pooled'] for j in ix) for i in ix) / len(ix) for ix in val_states]
            row = dict(epoch=epoch, step=step, train_mse=float(loss), val_pick_best=float(np.mean(hits)) if hits else None,
                       val_uniform=float(np.mean(uniform)) if uniform else None, val_states=len(val_states))
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
            if hits and np.mean(hits) > best:
                best = float(np.mean(hits))
                torch.save(dict(q=q.state_dict(), dim=d, embedder=os.environ.get('BOQ_EMBEDDER'), step=step,
                                val_v0_success_auc=None, val_pick_best=best, source='zixian branch states, 1 sampled outcome'),
                           a.out / 'q_function_best.pt')
    torch.save(dict(q=q.state_dict(), dim=d, embedder=os.environ.get('BOQ_EMBEDDER'), step=step), a.out / 'q_function.pt')
    if not (a.out / 'q_function_best.pt').exists():
        import shutil; shutil.copy(a.out / 'q_function.pt', a.out / 'q_function_best.pt')
    print(json.dumps(dict(best_val_pick_best=best, held_groups=len(held), train_rows=len(tr))), flush=True)


if __name__ == '__main__':
    main()
