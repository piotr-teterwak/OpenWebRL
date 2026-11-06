#!/usr/bin/env python3
"""Best-of-Q Q-function (arXiv 2601.22701, Sec. 3 / App. A.3-A.4) on our frozen OpenWebRL-4B-SFT embeddings.

    train_q.py --rows transitions.jsonl --emb DIR --out DIR [--steps 50000] [--tau 0.7]

Architecture: state, action and task embeddings each projected to 1024, concatenated, then an MLP
[1024, 1024, 512, 256, 128, 64] -> 1 with ReLU and Dropout (rate unspecified in the paper; 0.1 here). V(s) is the
same tower without the action. IQL: V by expectile regression (tau) on min-free target Q(s,a); Q by MSE on
r + gamma * (1 - done) * V(s'); target Q updated every 100 steps. Adam 3e-4 cosine, wd 1e-4, clip 1.0, batch 128,
gamma 0.99, sparse reward (1 on task success). 5% of tasks are held out (by task text) for validation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn


class Tower(nn.Module):
    def __init__(self, dim, n_inputs, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(dim, 1024) for _ in range(n_inputs))
        widths = [1024 * n_inputs, 1024, 1024, 512, 256, 128, 64]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers, nn.Linear(64, 1))

    def forward(self, *xs):
        return self.mlp(torch.cat([p(x) for p, x in zip(self.proj, xs)], -1)).squeeze(-1)


def load(rows_path, emb):
    rows = [json.loads(l) for l in rows_path.open()]
    shards = sorted(emb.glob('states-*.npy'))
    states = np.concatenate([np.load(p) for p in shards])
    if len(states) != len(rows):
        raise SystemExit(f'{len(states)} state embeddings for {len(rows)} rows (missing shards?)')
    actions = json.loads((emb / 'actions.json').read_text()); a_index = {t: i for i, t in enumerate(actions)}
    tasks = json.loads((emb / 'tasks.json').read_text()); t_index = {t: i for i, t in enumerate(tasks)}
    return rows, states, np.load(emb / 'actions.npy'), np.load(emb / 'tasks.npy'), a_index, t_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--emb', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows, S, A, T, a_index, t_index = load(a.rows, a.emb)
    dev = 'cuda'
    S, A, T = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (S, A, T))
    act = torch.tensor([a_index[r['action']] for r in rows], device=dev)
    task = torch.tensor([t_index[r['task']] for r in rows], device=dev)
    reward = torch.tensor([r['reward'] for r in rows], device=dev)
    done = torch.tensor([float(r['done']) for r in rows], device=dev)
    bad = [i for i, r in enumerate(rows) if not r['done'] and (i + 1 == len(rows) or rows[i + 1]['episode'] != r['episode']
                                                               or rows[i + 1]['step'] != r['step'] + 1)]
    if bad:  # a truncated episode without a done row would bootstrap from the next episode's first state
        raise SystemExit(f'{len(bad)} non-terminal rows without a next row in the same episode, e.g. {rows[bad[0]]["episode"]}')
    nxt = torch.tensor([i + 1 if not r['done'] else i for i, r in enumerate(rows)], device=dev)  # same episode, ordered
    held_tasks = set(random.Random(a.seed).sample(sorted(t_index), k=max(1, len(t_index) // 20)))
    is_val = torch.tensor([r['task'] in held_tasks for r in rows], device=dev)
    train_idx, val_idx = torch.nonzero(~is_val).squeeze(1), torch.nonzero(is_val).squeeze(1)
    d = S.shape[1]
    q, v = Tower(d, 3).to(dev), Tower(d, 2).to(dev)
    q_target = copy.deepcopy(q).eval()
    opt = torch.optim.Adam(list(q.parameters()) + list(v.parameters()), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def losses(idx):
        s, ac, tk = S[idx], A[act[idx]], T[task[idx]]
        with torch.no_grad():
            tq = q_target(s, ac, tk)
            target = reward[idx] + a.gamma * (1 - done[idx]) * v(S[nxt[idx]], T[task[idx]])
        diff = tq - v(s, tk)
        lv = (torch.abs(a.tau - (diff < 0).float()) * diff ** 2).mean()
        lq = ((q(s, ac, tk) - target) ** 2).mean()
        return lv, lq

    # held-out first steps and their episode outcome (reward on the episode's done row), for V(s0) vs success AUC
    first = torch.tensor([r['step'] == 0 for r in rows], device=dev) & is_val
    ends = {r['episode']: r['reward'] for r in rows if r['done']}
    ep_success = torch.tensor([ends[r['episode']] for r in rows], device=dev)
    a.out.mkdir(parents=True, exist_ok=True); log = (a.out / 'train-log.jsonl').open('w'); best_auc, best_step = -1.0, 0
    for step in range(1, a.steps + 1):
        q.train(); v.train()
        idx = train_idx[torch.randint(len(train_idx), (a.batch,), device=dev)]
        lv, lq = losses(idx)
        opt.zero_grad(); (lv + lq).backward()
        nn.utils.clip_grad_norm_(list(q.parameters()) + list(v.parameters()), 1.0); opt.step(); sched.step()
        if step % 100 == 0:
            q_target.load_state_dict(q.state_dict())
        if step % 1000 == 0 or step == a.steps:
            q.eval(); v.eval()
            with torch.no_grad():
                vl = [losses(val_idx[i:i + 4096]) for i in range(0, len(val_idx), 4096)]
                v0 = v(S[first], T[task[first]]); y = ep_success[first]
                auc = float(((v0[y > 0].unsqueeze(1) > v0[y == 0].unsqueeze(0)).float().mean()) if (y > 0).any() and (y == 0).any() else float('nan'))
            row = dict(step=step, train_v=float(lv), train_q=float(lq), val_v=float(sum(x[0] for x in vl) / len(vl)),
                       val_q=float(sum(x[1] for x in vl) / len(vl)), val_v0_success_auc=auc)
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
            if auc == auc and auc > best_auc:  # keep the best held-out V(s0) AUC checkpoint (final-step model overfits slightly)
                best_auc, best_step = auc, step
                torch.save(dict(q=q.state_dict(), v=v.state_dict(), dim=d, tau=a.tau, gamma=a.gamma, held_tasks=sorted(held_tasks), embedder=os.environ.get('BOQ_EMBEDDER'),
                                step=step, val_v0_success_auc=auc), a.out / 'q_function_best.pt')
    torch.save(dict(q=q.state_dict(), v=v.state_dict(), dim=d, tau=a.tau, gamma=a.gamma, held_tasks=sorted(held_tasks), embedder=os.environ.get('BOQ_EMBEDDER'),
                    step=a.steps), a.out / 'q_function.pt')
    print(json.dumps(dict(best_step=best_step, best_val_v0_success_auc=best_auc)), flush=True)


if __name__ == '__main__':
    main()
