#!/usr/bin/env python3
"""All iterative-round models over every eval pass they have, OM2W + DeepShop, T 1.0, task-clustered bootstrap.
For a pair (A, B), each task contributes the mean over shared passes (both valid) of success(A) - success(B); tasks are
resampled (10k), so each task's runs across passes stay together. Per-pass differences are listed so the pass-to-pass
spread is visible: within-pass task bootstraps understate it (2026-11-14: r2 - C2 OM2W was -5.2 on s2-a/b, +5.8 on rs-a).
    python3 tillicum/iter_pooled.py"""
import glob, json, random, collections
E = '/gpfs/scrubbed/piotrt/owrl/eval'; BENCH = ['online-mind2web', 'deepshop']
M = {'base': 'base', 'C2': 'r2-sol-r1-all-m0.3', 'r2': 'r2-sol2-r2', 'ctrl': 'r2-sol2-ctrl', 'refsft': 'r2-sol2-r2-refsft',
     'refsft@196': 'r2-sol2-r2-refsft@196', 'r2@196': 'r2-sol2-r2@196'}
PASSES = ['s2-a', 's2-b', 'rs-a', 'rs-b']
def load(m, p, b):
    out = {}
    for f in glob.glob(f'{E}/pass{p}/{m}/{b}/results/*.json'):
        r = json.load(open(f)); v = str(r.get('valid')) == 'True'; out[str(r['task_id'])] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
D = {k: {(p, b): load(m, p, b) for p in PASSES for b in BENCH} for k, m in M.items()}
have = {k: [p for p in PASSES if all(len(D[k][(p, b)]) >= 100 for b in BENCH)] for k in M}
print('passes per model:', have)
rng = random.Random(0)
def boot(vals):
    if not vals: return (float('nan'),) * 3
    n = len(vals); bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return 100 * sum(vals) / n, 100 * bs[250], 100 * bs[9749]
def pair(a, c, benches):
    shared = [p for p in have[a] if p in have[c]]
    per_task = []; per_pass = collections.defaultdict(list)
    for b in benches:
        tasks = set().union(*[set(D[a][(p, b)]) for p in shared]) if shared else set()
        for t in tasks:
            d = []
            for p in shared:
                x, y = D[a][(p, b)].get(t), D[c][(p, b)].get(t)
                if x and y and x[0] and y[0]: d.append(int(x[1]) - int(y[1])); per_pass[p].append(int(x[1]) - int(y[1]))
            if d: per_task.append(sum(d) / len(d))
    pp = {p: round(100 * sum(v) / len(v), 1) for p, v in per_pass.items()}
    return boot(per_task), len(per_task), shared, pp
print('\nsuccess (valid-only), mean over passes')
for k in M:
    row = []
    for b in BENCH:
        eps = [x for p in have[k] for x in D[k][(p, b)].values()]; v = sum(x[0] for x in eps); s = sum(x[1] for x in eps)
        row.append(f'{b} {100*s/max(v,1):5.1f}')
    print(f'  {k:11s} ({len(have[k])} passes)  ' + '   '.join(row))
for scope in BENCH + ['pooled']:
    benches = BENCH if scope == 'pooled' else [scope]
    print(f'\n{scope}: paired difference, task-clustered 95% CI   [per-pass differences]')
    for a, c in [('C2', 'base'), ('ctrl', 'base'), ('ctrl', 'C2'), ('r2', 'C2'), ('refsft', 'C2'), ('refsft@196', 'C2'),
                 ('r2@196', 'C2'), ('refsft', 'r2'), ('refsft@196', 'refsft'), ('refsft@196', 'r2@196')]:
        (x, lo, hi), n, shared, pp = pair(a, c, benches)
        sig = '*' if lo > 0 or hi < 0 else ' '
        print(f'  {a:10s} - {c:6s} {x:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]{sig} tasks={n:3d} passes={len(shared)}  {pp}')
