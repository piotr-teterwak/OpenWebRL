#!/usr/bin/env python3
"""Round 2 (Sol-steered, from C2) vs its data-matched control, C2 and base: passes s2-a + s2-b, OM2W + DeepShop, T 1.0.
Per benchmark and pooled: success over episodes, valid-only and overall (invalid = failure); paired differences on
(pass, task) units valid for both models, 10k-resample bootstrap 95% CI.
    python3 tillicum/round2_result.py      (PASSES=a,b MODELS=name=run,... PAIRS=x-y,... to override)"""
import glob, json, os, random
E = '/gpfs/scrubbed/piotrt/owrl/eval'; PASSES = os.environ.get('PASSES', 's2-a,s2-b').split(','); BENCH = ['online-mind2web', 'deepshop']
M = dict(x.split('=') for x in os.environ.get('MODELS', 'base=base,C2=r2-sol-r1-all-m0.3,r2=r2-sol2-r2,ctrl=r2-sol2-ctrl').split(','))
PAIRS = [tuple(x.split('-')) for x in os.environ.get('PAIRS', 'r2-C2,r2-ctrl,r2-base,ctrl-C2,ctrl-base,C2-base').split(',')]
def load(m, p, b):
    out = {}
    for f in glob.glob(f'{E}/pass{p}/{m}/{b}/results/*.json'):
        r = json.load(open(f)); v = str(r.get('valid')) == 'True'; out[str(r['task_id'])] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
D = {k: {(p, b): load(m, p, b) for p in PASSES for b in BENCH} for k, m in M.items()}
rng = random.Random(0)
def boot(d):
    if not d: return (float('nan'),) * 3
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    return 100 * sum(d) / len(d), 100 * bs[250], 100 * bs[9749]
print('episodes done per model/pass/benchmark:', {k: {f'{p}/{b}': len(x) for (p, b), x in v.items()} for k, v in D.items()})
for scope in BENCH + ['pooled']:
    keys = [(p, b) for p in PASSES for b in BENCH if scope in ('pooled', b)]
    print(f'\n{scope}')
    for k in M:
        eps = [x for pb in keys for x in D[k][pb].values()]; v = sum(x[0] for x in eps); s = sum(x[1] for x in eps)
        print(f'  {k:5s} episodes {len(eps):4d}  valid-only {100*s/max(v,1):5.1f}  overall {100*s/max(len(eps),1):5.1f}')
    for a, c in PAIRS:
        d = [int(D[a][pb][t][1]) - int(D[c][pb][t][1]) for pb in keys for t in D[a][pb] if t in D[c][pb] and D[a][pb][t][0] and D[c][pb][t][0]]
        x, lo, hi = boot(d); print(f'  {a:4s} - {c:4s}  {x:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  n={len(d)}')
