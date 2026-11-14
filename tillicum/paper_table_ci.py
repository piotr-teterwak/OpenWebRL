#!/usr/bin/env python3
"""Paired bootstrap CIs (over tasks, 10k resamples) for the paper table's overall metric: each task's score is its mean
success over a system's 3 runs (invalid = failure), Wolfram Alpha excluded. Rows: ADM minus each other system, and
each baseline minus SFT; per benchmark and for the mean of the 3 benchmarks.
    python3 tillicum/paper_table_ci.py"""
import sys, random
sys.argv.append('--no-wolfram')
sys.path.insert(0, 'tillicum')
import paper_table as P   # prints the table on import
B = list(P.BENCH); rng = random.Random(0)
def task_scores(n, b):
    return [sum(r.get(t, (0, 0))[1] for r in P.RUNS_[n][b]) / len(P.RUNS_[n][b]) for t in P.TASKS[b]]
S = {n: {b: task_scores(n, b) for b in B} for n in P.SYS}
def ci(a, c):
    d = {b: [x - y for x, y in zip(S[a][b], S[c][b])] for b in B}; out = {}
    for b in B:
        v = d[b]; N = len(v); boots = sorted(sum(v[rng.randrange(N)] for _ in range(N)) / N for _ in range(10000))
        out[b] = (100 * sum(v) / N, 100 * boots[250], 100 * boots[9749])
    idx = {b: range(len(d[b])) for b in B}; boots = []
    for _ in range(10000):
        boots.append(sum(sum(d[b][rng.randrange(len(d[b]))] for _ in idx[b]) / len(d[b]) for b in B) / 3)
    boots.sort(); out['mean'] = (sum(out[b][0] for b in B) / 3, 100 * boots[250], 100 * boots[9749])
    return out
print('\npaired differences, overall, Wolfram excluded: point [95% CI]   columns: WebVoyager / OM2W / DeepShop / mean')
for a, c in [('ADM', x) for x in ['SFT', 'BoQ', 'PRO-CUA', 'Dreamer']] + [(x, 'SFT') for x in ['BoQ', 'PRO-CUA', 'Dreamer']]:
    o = ci(a, c); print(f'  {a:7s} - {c:7s}' + ''.join(f'  {o[k][0]:+5.1f} [{o[k][1]:+5.1f},{o[k][2]:+5.1f}]' for k in B + ['mean']))
