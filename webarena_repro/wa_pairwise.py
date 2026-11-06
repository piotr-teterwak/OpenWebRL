#!/usr/bin/env python3
"""Pairwise comparison of two rows (method + tags) on the tasks both have valid cells for.

    wa_pairwise.py arbiter:r1,r2,r3 mini_prm:r1,r2,r3 [more pairs as "A B" ...]

Per task the row's success is the mean over its valid cells (repeats). Reports both rates on the shared tasks, the
paired difference with a bootstrap 95% CI over tasks (2,000 resamples, seed 0), task-level wins / ties / losses, and
the per-site difference. Cells are loaded as in wa_report.py (latest valid cell per site / method / tag / task).
"""
import collections, random, statistics, sys
from pathlib import Path

import wa_paired

RUNS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells')


def parse(spec):
    method, tags = spec.split(':'); return method, tags.split(',')


def task_means(cells, method, tags):
    per = collections.defaultdict(list)
    for (site, m, t, task), s in cells.items():
        if m == method and t in tags and s['valid']:
            per[(site, task)].append(float(s['success']))
    return {k: statistics.mean(v) for k, v in per.items()}


def main():
    specs = sys.argv[1:]
    if len(specs) < 2 or len(specs) % 2:
        sys.exit(__doc__)
    pairs = [(parse(specs[i]), parse(specs[i + 1])) for i in range(0, len(specs), 2)]
    cells = wa_paired.load(RUNS, {t for (a, b) in pairs for _, tags in (a, b) for t in tags})
    print(f"{'A':22}{'B':22}{'tasks':>6}{'A':>8}{'B':>8}{'A-B':>7}{'95% CI':>13}{'win/tie/loss':>14}  per site A-B (n)")
    for (ma, ta), (mb, tb) in pairs:
        A, B = task_means(cells, ma, ta), task_means(cells, mb, tb)
        keys = sorted(set(A) & set(B))
        if not keys:
            print(f'{ma}:{",".join(ta)} vs {mb}:{",".join(tb)}: no shared tasks'); continue
        x, y = [A[k] for k in keys], [B[k] for k in keys]
        rng = random.Random(0); diffs = []
        for _ in range(2000):
            idx = [rng.randrange(len(keys)) for _ in keys]
            diffs.append(100 * (sum(x[i] for i in idx) - sum(y[i] for i in idx)) / len(idx))
        diffs.sort()
        wtl = (sum(a > b for a, b in zip(x, y)), sum(a == b for a, b in zip(x, y)), sum(a < b for a, b in zip(x, y)))
        sites = '  '.join(f"{site} {100 * (statistics.mean([A[k] for k in keys if k[0] == site]) - statistics.mean([B[k] for k in keys if k[0] == site])):+.1f} ({sum(k[0] == site for k in keys)})"
                          for site in sorted({k[0] for k in keys}))
        ci = '[%+.0f, %+.0f]' % (diffs[50], diffs[1949])
        print(f"{ma + ':' + ','.join(ta):22}{mb + ':' + ','.join(tb):22}{len(keys):>6}{100 * statistics.mean(x):>7.1f}%{100 * statistics.mean(y):>7.1f}%"
              f"{100 * (statistics.mean(x) - statistics.mean(y)):>+7.1f}{ci:>13}{'/'.join(map(str, wtl)):>14}  {sites}")


if __name__ == '__main__':
    main()
