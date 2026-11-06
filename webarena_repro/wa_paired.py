#!/usr/bin/env python3
"""Paired comparison of ablation cells with the step-1 repeats on the same tasks.

    wa_paired.py --tags A15-r1 A16-r1 A12full-r1 A13-r1 [--base r1 r2 r3] [--methods arbiter mini_prm]

For each (method, ablation tag): the ablation's success over its valid cells, the mean success of the valid base-repeat
cells on exactly those tasks, the difference, a paired bootstrap 95% CI over tasks (2,000 resamples, seed 0), the number
of ablation episodes that hit the 30-step N/A fallback, and the per-site split. Cells are loaded as in wa_report.py
(latest valid cell per site / method / tag / task; unclosed-fence cells invalid).
"""
import argparse, collections, json, random, statistics
from pathlib import Path

import wa_report

MAX_STEP = 'max_steps_reached_na_fallback'


def load(runs, tags):
    cells = {}
    for p in sorted(runs.glob('*/task-*/summary.json')):
        parts = p.parent.parent.name.split('-'); job = parts[-1]; tag = '-'.join(parts[2:-1])
        if tag not in tags:
            continue
        s = json.loads(p.read_text()); s['job'] = job; s['tag'] = tag
        if s['valid'] and wa_report.unclosed_fence(p.parent):
            s.update(valid=False, invalid_reason='unclosed_fence')
        key = (s['site'], s['method'], tag, s['task_id'])
        if key not in cells or (s['valid'] and not cells[key]['valid']) or (s['valid'] == cells[key]['valid'] and job > cells[key]['job']):
            cells[key] = s
    return cells


def compare(abl, base):
    """abl: {(site, task): cell}; base: {(site, task): [success, ...]} -> stats over the shared tasks."""
    keys = sorted(k for k in abl if base.get(k))
    if not keys:
        return None
    x = [float(abl[k]['success']) for k in keys]; y = [statistics.mean(base[k]) for k in keys]
    rng = random.Random(0); diffs = []
    for _ in range(2000):
        idx = [rng.randrange(len(keys)) for _ in keys]
        diffs.append(100 * (sum(x[i] for i in idx) - sum(y[i] for i in idx)) / len(idx))
    diffs.sort()
    return dict(n=len(keys), abl=100 * statistics.mean(x), base=100 * statistics.mean(y), diff=100 * (statistics.mean(x) - statistics.mean(y)),
                lo=diffs[50], hi=diffs[1949], max_step=sum(abl[k]['status'] == MAX_STEP or abl[k]['steps'] >= 30 for k in keys),
                sites={site: (sum(x[i] for i, k in enumerate(keys) if k[0] == site), sum(y[i] for i, k in enumerate(keys) if k[0] == site),
                              sum(k[0] == site for k in keys)) for site in sorted({k[0] for k in keys})})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=Path('/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells'))
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--base', nargs='+', default=['r1', 'r2', 'r3'])
    ap.add_argument('--methods', nargs='+', default=['arbiter', 'mini_prm'])
    a = ap.parse_args()
    cells = load(a.runs, set(a.tags) | set(a.base))
    print(f"{'ablation / method':28}{'cells':>6}{'ablation':>10}{'step 1 same':>12}{'diff':>7}{'95% CI':>13}{'max-step':>10}  per site (ablation succ / step-1 mean succ / n)")
    for tag in a.tags:
        for method in a.methods:
            abl = {(s['site'], s['task_id']): s for (site, m, t, task), s in cells.items() if t == tag and m == method and s['valid']}
            base = collections.defaultdict(list)
            for (site, m, t, task), s in cells.items():
                if t in a.base and m == method and s['valid']:
                    base[(site, task)].append(float(s['success']))
            r = compare(abl, base)
            if r is None:
                continue
            sites = '  '.join(f"{site} {x:.0f}/{y:.1f}/{n}" for site, (x, y, n) in r['sites'].items())
            ci = '[%+.0f, %+.0f]' % (r['lo'], r['hi']); ms = '%d/%d' % (r['max_step'], r['n'])
            print(f"{tag + ' / ' + method:28}{r['n']:>6}{r['abl']:>9.1f}%{r['base']:>11.1f}%{r['diff']:>+7.1f}{ci:>13}{ms:>10}  {sites}")


if __name__ == '__main__':
    main()
