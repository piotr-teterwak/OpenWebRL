#!/usr/bin/env python3
"""Per-site status of one ablation row vs the step-1 rows on the same tasks (paired bootstrap, 2,000 resamples).

    wa_site_status.py A18-r1 [arbiter]
"""
import collections, random, statistics, sys
import wa_paired

TOTAL = {'gitlab': 30, 'shopping_admin': 35, 'reddit': 19, 'shopping': 45}
PAPER = {'gitlab': (56.67, 40.00), 'shopping_admin': (42.86, 37.14), 'reddit': (52.63, 42.11), 'shopping': (44.44, 26.67)}


def main():
    tag = sys.argv[1]; method = sys.argv[2] if len(sys.argv) > 2 else 'arbiter'
    cells = wa_paired.load(wa_paired.Path('/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells'), {tag, 'r1', 'r2', 'r3'})
    def per(m, tags):
        d = collections.defaultdict(list)
        for (site, mm, t, task), s in cells.items():
            if mm == m and t in tags and s['valid']:
                d[(site, task)].append(float(s['success']))
        return {k: statistics.mean(v) for k, v in d.items()}
    a = per(method, {tag}); refs = {'step-1 WebArbiter': per('arbiter', {'r1', 'r2', 'r3'}), 'step-1 mini-PRM': per('mini_prm', {'r1', 'r2', 'r3'})}
    for site, name in [('gitlab', 'GitLab'), ('shopping_admin', 'CMS'), ('reddit', 'Reddit'), ('shopping', 'Shop'), (None, 'All')]:
        keys = sorted(k for k in a if site is None or k[0] == site)
        if not keys:
            continue
        parts = []
        for rname, y in refs.items():
            ks = [k for k in keys if k in y]
            if not ks:
                parts.append(f'{rname} n/a'); continue
            rng = random.Random(0); ds = []
            for _ in range(2000):
                idx = [rng.choice(ks) for _ in ks]; ds.append(100 * sum(a[k] - y[k] for k in idx) / len(idx))
            ds.sort(); ym = 100 * statistics.mean(y[k] for k in ks); x = 100 * statistics.mean(a[k] for k in ks)
            parts.append(f'{rname} {ym:.1f}% ({x - ym:+.1f} [{ds[50]:+.0f}, {ds[1949]:+.0f}])')
        tot = TOTAL.get(site, sum(TOTAL.values())); pap = f' | paper {PAPER[site][0]} / {PAPER[site][1]}' if site else ''
        print(f'{name:6} {len(keys)}/{tot} cells: {tag} {sum(a[k] for k in keys):.0f}/{len(keys)} = {100 * statistics.mean(a[k] for k in keys):.1f}% | ' + ' | '.join(parts) + pap)


if __name__ == '__main__':
    main()
