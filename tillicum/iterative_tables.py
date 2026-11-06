#!/usr/bin/env python3
"""Tables for tillicum/ITERATIVE_DPO.md, recomputed from the raw eval results (stdlib only).

    python3 tillicum/iterative_tables.py

Every round-2 variant was evaluated on Online-Mind2Web in the same pass (same job, same day) as the round-1
reference model onpol-dpo-b0.3-s1, so each variant is compared with the round-1 run of its own pass.
overall = successes / 300 (invalid = failure); valid-only = successes / valid; cut-off = episodes ended by the
per-generation token cap (terminate_reason generation_length_limit). Paired differences: tasks valid in both,
95% percentile interval over 4,000 task resamples (seed 0).
"""
import glob
import json
import os
import random

EVAL = '/gpfs/scrubbed/piotrt/owrl/eval/'
R1 = 'onpol-dpo-b0.3-s1'
VARIANTS = [  # (label, [(pass, arm), ...]) one entry per seed
    ('round 2, DPO, reference = round 1', [('passround2', f'r2-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('round 2, RPO (DPO + SFT weight 1), reference = round 1', [('passround2b', f'r2-rpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('round 1 + round 2 pairs, one DPO pass from base', [('passround2b', f'comb-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('round 2, reference = base', [(f'passr2ref{s}', f'r2-refbase-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('round 2, Luna relabels the chosen, reference = round 1', [(f'passr2luna{s}', f'r2-luna-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('round 2, only pairs Luna agrees with, reference = base', [(f'passr2agree{s}', f'r2-agree-refbase-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('new task pool, margin 0.3, reference = base', [(f'passr2np{s}', f'r2-newpool-m0.3-refbase-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('new pool, Luna labels all states, margin 0.3, ref = base', [(f'passr2nplall{s}', f'r2-np-luna-all-refbase-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
    ('new pool, Luna = executed pick only, margin 0.3, ref = base', [(f'passr2nplexec{s}', f'r2-np-luna-exec-refbase-dpo-b0.3-s{s}') for s in (1, 2, 42)]),
]


def load(path):
    out = {}
    for f in glob.glob(path + '/results/*.json'):
        r = json.load(open(f)); ok = bool(r.get('valid')) and r.get('reward') is not None
        out[r['task_id']] = (ok, ok and float(r['reward']) >= 0.5, r.get('terminate_reason') == 'generation_length_limit')
    return out


def stats(res, n=300):
    v = [x for x in res.values() if x[0]]
    return dict(n=len(res), valid=len(v), vo=100 * sum(x[1] for x in v) / max(1, len(v)), ov=100 * sum(x[1] for x in v) / n,
                cut=100 * sum(x[2] for x in res.values()) / max(1, len(res)))


def paired(a, b):
    k = [t for t in a if t in b and a[t][0] and b[t][0]]
    d = [int(a[t][1]) - int(b[t][1]) for t in k]
    rng = random.Random(0)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(4000)) if d else [0]
    return 100 * sum(d) / max(1, len(d)), 100 * bs[100], 100 * bs[3899], len(d)


def main():
    print('## Round-2 variants vs the round-1 model in the same pass (Online-Mind2Web, overall %)')
    print('| variant | seed | variant overall | round 1 overall | diff (overall) | paired valid-only diff [95% CI] | cut-off variant / round 1 |')
    for label, runs in VARIANTS:
        diffs, cuts = [], []
        for (p, arm), seed in zip(runs, (1, 2, 42)):
            a, b = load(f'{EVAL}{p}/{arm}/online-mind2web'), load(f'{EVAL}{p}/{R1}/online-mind2web')
            if not a:
                print(f'| {label} | {seed} | missing |'); continue
            sa, sb = stats(a), stats(b)
            if sa['valid'] < 150 or sb['valid'] < 150:
                print(f'| {label} | {seed} | invalid pass ({sa["valid"]} / {sb["valid"]} valid) - excluded |'); continue
            pd = paired(a, b); diffs.append(sa['ov'] - sb['ov']); cuts.append(sa['cut'])
            print(f'| {label} | {seed} | {sa["ov"]:.1f} | {sb["ov"]:.1f} | {sa["ov"] - sb["ov"]:+.1f} | '
                  f'{pd[0]:+.1f} [{pd[1]:+.1f}, {pd[2]:+.1f}] n={pd[3]} | {sa["cut"]:.0f}% / {sb["cut"]:.0f}% |')
        if diffs:
            print(f'|   mean over {len(diffs)} seeds | | | | **{sum(diffs) / len(diffs):+.1f}** | | {sum(cuts) / len(cuts):.0f}% |')
    print('\n## Base actor and round 1 in the round-2 passes (Online-Mind2Web)')
    for p in ('passround2', 'passround2b'):
        b, r = stats(load(f'{EVAL}{p}/base/online-mind2web')), stats(load(f'{EVAL}{p}/{R1}/online-mind2web'))
        print(f'| {p} | base overall {b["ov"]:.1f} (cut-off {b["cut"]:.0f}%) | round 1 overall {r["ov"]:.1f} (cut-off {r["cut"]:.0f}%) | diff {r["ov"] - b["ov"]:+.1f} |')
    r1 = [stats(load(f'{EVAL}{p}/{R1}/online-mind2web'))['ov'] for p in sorted({p for _, runs in VARIANTS for p, _ in runs}) + ['passround2', 'passround2b']
          if stats(load(f'{EVAL}{p}/{R1}/online-mind2web'))['valid'] >= 150]
    print(f'round-1 model across {len(r1)} passes: overall {min(r1):.1f} to {max(r1):.1f}, mean {sum(r1) / len(r1):.1f}')
    print('\n## Clean WebVoyager re-run (base vs round-1 beta 0.3, same job, 595 tasks)')
    for s in (1, 2, 42):
        p = f'{EVAL}passwvclean{s}'
        if not os.path.isdir(p):
            continue
        b, a = load(f'{p}/base/webvoyager'), load(f'{p}/onpol-dpo-b0.3-s{s}/webvoyager')
        if a and b:
            sb, sa, pd = stats(b, 595), stats(a, 595), paired(a, b)
            print(f'| seed {s} | base valid-only {sb["vo"]:.1f} overall {sb["ov"]:.1f} | round 1 valid-only {sa["vo"]:.1f} overall {sa["ov"]:.1f} | '
                  f'paired {pd[0]:+.1f} [{pd[1]:+.1f}, {pd[2]:+.1f}] n={pd[3]} |')


if __name__ == '__main__':
    main()
