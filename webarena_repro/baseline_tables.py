#!/usr/bin/env python3
"""Recompute every baseline table in docs/BASELINES_2026-11-06.md from the raw per-task results (stdlib only).

    python3 webarena_repro/baseline_tables.py            # all sections
    python3 webarena_repro/baseline_tables.py wa om2w    # some of: wa om2w ext dreamer

wa      WebArena-Lite (129 tasks, four sites, no Map): GPT-4o actor, A18 harness, per-site success (valid-only), site
        macro-mean, and paired differences with a site-stratified task bootstrap
om2w    Online-Mind2Web (300 live tasks): GPT-4o actor, A18 harness at 1280x720
ext     external selectors on our OpenWebRL-4B-SFT actor (DeepShop / OM2W / WebVoyager): Best-of-Q, PRO-CUA PRM,
        WebDreamer, against our base actor and Zixian's same-window SelectionARM ("our ADM") runs
Conventions: valid-only = successes / valid episodes; overall = successes / all tasks (invalid = failure); paired
differences use tasks valid in both rows (or all tasks for "overall"); 95% percentile intervals, 4,000 resamples, seed 0.
"""
import glob
import json
import random
import sys

CELLS = '/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells/'
OM2W = '/gpfs/scrubbed/piotrt/owrl/webarena/runs/om2w/'
EVAL = '/gpfs/scrubbed/piotrt/owrl/eval/'
BASE_T1 = '/gpfs/scrubbed/piotrt/owrl/base_t1/runs/base-t1-{}/results/*.json'
ZIXIAN = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/controls-sameday-20261103/run{}/{}/{}/{}/collection/records/*.json'
SITES = ('shopping', 'cms', 'reddit', 'gitlab')
BENCH = (('online-mind2web', 'online_mind2web', 300), ('webvoyager', 'webvoyager', 595), ('deepshop', 'deepshop', 150))


def boot(d, strata=None, n=4000):
    """Mean of d (or of per-stratum means) and its 95% percentile interval, in points."""
    rng = random.Random(0)
    if strata is None:
        bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
        return 100 * sum(d) / len(d), 100 * bs[n // 40], 100 * bs[n - n // 40 - 1]
    groups = [v for v in strata.values() if v]
    mean = lambda gs: sum(sum(g) / len(g) for g in gs) / len(gs)
    bs = sorted(mean([rng.choices(g, k=len(g)) for g in groups]) for _ in range(n))
    return 100 * mean(groups), 100 * bs[n // 40], 100 * bs[n - n // 40 - 1]


def fmt(t):
    return f'{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]'


# ---------------------------------------------------------------- WebArena-Lite
def wa_load(site, tag):
    out = {}
    for p in glob.glob(f'{CELLS}{site}-{tag}-*/task-*/summary.json'):
        s = json.load(open(p))
        if s.get('valid'):
            out[s['task_id']] = float(s.get('score') or 0) >= 0.5
    return out


WA_ROWS = [('GPT-4o, no selection, 1280x720', 'no_search-N18v720-r1'),
           ('+ WebArbiter-7B, 1280x720', 'arbiter-A18v720-r1'),
           ('+ our ADM, 1280x720', 'selectionarm-S18-r1'),
           ('+ WebArbiter-7B, 1024x2490 (paper viewport)', 'arbiter-A18-r1'),
           ('+ our ADM, 1024x2490, 1024x576 best-coverage crop', 'selectionarm-S18c1024-r1'),
           ('+ our ADM, 1024x2490, full tall screenshot (GitLab+CMS)', 'selectionarm-S18tall-r1'),
           ('+ our ADM, 1280x2490, centred 720 crop (GitLab+CMS)', 'selectionarm-S18crop-r1')]


def section_wa():
    R = {name: {s: wa_load(s, tag) for s in SITES} for name, tag in WA_ROWS}
    print('## WebArena-Lite (GPT-4o actor; valid-only success %, n valid; site macro-mean over sites present)')
    for name, _ in WA_ROWS:
        r = R[name]; have = [s for s in SITES if r[s]]
        rates = {s: 100 * sum(r[s].values()) / len(r[s]) for s in have}
        cells = ' | '.join(f'{rates[s]:.1f} ({len(r[s])})' if s in rates else '-' for s in SITES)
        print(f'| {name} | {cells} | {sum(rates.values()) / len(rates):.1f} |')
    print('paired (site-stratified):')
    pairs = [('+ our ADM, 1280x720', '+ WebArbiter-7B, 1280x720'), ('+ WebArbiter-7B, 1280x720', 'GPT-4o, no selection, 1280x720'),
             ('+ our ADM, 1280x720', 'GPT-4o, no selection, 1280x720'),
             ('+ our ADM, 1024x2490, 1024x576 best-coverage crop', '+ WebArbiter-7B, 1024x2490 (paper viewport)'),
             ('+ our ADM, 1024x2490, 1024x576 best-coverage crop', '+ our ADM, 1280x720'),
             ('+ WebArbiter-7B, 1280x720', '+ WebArbiter-7B, 1024x2490 (paper viewport)')]
    for a, b in pairs:
        st = {s: [int(R[a][s][t]) - int(R[b][s][t]) for t in set(R[a][s]) & set(R[b][s])] for s in SITES}
        st = {s: v for s, v in st.items() if v}
        print(f'  {a}  minus  {b}: {fmt(boot(None, st))}  n={sum(map(len, st.values()))} sites={list(st)}')


# ---------------------------------------------------------------- Online-Mind2Web, GPT-4o
OM2W_ROWS = {'GPT-4o, no selection': ['no_search-N18v720-4215*', 'no_search-N18v720-redo*', 'no_search-N18v720-full-*'],
             '+ WebArbiter-7B': ['arbiter-A18v720-4215*', 'arbiter-A18v720-full-*'],
             '+ our ADM': ['selectionarm-S18-4215*', 'selectionarm-S18-full-*']}


def section_om2w():
    R = {}
    print('## Online-Mind2Web, GPT-4o actor, A18 harness, 1280x720 (300 tasks)')
    for name, pats in OM2W_ROWS.items():
        by = {}
        for pat in pats:
            for p in sorted(glob.glob(OM2W + pat + '/task-*/summary.json')):
                s = json.load(open(p))
                if s['index'] not in by or s['valid']:
                    by[s['index']] = s
        R[name] = by
        v = [s for s in by.values() if s['valid']]; k = sum(s['success'] or 0 for s in v)
        usd = sum((s.get('usd') or {}).get('policy', 0) for s in by.values())
        print(f'| {name} | {len(by)}/300 | {len(v)} | {100 * k / max(1, len(v)):.1f} | {100 * k / 300:.1f} | '
              f'{sum(s["steps"] for s in by.values()) / max(1, len(by)):.1f} | ${usd:.0f} |')
    V = {n: {i: s['success'] for i, s in by.items() if s['valid']} for n, by in R.items()}
    for a, b in (('+ our ADM', '+ WebArbiter-7B'), ('+ our ADM', 'GPT-4o, no selection'), ('+ WebArbiter-7B', 'GPT-4o, no selection')):
        ks = sorted(set(V[a]) & set(V[b])); d = [V[a][i] - V[b][i] for i in ks]
        print(f'  {a} minus {b}: {fmt(boot(d))} n={len(ks)} (only {a} {sum(x > 0 for x in d)}, only {b} {sum(x < 0 for x in d)})')


# ---------------------------------------------------------------- external selectors on our SFT actor
def ext_load(pats):
    out = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            r = json.load(open(f)); ok = bool(r.get('valid')) and r.get('reward') is not None
            if r['task_id'] not in out or ok:
                out[r['task_id']] = (ok, ok and float(r['reward']) >= 0.5)
    return out


def zixian(zb, slug):
    runs = []
    for k in (1, 2, 3):
        d = {}
        for f in glob.glob(ZIXIAN.format(k, zb, slug, zb)):
            x = json.load(open(f)); d[x['task_id']] = (bool(x['valid']), bool(x['valid']) and (x['reward'] or 0) >= 1)
        runs.append(d)
    return runs


EXT = {'Best-of-Q (T 1.0)': ['passboqbranchsim1/base+boq'], 'PRO-CUA PRM (T 1.0)': ['passprocua-t1/base+procuasel'],
       'WebDreamer, n=1 value sample': ['passwebdreamer-n1/base+webdreamer*', 'passwebdreamer-n1-redo/base+webdreamer*'],
       'WebDreamer, n=10 value samples': ['passwebdreamer2/base+webdreamer*', 'passwebdreamer2-redo/base+webdreamer*']}


def section_ext():
    print('## External selectors on OpenWebRL-4B-SFT, T 1.0, 5 candidates (valid-only % / overall %)')
    mean_v, mean_o = {}, {}
    for b, zb, n in BENCH:
        base = ext_load([BASE_T1.format(b)])
        adm_runs = zixian(zb, 'piotr5'); ids = set().union(*adm_runs)
        adm_v = {t: (lambda v: sum(v) / len(v) if v else None)([r[t][1] for r in adm_runs if t in r and r[t][0]]) for t in ids}
        adm_o = {t: sum(r.get(t, (0, 0))[1] for r in adm_runs) / 3 for t in ids}
        rows = {'our base actor, 1 sample (same days)': base}
        rows.update({name: ext_load([f'{EVAL}{p}/{b}/results/*.json' for p in pats]) for name, pats in EXT.items()})
        print(f'### {b}')
        for name, w in rows.items():
            v = [x[1] for x in w.values() if x[0]]
            kb = [t for t in w if w[t][0] and t in base and base[t][0]]
            ka = [t for t in w if w[t][0] and adm_v.get(t) is not None]; ko = [t for t in ids if t in w]
            vo, ov = 100 * sum(v) / max(1, len(v)), 100 * sum(v) / n
            mean_v.setdefault(name, []).append(vo); mean_o.setdefault(name, []).append(ov)
            line = f'| {name} | {len(w)}/{n} | {len(v)} | {vo:.1f} | {ov:.1f} |'
            if name != 'our base actor, 1 sample (same days)':
                line += f' vs base {fmt(boot([int(w[t][1]) - int(base[t][1]) for t in kb]))} |'
                line += f' vs ADM valid {fmt(boot([w[t][1] - adm_v[t] for t in ka]))} | vs ADM overall {fmt(boot([int(w[t][1]) - adm_o[t] for t in ko]))} |'
            print(line)
        av = [adm_v[t] for t in ids if adm_v[t] is not None]
        mean_v.setdefault('our ADM (Zixian, 3 runs, Nov 3-4)', []).append(100 * sum(av) / len(av))
        mean_o.setdefault('our ADM (Zixian, 3 runs, Nov 3-4)', []).append(100 * sum(adm_o.values()) / n)
        print(f'| our ADM (Zixian, 3 runs, Nov 3-4) | {n} | - | {mean_v["our ADM (Zixian, 3 runs, Nov 3-4)"][-1]:.1f} | '
              f'{mean_o["our ADM (Zixian, 3 runs, Nov 3-4)"][-1]:.1f} |')
    print('### equal-weight means over OM2W, WebVoyager, DeepShop (valid-only / overall)')
    for k in mean_v:
        print(f'| {k} | {sum(mean_v[k]) / 3:.1f} | {sum(mean_o[k]) / 3:.1f} |')


if __name__ == '__main__':
    want = set(sys.argv[1:]) or {'wa', 'om2w', 'ext'}
    if 'wa' in want: section_wa()
    if 'om2w' in want: section_om2w()
    if 'ext' in want or 'dreamer' in want: section_ext()
