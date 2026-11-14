#!/usr/bin/env python3
"""Paper table (SFT, ADM, Best-of-Q, PRO-CUA, Dreamer-7B), 3 runs each, OM2W / DeepShop / WebVoyager.
Each run = original records with the redo2 reruns of invalid episodes merged (a rerun replaces the original only if it
is valid or the original is missing). Metrics:
  overall    successes / all tasks per run, averaged over runs (invalid = failure)
  overall-W  as overall, but each Wolfram Alpha task is scored over that system's valid runs only (the site blocked the
             cluster after 2026-11-03, so baselines' runs 2-3 have no valid Wolfram episodes)
  valid      pooled successes / valid episodes over the runs
    python3 tillicum/paper_table.py [--no-redo] [--no-wolfram] [--wolfram3]
"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from redo_baselines import RUNS as BRUNS, LABEL
Z = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'; E = '/gpfs/scrubbed/piotrt/owrl/eval'
REPO = Path(__file__).resolve().parents[1]
BENCH = {'webvoyager': 'openwebrl/data/eval/webvoyager_fara.jsonl', 'online-mind2web': 'openwebrl/data/eval/online-mind2web.jsonl',
         'deepshop': 'openwebrl/data/eval/deepshop.jsonl'}
REDO = '--no-redo' not in sys.argv
NOW = '--no-wolfram' in sys.argv  # drop Wolfram Alpha tasks from WebVoyager for every system
W3 = '--wolfram3' in sys.argv  # Wolfram tasks from the fresh same-window 3-run rerun (tillicum/wolfram_rerun.sh) for every system
def tid(r): return str(r.get('task_id') or (r.get('metadata') or {}).get('task_id') or r.get('id'))
TASKS = {b: [t for t in (tid(json.loads(l)) for l in (REPO / f).read_text().splitlines()) if not (NOW and t.startswith('Wolfram'))]
         for b, f in BENCH.items()}
def ours(d):  # -> {task: (valid, success)}
    out = {}
    for p in glob.glob(f'{d}/results/*.json'):
        r = json.loads(Path(p).read_text()); v = str(r.get('valid')) == 'True'
        out[str(r['task_id'])] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
def zx(root, run, cond, b):
    bu = b.replace('-', '_'); out = {}
    for p in glob.glob(f'{Z}/{root}/{run}/{bu}/{cond}/{bu}/collection/records/*.json'):
        r = json.loads(Path(p).read_text()); v = str(r.get('valid')) == 'True'
        out[str(r['task_id'])] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
def merge(*states):
    m = {}
    for s in states:
        for t, x in s.items():
            if x[0] or t not in m: m[t] = x
    return m
W3LABEL = {'SFT': 'sft1', 'ADM': 'piotr5', 'BoQ': 'base+boq', 'PRO-CUA': 'base+procuasel', 'Dreamer': 'base+webdreamer'}
def system_runs(name, b):
    runs = _system_runs(name, b)
    if W3 and b == 'webvoyager':
        for k, r in enumerate(runs, 1):
            for t in [t for t in r if t.startswith('Wolfram')]: del r[t]
            r.update(ours(f'{E}/passwolfram3-r{k}/{W3LABEL[name]}/webvoyager'))
    return runs
def _system_runs(name, b):
    if name in ('SFT', 'ADM'):
        root, cond, runs = {'SFT': ('head-k5-20261101', 'sft1', ['run1', 'rerun2', 'rerun3']),
                            'ADM': ('controls-sameday-20261103', 'piotr5', ['run1', 'run2', 'run3'])}[name]
        return [merge(zx(root, r, cond, b), ours(f'{E}/passzx-{cond}-{r}-redo2/{cond}/{b}') if REDO else {}) for r in runs]
    key = {'BoQ': 'boq', 'PRO-CUA': 'procua', 'Dreamer': 'webdreamer'}[name]
    return [merge(*[ours(f'{E}/{d}/{b}') for d in dirs], ours(f'{E}/pass{ps}-redo2/{LABEL[key]}/{b}') if REDO else {})
            for ps, dirs in BRUNS[key]]
def metrics(runs, b):
    T = TASKS[b]; N = len(T)
    overall = 100 * sum(sum(r.get(t, (0, 0))[1] for t in T) / N for r in runs) / len(runs)
    tot = 0
    for t in T:
        rs = [r[t] for r in runs if t in r and r[t][0]] if t.startswith('Wolfram') else [r.get(t, (0, 0)) for r in runs]
        tot += sum(x[1] for x in rs) / len(rs) if rs else 0
    v = sum(r.get(t, (0, 0))[0] for r in runs for t in T); s = sum(r.get(t, (0, 0))[1] for r in runs for t in T)
    return overall, 100 * tot / N, 100 * s / max(v, 1), 100 * v / (N * len(runs))
SYS = ['SFT', 'BoQ', 'PRO-CUA', 'Dreamer', 'ADM']
RUNS_ = {n: {b: system_runs(n, b) for b in BENCH} for n in SYS}
res = {n: {b: metrics(RUNS_[n][b], b) for b in BENCH} for n in SYS}
def common(b):  # tasks with >= 1 valid run for every system; each system scored per task over its valid runs
    T = [t for t in TASKS[b] if all(any(r.get(t, (0, 0))[0] for r in RUNS_[n][b]) for n in SYS)]
    out = {}
    for n in SYS:
        sc = [sum(r[t][1] for r in RUNS_[n][b] if t in r and r[t][0]) / sum(1 for r in RUNS_[n][b] if t in r and r[t][0]) for t in T]
        out[n] = 100 * sum(sc) / len(T)
    return len(T), out
COMMON = {b: common(b) for b in BENCH}
print(f"redo merged: {REDO}   Wolfram from fresh 3-run rerun: {W3}   Wolfram excluded: {NOW}   WebVoyager tasks: {len(TASKS['webvoyager'])}   columns: WebVoyager / OM2W / DeepShop / mean (paper order)")
for i, lab in enumerate(['overall', 'overall-W (Wolfram over valid runs)', 'valid-only', '% episodes valid']):
    print(f'\n{lab}')
    for n in SYS:
        x = [res[n][b][i] for b in BENCH]; print(f'  {n:8s}' + ''.join(f'{v:7.1f}' for v in x) + f'{sum(x)/3:7.1f}')
print(f"\ncommon tasks (valid for every system in >= 1 run; tasks per benchmark {[COMMON[b][0] for b in BENCH]})")
for n in SYS:
    x = [COMMON[b][1][n] for b in BENCH]; print(f'  {n:8s}' + ''.join(f'{v:7.1f}' for v in x) + f'{sum(x)/3:7.1f}')
