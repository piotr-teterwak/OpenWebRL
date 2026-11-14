#!/usr/bin/env python3
"""Scan every episode of the paper-table systems (originals + redo2 reruns, same merge as paper_table.py) for signs
that the site blocked the browser (CAPTCHA / bot-wall / access-denied text anywhere in the record: page text, model
thoughts, judge explanation). Reports, per system x benchmark, valid episodes that look blocked, and the top sites.
    python3 tillicum/blocked_scan.py > tillicum/results/paper-table/blocked-scan.txt"""
import glob, json, re, sys, collections
from pathlib import Path
sys.argv = [sys.argv[0]]; sys.path.insert(0, str(Path(__file__).parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): import paper_table as P
PAT = re.compile(r'captcha|are you a robot|not a robot|unusual traffic|automated (?:access|behavio|traffic|queries)|access denied|'
                 r'press (?:&|and) hold|verify you are (?:a )?human|blocked due to|bot detection|request blocked|you have been blocked', re.I)
Z, E = P.Z, P.E
def files(name, b):  # -> list per run of {task: path} with the same override order as paper_table.merge
    out = []
    if name in ('SFT', 'ADM'):
        root, cond, runs = {'SFT': ('head-k5-20261101', 'sft1', ['run1', 'rerun2', 'rerun3']),
                            'ADM': ('controls-sameday-20261103', 'piotr5', ['run1', 'run2', 'run3'])}[name]
        bu = b.replace('-', '_')
        for r in runs:
            out.append([glob.glob(f'{Z}/{root}/{r}/{bu}/{cond}/{bu}/collection/records/*.json'),
                        glob.glob(f'{E}/passzx-{cond}-{r}-redo2/{cond}/{b}/results/*.json')])
    else:
        key = {'BoQ': 'boq', 'PRO-CUA': 'procua', 'Dreamer': 'webdreamer'}[name]
        for ps, dirs in P.BRUNS[key]:
            out.append([g for d in dirs for g in [glob.glob(f'{E}/{d}/{b}/results/*.json')]] + [glob.glob(f'{E}/pass{ps}-redo2/{P.LABEL[key]}/{b}/results/*.json')])
    return out
SITE = {}
for b, f in P.BENCH.items():
    for l in (P.REPO / f).read_text().splitlines():
        t = json.loads(l); u = str(t.get('start_url') or (t.get('metadata') or {}).get('start_url') or '')
        SITE[P.tid(t)] = u.split('/')[2] if '//' in u else P.tid(t).split('--')[0]
print('valid episodes whose record mentions a CAPTCHA / bot wall / access denial (merged state, all runs)')
for b in P.BENCH:
    print(f'\n{b}')
    for n in P.SYS:
        state = []
        for groups in files(n, b):
            m = {}
            for g in groups:
                for p in g:
                    txt = Path(p).read_text(errors='ignore'); r = json.loads(txt); v = str(r.get('valid')) == 'True'
                    ok = v and float(r.get('reward') or 0) >= 1; t = str(r['task_id'])
                    if v or t not in m: m[t] = (v, ok, bool(PAT.search(txt)))
            state.append(m)
        valid = [(t, x) for m in state for t, x in m.items() if x[0]]
        bl = [(t, x) for t, x in valid if x[2]]
        sites = collections.Counter(SITE.get(t, '?') for t, x in bl)
        print(f'  {n:8s} valid {len(valid):5d}  flagged {len(bl):4d} ({100*len(bl)/max(len(valid),1):4.1f}%)  of which success {sum(x[1] for t,x in bl):3d}  '
              f'top sites {sites.most_common(5)}')
