#!/usr/bin/env python3
"""Bot-wall scan for the paper-table systems: valid episodes in which the agent ran into a CAPTCHA / bot wall / access
denial. Episodes that hit one count as *valid* failures, so the invalid-episode reruns never touch them; if different
systems hit them at different rates (e.g. evaluated on different days), the overall comparison is biased.

Searched: what the agent saw and said: page observations and tool responses (non-system messages), its thoughts,
and its final answer. Not searched: the prompt / tool definitions (the `done` tool's description itself mentions
CAPTCHAs, which made the first version of this scan flag every episode). Same rule for both harnesses: Zixian's
records point at the attempt's result.json, which has our layout. Merge order matches paper_table.py (original
runs, then redo2 reruns, then the Wolfram rerun with --wolfram3 semantics: Wolfram tasks reported separately).
    python3 tillicum/blocked_scan.py > tillicum/results/paper-table/blocked-scan.txt"""
import glob, json, re, sys, collections
from pathlib import Path
sys.argv = [sys.argv[0]]; sys.path.insert(0, str(Path(__file__).parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): import paper_table as P
PAT = re.compile(r'captcha|are you a robot|not a robot|unusual traffic|automated (?:access|behavio|traffic|queries)|access denied|'
                 r'press (?:&|and) hold|verify you are (?:a )?human|blocked due to|bot detection|request blocked|you have been blocked|'
                 r'anti-bot|security check|cloudflare', re.I)
def seen_text(res):  # result.json -> text the agent saw or wrote
    m = res.get('metadata') or {}
    parts = [str(res.get('response') or ''), json.dumps(m.get('step_tool_responses') or [])]
    for msg in m.get('messages') or []:
        if msg.get('role') != 'system': parts.append(json.dumps(msg.get('content')))
    return '\n'.join(parts)
def hits(txt): return sorted({x.group(0).lower() for x in PAT.finditer(txt)})
IMG = re.compile(r'"(?:data:image/[^"]{100,}|[A-Za-z0-9+/=]{2000,})"')  # inline screenshots: most of each file
def load(p): return json.loads(IMG.sub('""', Path(p).read_text(errors='ignore')))
def one(arg):  # (path, kind) -> (task, valid, success, terms); kind 'zx' = Zixian record (valid/reward there, text in artifact)
    p, kind = arg
    try:
        rec = load(p)
    except Exception:
        return None
    v = str(rec.get('valid')) == 'True'; ok = v and float(rec.get('reward') or 0) >= 1
    if kind == 'zx':
        try: res = load(rec['artifact'])
        except Exception: res = {}
    else:
        res = rec
    return str(rec['task_id']), v, ok, hits(seen_text(res))
POOL = None
def scan(paths, kind):
    return [x for x in POOL.map(one, [(p, kind) for p in paths], chunksize=4) if x]
def episodes(name, b):  # -> per run {task: (valid, success, matched terms)}
    runs = []
    if name in ('SFT', 'ADM'):
        root, cond, rr = {'SFT': ('head-k5-20261101', 'sft1', ['run1', 'rerun2', 'rerun3']),
                          'ADM': ('controls-sameday-20261103', 'piotr5', ['run1', 'run2', 'run3'])}[name]
        bu = b.replace('-', '_')
        for r in rr:
            m = {}
            for t, v, ok, h in scan(glob.glob(f'{P.Z}/{root}/{r}/{bu}/{cond}/{bu}/collection/records/*.json'), 'zx'):
                m[t] = (v, ok, h)
            for t, v, ok, h in scan(glob.glob(f'{P.E}/passzx-{cond}-{r}-redo2/{cond}/{b}/results/*.json'), 'ours'):
                if v or t not in m: m[t] = (v, ok, h)
            runs.append(m)
    else:
        key = {'BoQ': 'boq', 'PRO-CUA': 'procua', 'Dreamer': 'webdreamer', 'WebArbiter': 'webarbiter'}[name]
        for ps, dirs in P.BRUNS[key]:
            m = {}
            for d in dirs + [f'pass{ps}-redo2/{P.LABEL[key]}']:
                for t, v, ok, h in scan(glob.glob(f'{P.E}/{d}/{b}/results/*.json'), 'ours'):
                    if v or t not in m: m[t] = (v, ok, h)
            runs.append(m)
    return runs
SITE = {}
for b, f in P.BENCH.items():
    for l in (P.REPO / f).read_text().splitlines():
        t = json.loads(l); u = str(t.get('start_url') or (t.get('metadata') or {}).get('start_url') or '')
        SITE[P.tid(t)] = u.split('/')[2] if '//' in u else P.tid(t).split('--')[0]
import multiprocessing as mp, os
POOL = mp.Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
print(__doc__.split('\n')[0]); print('(Wolfram Alpha excluded: rerun 2026-11-13, 0/230 valid, counted as failed for every system)')
ALL = {}
for b in P.BENCH:
    print(f'\n{b}')
    for n in P.SYS:
        runs = episodes(n, b); ALL[(n, b)] = runs
        valid = [(t, x) for m in runs for t, x in m.items() if x[0] and not t.startswith('Wolfram')]
        bl = [(t, x) for t, x in valid if x[2]]
        sites = collections.Counter(SITE.get(t, '?') for t, x in bl); terms = collections.Counter(w for t, x in bl for w in x[2])
        per_run = [sum(1 for t, x in m.items() if x[0] and x[2] and not t.startswith('Wolfram')) for m in runs]
        print(f'  {n:8s} valid {len(valid):5d}  flagged {len(bl):4d} ({100*len(bl)/max(len(valid),1):4.1f}%)  per run {per_run}  '
              f'flagged successes {sum(x[1] for t,x in bl):3d}', flush=True)
        print(f'           sites {sites.most_common(6)}')
        print(f'           terms {terms.most_common(5)}', flush=True)
json.dump({f'{n}|{b}': [{t: x for t, x in m.items()} for m in runs] for (n, b), runs in ALL.items()},
          open(Path(__file__).parent / 'results/paper-table/blocked-scan.json', 'w'))
