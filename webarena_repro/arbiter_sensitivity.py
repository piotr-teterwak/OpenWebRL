#!/usr/bin/env python3
"""Does WebArbiter's verdict on our own tournament comparisons depend on the parts of the prompt WebPRMBench lacks?

    arbiter_sensitivity.py --url http://127.0.0.1:PORT/v1 --out DIR [--n-long 600] [--n-short 600] [--seed 0]

Samples logged step-1 WebArbiter reward calls (--n-long random calls over 8,192 prompt tokens and --n-short random
shorter ones; 18% of all calls are over), re-asks WebArbiter greedily under each condition, and reports how often the verdict differs from S0:
S0  as logged (re-query; measures greedy self-consistency under batching)
S1  without the '#### Prior Knowledge ####' and '#### State Difference Summary ####' sections (absent from WebPRMBench)
S2  S1 + AXTree cut so the prompt fits WebArbiter's 8,192-token training cap (tokens estimated from the logged count)
No labels exist for these comparisons, so this measures sensitivity, not accuracy.
"""
import argparse, asyncio, collections, glob, json, random, re
from pathlib import Path

import httpx

CELLS = '/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells'
ANSWER = re.compile(r'<Answer>\s*Response\s*([12])\s*</Answer>')
SECTION = re.compile(r'^#### (.+?) ####\s*$', re.M)


def sections(user):
    """[(name or None, text)] in order; the text before the first header has name None."""
    marks = list(SECTION.finditer(user))
    out = [(None, user[:marks[0].start()] if marks else user)]
    for m, nxt in zip(marks, marks[1:] + [None]):
        out.append((m.group(1), user[m.start():nxt.start() if nxt else len(user)]))
    return out


def strip_extra(user):
    return ''.join(t for n, t in sections(user) if n not in ('Prior Knowledge', 'State Difference Summary'))


def fit_axtree(system, user, logged_tokens, cap=8192, reserve=200):
    """Cut the AXTREE section (keeping its head) so the estimated prompt length is under the cap."""
    total_chars = len(system) + len(user)
    chars_per_token = total_chars / max(logged_tokens, 1)
    budget = int((cap - reserve) * chars_per_token)
    if total_chars <= budget:
        return user
    parts = sections(user); excess = total_chars - budget
    return ''.join(t[:max(200, len(t) - excess)] + '\n[... AXTree truncated ...]\n\n' if n == 'AXTREE' else t for n, t in parts)


def load(n_long, n_short, seed):
    rows = []
    for f in sorted(glob.glob(f'{CELLS}/*-arbiter-r[123]-*/task-*/calls.jsonl')):
        for line in open(f):
            if '"role": "reward"' not in line:
                continue
            r = json.loads(line); u = r['usage']; u = eval(u) if isinstance(u, str) else u
            m = ANSWER.search(r.get('reply') or '')
            if m:
                rows.append(dict(cell=f.split('/cells/')[1].rsplit('/', 1)[0], system=r['system'], user=r['user'],
                                 tokens=u['prompt_tokens'], logged=int(m.group(1))))
    long = [r for r in rows if r['tokens'] > 8192]
    short = [r for r in rows if r['tokens'] <= 8192]
    rng = random.Random(seed); rng.shuffle(long); rng.shuffle(short)
    return long[:n_long] + short[:n_short], len(rows)


async def ask(client, url, system, user, gate):
    async with gate:
        for attempt in range(3):
            try:
                resp = await client.post(f'{url}/chat/completions', json=dict(
                    model='WebArbiter', max_tokens=2048, temperature=0.0, top_p=1.0,
                    messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))
                d = resp.json(); text = d['choices'][0]['message']['content']
                m = ANSWER.search(text or '')
                return (int(m.group(1)) if m else 0), d.get('usage', {}).get('prompt_tokens')
            except Exception:
                await asyncio.sleep(5)
    return -1, None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-long', type=int, default=600)
    ap.add_argument('--n-short', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--concurrency', type=int, default=48)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sample, pool = load(a.n_long, a.n_short, a.seed)
    print(f'{len(sample)} sampled of {pool} parsed logged calls ({sum(r["tokens"] > 8192 for r in sample)} over 8,192 tokens)', flush=True)
    gate = asyncio.Semaphore(a.concurrency)
    variants = {'S0': lambda r: r['user'], 'S1': lambda r: strip_extra(r['user']),
                'S2': lambda r: fit_axtree(r['system'], strip_extra(r['user']), r['tokens'])}
    async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
        for name, fn in variants.items():
            res = await asyncio.gather(*(ask(client, a.url, r['system'], fn(r), gate) for r in sample))
            for r, (v, t) in zip(sample, res):
                r[name], r[name + '_tokens'] = v, t
    with (a.out / 'verdicts.jsonl').open('w') as f:
        for r in sample:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ('system', 'user')}) + '\n')
    summary = {}
    for bucket, sel in (('all', sample), ('over_8k', [r for r in sample if r['tokens'] > 8192]),
                        ('under_8k', [r for r in sample if r['tokens'] <= 8192])):
        n = len(sel); s = summary[bucket] = dict(n=n)
        s['S0_vs_logged_same'] = 100 * sum(r['S0'] == r['logged'] for r in sel) / n if n else None
        for c in ('S1', 'S2'):
            ok = [r for r in sel if r['S0'] in (1, 2) and r[c] in (1, 2)]
            s[f'{c}_flip_vs_S0'] = 100 * sum(r[c] != r['S0'] for r in ok) / len(ok) if ok else None
        s['response2_rate'] = {c: 100 * sum(r[c] == 2 for r in sel) / n if n else None for c in ('logged', 'S0', 'S1', 'S2')}
        s['unparsed'] = {c: sum(r[c] in (0, -1) for r in sel) for c in ('S0', 'S1', 'S2')}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
