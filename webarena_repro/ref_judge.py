#!/usr/bin/env python3
"""Which judge agrees with a strong reference on our own tournament comparisons (no ground-truth labels exist)?

    ref_judge.py --arbiter-url http://127.0.0.1:PORT/v1 --out DIR [--n-per-method 200] [--n-swap 100] [--seed 0]

Samples logged step-1 reward comparisons from valid cells (half from WebArbiter runs, half from mini-PRM runs) and asks,
greedily and with the identical released judge prompt:
  ref       gpt-4.1-2025-04-14 (reference)          ref_swap  gpt-4.1 with Response 1/2 swapped (reference consistency)
  mini      gpt-4o-mini                              arb       WebArbiter-7B (released prompt)
  arb_bg    WebArbiter-7B on the A11 BrowserGym-format prompt (wa_protocol.to_browsergym)
Reports each judge's agreement with ref, overall and where mini and arb disagree, plus Response-2 rates and USD.
"""
import argparse, asyncio, collections, glob, json, random, re, sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
import wa_protocol as wp
from wa_report import unclosed_fence

CELLS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells')
ANSWER = re.compile(r'<Answer>\s*Response\s*([12])\s*</Answer>')
PRICES = {'gpt-4.1-2025-04-14': (2.0, 8.0), 'gpt-4o-mini': (0.15, 0.6)}
BLOCK = re.compile(r'\[The Begin of Response 1\](.*?)\[The End of Response 1\](\s*)\[The Begin of Response 2\](.*?)\[The End of Response 2\]', re.S)


def swap(user):
    return BLOCK.sub(lambda m: f'[The Begin of Response 1]{m.group(3)}[The End of Response 1]{m.group(2)}'
                               f'[The Begin of Response 2]{m.group(1)}[The End of Response 2]', user, count=1)


def load(n, seed):
    pools = collections.defaultdict(list)
    for cell in sorted(CELLS.glob('*-r[123]-*/task-*')):
        method = 'arbiter' if '-arbiter-' in cell.parent.name else 'mini_prm' if '-mini_prm-' in cell.parent.name else None
        s = cell / 'summary.json'
        if not method or not s.exists() or not json.loads(s.read_text())['valid'] or unclosed_fence(cell):
            continue
        for line in (cell / 'calls.jsonl').open():
            if '"role": "reward"' in line:
                r = json.loads(line)
                if BLOCK.search(r['user']):
                    pools[method].append(dict(source=method, cell=f'{cell.parent.name}/{cell.name}', system=r['system'], user=r['user']))
    rng = random.Random(seed)
    return [x for m in ('arbiter', 'mini_prm') for x in rng.sample(pools[m], n)]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arbiter-url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-per-method', type=int, default=200)
    ap.add_argument('--n-swap', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = load(a.n_per_method, a.seed)
    swap_ids = set(random.Random(a.seed + 1).sample(range(len(rows)), a.n_swap))
    api = AsyncOpenAI(timeout=600, max_retries=3)  # through the cluster proxy (trust_env)
    local = httpx.AsyncClient(timeout=900, trust_env=False)
    api_gate, local_gate = asyncio.Semaphore(8), asyncio.Semaphore(32)  # few concurrent outbound connections
    usd = collections.Counter()

    async def openai_judge(model, system, user):
        async with api_gate:
            try:
                resp = await api.chat.completions.create(model=model, temperature=0, top_p=1, max_tokens=2048,
                                                         messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
            except Exception as e:
                return -1, f'ERROR {type(e).__name__}: {e}'[:200]
        u = resp.usage; pi, po = PRICES[model]; usd[model] += (u.prompt_tokens * pi + u.completion_tokens * po) / 1e6
        text = resp.choices[0].message.content or ''
        m = ANSWER.search(text); return (int(m.group(1)) if m else 0), text[-300:]

    async def arbiter(system, user):
        async with local_gate:
            for _ in range(3):
                try:
                    d = (await local.post(f'{a.arbiter_url}/chat/completions', json=dict(model='WebArbiter', temperature=0, top_p=1,
                         max_tokens=2048, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))).json()
                    text = d['choices'][0]['message']['content'] or ''
                    m = ANSWER.search(text); return (int(m.group(1)) if m else 0), text[-300:]
                except Exception:
                    await asyncio.sleep(5)
        return -1, 'ERROR'

    async def one(i, r):
        jobs = dict(ref=openai_judge('gpt-4.1-2025-04-14', r['system'], r['user']),
                    mini=openai_judge('gpt-4o-mini', r['system'], r['user']),
                    arb=arbiter(r['system'], r['user']), arb_bg=arbiter(r['system'], wp.to_browsergym(r['user'])))
        if i in swap_ids:
            jobs['ref_swap'] = openai_judge('gpt-4.1-2025-04-14', r['system'], swap(r['user']))
        res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        rec = dict(i=i, source=r['source'], cell=r['cell'], **{k: v[0] for k, v in res.items()}, tails={k: v[1] for k, v in res.items()})
        with (a.out / 'verdicts.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        return rec

    recs = await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    ok = lambda v: v in (1, 2)
    summary = dict(n=len(recs), usd={k: round(v, 3) for k, v in usd.items()})
    for scope, sel in (('all', recs), ('from_arbiter_runs', [r for r in recs if r['source'] == 'arbiter']),
                       ('from_mini_runs', [r for r in recs if r['source'] == 'mini_prm']),
                       ('mini_vs_arb_disagree', [r for r in recs if ok(r['mini']) and ok(r['arb']) and r['mini'] != r['arb']])):
        s = summary[scope] = dict(n=len(sel))
        for j in ('mini', 'arb', 'arb_bg'):
            v = [r for r in sel if ok(r['ref']) and ok(r[j])]
            s[f'{j}_agrees_with_ref'] = round(100 * sum(r[j] == r['ref'] for r in v) / len(v), 1) if v else None
        s['response2_rate'] = {j: round(100 * sum(r[j] == 2 for r in sel) / len(sel), 1) if sel else None for j in ('ref', 'mini', 'arb', 'arb_bg')}
    sw = [r for r in recs if 'ref_swap' in r and ok(r['ref']) and ok(r['ref_swap'])]
    summary['ref_swap_consistent'] = round(100 * sum(r['ref_swap'] != r['ref'] for r in sw) / len(sw), 1) if sw else None
    summary['unparsed'] = {j: sum(r.get(j) in (0, -1) for r in recs) for j in ('ref', 'mini', 'arb', 'arb_bg')}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
