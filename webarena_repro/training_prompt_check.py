#!/usr/bin/env python3
"""Does WebArbiter judge better on the prompt it was trained on (paper Appendix D) than on the released tournament prompt?

    training_prompt_check.py --url http://127.0.0.1:PORT/v1 --out DIR --ref-dir runs/ref-judge-414435

(i) WebPRMBench WebArena pairs (804, labelled): the benchmark prompt as shipped (fixed criteria block, raw string) vs the
    training format (training system text; user from '#### Intent ####' on, criteria block removed).
(ii) The 400 logged tournament comparisons of the reference-judge test (same seed, same rows): WebArbiter on the
     training format vs its stored verdicts on the released prompt, each scored by agreement with the stored gpt-4.1
     verdicts. Greedy decoding throughout; no paid calls.
"""
import argparse, asyncio, json, re, sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import wa_protocol as wp
import ref_judge

ANSWER = re.compile(r'<Answer>\s*Response\s*([12])\s*</Answer>')
WPB = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/WebPRMBench.jsonl')


def training_split(prompt):
    i = prompt.find('#### Intent ####')
    return wp.TRAINING_JUDGE_SYSTEM, prompt[i:]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ref-dir', type=Path, required=True); ap.add_argument('--concurrency', type=int, default=48)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(a.concurrency)

    async def ask(client, system, user=None, raw=None):
        async with gate:
            for _ in range(3):
                try:
                    if raw is not None:
                        d = (await client.post(f'{a.url}/completions', json=dict(model='WebArbiter', prompt=raw, max_tokens=1024, temperature=0.0, top_p=1.0))).json()
                        text = d['choices'][0]['text']
                    else:
                        d = (await client.post(f'{a.url}/chat/completions', json=dict(model='WebArbiter', max_tokens=2048, temperature=0.0, top_p=1.0,
                             messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))).json()
                        text = d['choices'][0]['message']['content']
                    m = ANSWER.search(text or ''); return int(m.group(1)) if m else 0
                except Exception:
                    await asyncio.sleep(5)
        return -1

    out = {}
    async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
        rows = [json.loads(l) for l in WPB.open() if '"webarena"' in l]
        rel = await asyncio.gather(*(ask(client, None, raw=r['prompt']) for r in rows))
        trn = await asyncio.gather(*(ask(client, *training_split(r['prompt'])) for r in rows))
        def acc(preds):
            ok = [p == r['expected'] for p, r in zip(preds, rows)]; states = {}
            for p, r in zip(preds, rows): states.setdefault(r['state_idx'], []).append(p == r['expected'])
            return dict(pairs=len(rows), pair_acc=round(100 * sum(ok) / len(ok), 2), bon_acc=round(100 * sum(all(v) for v in states.values()) / len(states), 2), unparsed=sum(p == 0 for p in preds))
        flips = sum(p != q for p, q in zip(rel, trn) if p in (1, 2) and q in (1, 2))
        out['webprmbench_webarena'] = dict(released_prompt=acc(rel), training_prompt=acc(trn), verdict_flips=flips)
        print(json.dumps(out), flush=True)

        sample = ref_judge.load(200, 0)
        stored = {r['i']: r for r in map(json.loads, (a.ref_dir / 'verdicts.jsonl').open())}
        trn2 = await asyncio.gather(*(ask(client, *wp.training_judge_prompt(r['system'], r['user'])) for r in sample))
        def agree(get):
            v = [(get(i), stored[i]['ref']) for i in range(len(sample)) if i in stored and stored[i]['ref'] in (1, 2) and get(i) in (1, 2)]
            return round(100 * sum(p == q for p, q in v) / len(v), 1) if v else None
        out['logged_comparisons'] = dict(n=len(sample), released_prompt_agrees_with_gpt41=agree(lambda i: stored[i]['arb']),
                                         training_prompt_agrees_with_gpt41=agree(lambda i: trn2[i]), mini_agrees_with_gpt41=agree(lambda i: stored[i]['mini']),
                                         flips_vs_released=sum(trn2[i] != stored[i]['arb'] for i in range(len(sample)) if i in stored),
                                         training_prompt_response2_rate=round(100 * sum(v == 2 for v in trn2) / len(trn2), 1))
        with (a.out / 'logged_verdicts.jsonl').open('w') as f:
            for i, v in enumerate(trn2):
                f.write(json.dumps(dict(i=i, training=v, released=stored.get(i, {}).get('arb'), ref=stored.get(i, {}).get('ref'))) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
