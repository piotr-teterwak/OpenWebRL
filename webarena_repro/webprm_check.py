#!/usr/bin/env python3
"""WebArbiter-7B on WebPRMBench (labelled pairs), to separate checkpoint/serving from how the tournament feeds it.

    webprm_check.py --url http://127.0.0.1:PORT/v1 --data WebPRMBench.jsonl --out DIR [--conditions C0 C1 C2 C3]

C0  authors' eval_one_command.sh: raw prompt (no chat template), greedy, max_tokens 1024, all sources.
    Paper Table 2 target: pairwise 89.19 / BoN 74.60 (WebArena 88.43 / 68.66).
C1  WebArena source, delivered like the released tournament: system = text before '#### Intent ####', user = rest
    (chat template applied by the server), greedy.
C2  C1 + BrowserGym actions rewritten to the WebArena bracket syntax the tournament shows (click('5') -> click [5]).
C3  C2 + the released tournament's default decoding (model_config.py: temperature 1.0, top_p 0.9).
C4  C2 + BrowserGym AXTree rewritten in the WebArena ScriptBrowserEnv style the tournament shows: `key: value`
    attributes, unnamed list/listitem/Section wrappers pruned (children dedented), a RootWebArea root (no link URLs:
    WebPRMBench does not record them).
Scoring and parsing are the authors' (evaluate_webprm_vllm.parse_answer); a state counts if all its pairs are right.
"""
import argparse, asyncio, collections, importlib.util, json, re
from pathlib import Path

import httpx

AUTHORS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26/eval/WebPRMBench/script/evaluate_webprm_vllm.py')
spec = importlib.util.spec_from_file_location('authors_eval', AUTHORS); authors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(authors)
PAPER = dict(pair=89.19, bon=74.60, webarena_pair=88.43, webarena_bon=68.66)
CALL = re.compile(r"\b(click|hover|fill|press|scroll|go_back|go_forward|send_msg_to_user|select_option|goto|new_tab|"
                  r"tab_close|focus|clear|dblclick)\(([^()]*)\)")


def args_of(s):
    out = []
    for m in re.finditer(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?)""", s):
        out.append(next(g for g in m.groups() if g is not None))
    return out


def bracket(m):
    name, a = m.group(1), args_of(m.group(2))
    if name == 'fill' and len(a) >= 2:
        return f'type [{a[0]}] [{a[1]}]'
    if name == 'press' and a:
        return f'press [{a[-1]}]'
    if name == 'scroll' and len(a) >= 2:
        return 'scroll [down]' if float(a[1]) >= 0 else 'scroll [up]'
    if name == 'tab_close':
        return 'close_tab'
    if name == 'select_option' and len(a) >= 2:
        return f'select_option [{a[0]}] [{a[1]}]'
    return name + ''.join(f' [{x}]' for x in a)


def to_bracket(prompt):
    """Rewrite actions in the Trajectory and Assistant Responses sections only (the AXTree is left alone)."""
    i = prompt.find('#### Trajectory ####')
    return prompt if i < 0 else prompt[:i] + CALL.sub(bracket, prompt[i:])


ATTR = re.compile(r", (\w+)=('(?:[^'\\]|\\.)*'|\w+)")
WRAPPER = re.compile(r"^\[\w+\] (?:list|listitem|Section) ''$")


def webarena_tree(prompt):
    """Rewrite the AXTREE section of a WebPRMBench prompt into the WebArena accessibility-tree style."""
    i = prompt.find('#### AXTREE ####'); j = prompt.find('#### ', i + 16)
    if i < 0 or j < 0:
        return prompt
    head, *lines = prompt[i:j].split('\n')
    note = lines.pop(0) if lines and lines[0].startswith('Note:') else None
    out, removed = [], []  # removed: indent levels of pruned ancestors still open
    for line in lines:
        if not line.strip():
            continue
        level = len(line) - len(line.lstrip('\t')); body = line.strip()
        while removed and removed[-1] >= level:
            removed.pop()
        if WRAPPER.match(body):
            removed.append(level); continue
        body = ATTR.sub(lambda m: f" {m.group(1)}: {m.group(2).strip(chr(39))}", body)
        out.append('\t' * (level - len(removed) + 1) + body)
    tree = '\n'.join([head] + ([note] if note else []) + ["[1] RootWebArea '' focused: True"] + out) + '\n\n'
    return prompt[:i] + tree + prompt[j:]


def as_chat(prompt):
    i = prompt.find('#### Intent ####')
    return [{'role': 'system', 'content': prompt[:i].rstrip()}, {'role': 'user', 'content': prompt[i:]}]


async def run(cond, rows, a, client):
    gate = asyncio.Semaphore(a.concurrency)
    sampling = dict(temperature=1.0, top_p=0.9) if cond == 'C3' else dict(temperature=0.0, top_p=1.0)
    out = a.out / f'{cond}.jsonl'
    done = {json.loads(l)['i'] for l in out.open()} if out.exists() else set()

    async def one(i, r):
        p = r['prompt'] if cond in ('C0', 'C1') else to_bracket(r['prompt'])
        if cond == 'C4':
            p = webarena_tree(p)
        async with gate:
            for attempt in range(3):
                try:
                    if cond == 'C0':
                        resp = await client.post(f'{a.url}/completions', json=dict(model='WebArbiter', prompt=p, max_tokens=1024, **sampling))
                        text = resp.json()['choices'][0]['text']
                    else:
                        resp = await client.post(f'{a.url}/chat/completions', json=dict(model='WebArbiter', messages=as_chat(p), max_tokens=1024, **sampling))
                        text = resp.json()['choices'][0]['message']['content']
                    usage = resp.json().get('usage', {})
                    break
                except Exception as e:
                    if attempt == 2:
                        text, usage = f'ERROR {type(e).__name__}: {e}', {}
                    await asyncio.sleep(5)
        pred = authors.parse_answer(text)
        with out.open('a') as f:
            f.write(json.dumps(dict(i=i, source=r['source_name'], state=r['state_idx'], expected=r['expected'], pred=pred,
                                    correct=pred == r['expected'], prompt_tokens=usage.get('prompt_tokens'), raw=text[-400:])) + '\n')
    await asyncio.gather(*(one(i, r) for i, r in rows if i not in done))
    return summarize(cond, [json.loads(l) for l in out.open()])


def summarize(cond, recs):
    by = collections.defaultdict(list)
    for r in recs:
        group = 'mind2web' if r['source'].startswith('mind2web') else r['source']
        by[group].append(r)
    res = {}
    for g, rs in sorted(by.items()):
        states = collections.defaultdict(list)
        for r in rs:
            states[(r['source'], r['state'])].append(r['correct'])
        res[g] = dict(pairs=len(rs), pair_acc=100 * sum(r['correct'] for r in rs) / len(rs),
                      bon_acc=100 * sum(all(v) for v in states.values()) / len(states), unparsed=sum(r['pred'] == 0 for r in rs))
    if len(res) > 1:
        res['avg_of_groups'] = dict(pair_acc=sum(v['pair_acc'] for v in res.values()) / len(res),
                                    bon_acc=sum(v['bon_acc'] for v in res.values()) / len(res))
    lengths = [(r['prompt_tokens'] or 0, r['correct']) for r in recs if r['prompt_tokens']]
    for lo, hi in ((0, 4096), (4096, 8192), (8192, 10 ** 9)):
        b = [c for t, c in lengths if lo <= t < hi]
        res[f'tokens_{lo}-{hi if hi < 10 ** 9 else "inf"}'] = dict(pairs=len(b), pair_acc=100 * sum(b) / len(b) if b else None)
    print(cond, json.dumps(res), flush=True)
    return res


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--conditions', nargs='*', default=['C0', 'C1', 'C2', 'C3', 'C4'])
    ap.add_argument('--concurrency', type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in a.data.open()]
    summary = {}
    async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
        for cond in a.conditions:
            sel = list(enumerate(rows)) if cond == 'C0' else [(i, r) for i, r in enumerate(rows) if r['source_name'] == 'webarena']
            summary[cond] = await run(cond, sel, a, client)
    (a.out / 'summary.json').write_text(json.dumps(dict(paper=PAPER, results=summary), indent=1))


if __name__ == '__main__':
    asyncio.run(main())
