#!/usr/bin/env python3
"""Sample fresh actor candidates at stored collection states (mixed-pool Round 2, 2026-11-19).

    sample_sft_candidates.py --dataset <c2-collect/NAME-all-shardK/training.jsonl> [...] --port <sglang> \
        --out sft-samples.jsonl [--n 3] [--seed 42] [--limit 50]

For every row of an exported collection dataset (scripts/export_failed_turns.py --include-successful: one row per
executed turn of a valid episode), re-sends the exact actor prompt that state's actor saw -- the stored turn record's
chat-templated `prompt` text and screenshot, which openwebrl.generate_browser._run_inference_step posted to SGLang
/generate -- to a running SGLang server (e.g. the SFT model), with the collection sampling (T 0.7, top-p 0.9, 1,024 new
tokens) and a deterministic per-(state, index) sampling seed, candidate_seed('sftmix-<seed>', task, turn, index).
The response is normalised exactly as the collection generator does (_ensure_im_end_w_new_line), so the stored text
and token ids match what arm_c2 would have recorded for a candidate.

Prompt-reconstruction check, per state: the turn record's sha256s (record, prompt text, screenshot) must match, and
SGLang's prompt token count (meta_info.prompt_tokens, image tokens expanded) is recorded next to the stored
len(prompt_token_ids) of the executed turn (`prompt_tokens_match`). Resumable: finished ids are skipped.
One JSON line per state: {id, task_id, turn, source, prompt_sha256, prompt_tokens, stored_prompt_tokens,
prompt_tokens_match, candidates: [{response, response_token_ids, finish_type, sampling_seed}]}.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path

from openwebrl.arm_inference import candidate_seed


def sha(data): return hashlib.sha256(data).hexdigest()


def ensure_im_end(text, tokens):
    """openwebrl.generate_browser._ensure_im_end_w_new_line, without the logprobs."""
    text = text.rstrip()
    if not text.endswith('<|im_end|>'): text += '<|im_end|>'
    text += '\n'
    tokens = list(tokens)
    tokens += [151645, 198] if tokens[-1] != 151645 else [198]
    return text, tokens


def load_states(datasets):
    for dataset in datasets:
        for line in Path(dataset).read_text().splitlines():
            row = json.loads(line)
            yield row


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', nargs='+', required=True, help='exported collection training.jsonl file(s)')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=3, help='candidates per state')
    ap.add_argument('--seed', type=int, default=42, help='seed label of the sampling seeds (use the collection seed)')
    ap.add_argument('--temperature', type=float, default=.7)
    ap.add_argument('--top-p', type=float, default=.9)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--parallel', type=int, default=32, help='states in flight (x n requests)')
    ap.add_argument('--limit', type=int, help='first N states only (pilot)')
    ap.add_argument('--model', help='expected served model path (checked against /get_model_info)')
    args = ap.parse_args()
    import httpx
    done = set()
    if args.out.exists():
        done = {json.loads(l)['id'] for l in args.out.read_text().splitlines()}
    rows = list(load_states(args.dataset))
    if args.limit: rows = rows[:args.limit]
    todo = [r for r in rows if f"C2:{r['task_id']}:{r['turn']}" not in done]
    print(f'{len(rows)} states, {len(done)} already sampled, {len(todo)} to go', flush=True)
    base = f'http://127.0.0.1:{args.port}'
    gate = asyncio.Semaphore(args.parallel)
    stats = dict(sampled=0, failures=0, token_mismatch=0)
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        if args.model:
            info = (await client.get(base + '/get_model_info')).json()
            if Path(info['model_path']).resolve() != Path(args.model).resolve(): raise SystemExit(f'wrong served model {info}')

        async def one(row, stream):
            key = f"C2:{row['task_id']}:{row['turn']}"
            async with gate:
                try:
                    raw = Path(row['source']).read_bytes()
                    if sha(raw) != row['source_sha256']: raise ValueError('turn record hash')
                    turn = json.loads(raw)
                    if (turn['task_id'], turn['turn']) != (row['task_id'], row['turn']): raise ValueError('turn record join')
                    if sha(turn['prompt'].encode()) != turn['prompt_sha256']: raise ValueError('prompt hash')
                    if len(turn['images']) != 1: raise ValueError('expected one screenshot')
                    png = Path(turn['images'][0]['path']).read_bytes()
                    if sha(png) != turn['images'][0]['sha256'] or sha(png) != turn['screenshot_sha256']: raise ValueError('screenshot hash')
                    image = 'data:image/png;base64,' + base64.b64encode(png).decode()
                    seeds = [candidate_seed(f'sftmix-{args.seed}', row['task_id'], row['turn'], j) for j in range(args.n)]

                    async def generate(s):
                        payload = dict(text=turn['prompt'], image_data=[image], return_logprob=True,
                                       sampling_params=dict(temperature=args.temperature, top_p=args.top_p,
                                                            max_new_tokens=args.max_new_tokens, sampling_seed=s))
                        r = await client.post(base + '/generate', json=payload); r.raise_for_status(); return r.json()
                    # first sample alone, so the others reuse its prefilled prompt from SGLang's radix cache
                    outs = [await generate(seeds[0])] + list(await asyncio.gather(*(generate(s) for s in seeds[1:])))
                    candidates, counts = [], set()
                    for s, o in zip(seeds, outs):
                        meta = o['meta_info']; counts.add(meta['prompt_tokens'])
                        text, ids = ensure_im_end(o['text'], [x[1] for x in meta['output_token_logprobs']])
                        candidates.append(dict(response=text, response_token_ids=ids,
                                               finish_type=(meta.get('finish_reason') or {}).get('type', 'stop'), sampling_seed=s))
                    stored = len(row['prompt_token_ids'])
                    match = counts == {stored}
                    stats['token_mismatch'] += not match
                    out = dict(id=key, task_id=row['task_id'], turn=row['turn'], source=row['source'],
                               prompt_sha256=turn['prompt_sha256'], prompt_tokens=sorted(counts), stored_prompt_tokens=stored,
                               prompt_tokens_match=match, candidates=candidates)
                except Exception as exc:  # a failed state is left unsampled, never guessed
                    stats['failures'] += 1
                    print('FAIL', key, type(exc).__name__, str(exc)[:160], flush=True)
                    return
                stream.write(json.dumps(out, ensure_ascii=False) + '\n'); stream.flush()
                stats['sampled'] += 1
                if stats['sampled'] % 500 == 0: print(json.dumps(stats), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('a') as stream:
            await asyncio.gather(*(one(r, stream) for r in todo))
    rows = [json.loads(l) for l in args.out.read_text().splitlines()]
    print(json.dumps(dict(stats, total_rows=len(rows), prompt_tokens_match=sum(r['prompt_tokens_match'] for r in rows),
                          truncated=sum(c['finish_type'] != 'stop' for r in rows for c in r['candidates']))), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
