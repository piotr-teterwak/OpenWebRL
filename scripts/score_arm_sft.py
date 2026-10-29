#!/usr/bin/env python3
"""Teacher agreement of a Selection ARM on a build_selection_sft.py dataset (action-reward-models format).

Sends each sample's own system + user messages (the training prompt, screenshot where <image> sits)
to an sglang server hosting the merged ARM, greedy, and compares the parsed {"selection": N} with the
sample's target. Screenshots are downscaled to the training image_max_pixels (arm_lora.yaml, 262144)
before sending, so the server sees the resolution training saw.

    python scripts/score_arm_sft.py --data <sft-test dir> --port 30000 --out scores.jsonl
"""
import argparse
import asyncio
import base64
import io
import json
import math
from pathlib import Path
import re

MAX_PIXELS = 262144
SELECTION = re.compile(r'"selection"\s*:\s*(\d+)')


def image_url(path):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--port', type=int, default=30000)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parallel', type=int, default=32)
    a = ap.parse_args()
    import httpx
    info = json.loads((a.data / 'dataset_info.json').read_text())
    rows = json.loads((a.data / next(iter(info.values()))['file_name']).read_text())
    gate = asyncio.Semaphore(a.parallel)

    async def one(i, row, client):
        system, user, target = row['messages']
        before, after = user['content'].split('<image>')
        msgs = [{'role': 'system', 'content': system['content']},
                {'role': 'user', 'content': [{'type': 'text', 'text': before},
                                             {'type': 'image_url', 'image_url': {'url': image_url(a.data / row['images'][0])}},
                                             {'type': 'text', 'text': after}]}]
        async with gate:
            r = await client.post(f'http://127.0.0.1:{a.port}/v1/chat/completions',
                                  json=dict(model='arm', messages=msgs, temperature=0, max_tokens=64))
        r.raise_for_status()
        text = r.json()['choices'][0]['message']['content'] or ''
        want = int(SELECTION.search(target['content']).group(1))
        got = SELECTION.findall(text)
        pick = int(got[-1]) if got else None
        return dict(i=i, target=want, pick=pick, correct=pick == want, raw=text[-120:])

    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        results = await asyncio.gather(*(one(i, r, client) for i, r in enumerate(rows)))
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in results))
    n = len(results)
    summary = dict(data=str(a.data), samples=n, agreement=sum(r['correct'] for r in results) / n,
                   unparsed=sum(r['pick'] is None for r in results),
                   pick_histogram={k: sum(r['pick'] == k for r in results) for k in range(1, 6)},
                   target_histogram={k: sum(r['target'] == k for r in results) for k in range(1, 6)})
    a.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
