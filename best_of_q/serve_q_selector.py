#!/usr/bin/env python3
"""Serve the Best-of-Q Q-function as an arm_eval `scalar` selector: /select scores each candidate, argmax executes.

    serve_q_selector.py --q q_function_best.pt --port 19101 [--device cuda:0]

Same HTTP contract as scripts/serve_arm.py --mode scalar ({"scores": [...]}, /health {"mode": "scalar"}). Embeddings are
produced exactly as in embed.py (frozen OpenWebRL-4B-SFT, final layer mean-pooled, cast to float16): the state message
(task, executed-action history, current URL, screenshot), each candidate's action text, and the task text. The eval
passes `task` = the task intent, `history` = [{thought, action}] of executed responses, `url` = active tab.
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL, load_model, pooled, state_message, text_message  # noqa: E402
from train_q import Tower  # noqa: E402


class SelectionRequest(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str


def make_app(args):
    processor, model = load_model(args.device)  # BOQ_EMBEDDER: must be the embedder the Q-function was trained on
    ck = torch.load(args.q, map_location=args.device)
    q = Tower(ck['dim'], 3).to(args.device); q.load_state_dict(ck['q']); q.eval()
    info = dict(mode='scalar', selector='best_of_q', q_function=str(Path(args.q).resolve()), q_step=ck.get('step'),
                q_val_auc=ck.get('val_v0_success_auc'), embedder=MODEL, dim=ck['dim'])
    if ck.get('embedder') and ck['embedder'] != MODEL:
        raise SystemExit(f"Q-function was trained on {ck['embedder']} embeddings, server embeds with {MODEL}")
    app = FastAPI(); log = (Path(args.log) if args.log else None)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)

    def embed_texts(texts):
        batch = processor(text=[processor.apply_chat_template(text_message(t), tokenize=False, add_generation_prompt=False)
                                for t in texts], return_tensors='pt', padding=True).to(args.device)
        return pooled(model, batch)

    @app.get('/health')
    def health():
        return info

    @app.post('/select')
    def select(request: SelectionRequest):
        started = time.monotonic()
        if request.mode != 'scalar':
            raise ValueError(f'Q selector serves scalar mode, got {request.mode}')
        row = dict(task=request.task, url=request.url or '', history=[(h.get('action') or '').strip() for h in request.history])
        img = Image.open(io.BytesIO(base64.b64decode(request.screenshot))).convert('RGB')
        with torch.no_grad():
            text = processor.apply_chat_template(state_message(row), tokenize=False, add_generation_prompt=False)
            s = pooled(model, processor(text=[text], images=[img], return_tensors='pt', padding=True).to(args.device))
            actions = [(c.get('action') or '').strip() for c in request.candidates]
            a = embed_texts(actions); t = embed_texts([request.task])
            # training stored float16 embeddings and trained on their float32 values
            S = torch.tensor(np.repeat(s, len(actions), 0).astype(np.float32), device=args.device)
            A = torch.tensor(a.astype(np.float32), device=args.device)
            T = torch.tensor(np.repeat(t, len(actions), 0).astype(np.float32), device=args.device)
            scores = [float(v) for v in q(S, A, T).cpu()]
        if log:
            with log.open('a') as f:
                f.write(json.dumps(dict(task=request.task[:120], url=request.url, n=len(actions), scores=scores,
                                        actions=[x[:160] for x in actions], seconds=round(time.monotonic() - started, 2))) + '\n')
        return {'scores': scores, 'seconds': time.monotonic() - started}
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--q', required=True)
    ap.add_argument('--port', type=int, default=19101)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--log', help='JSONL of every scoring call')
    args = ap.parse_args()
    uvicorn.run(make_app(args), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
