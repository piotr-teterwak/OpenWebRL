#!/usr/bin/env python3
"""Frozen embeddings for Best-of-Q from OpenWebRL-4B-SFT (Qwen3-VL-4B): final hidden layer, mean-pooled over tokens.

    embed.py states  --rows transitions.jsonl --start 0 --end 20000 --out DIR   # screenshot + task + URL + history
    embed.py texts   --rows transitions.jsonl --out DIR                         # unique action texts and tasks

Best-of-Q (arXiv 2601.22701) embeds state, action and task separately with a frozen VLM; here the VLM is our actor,
so the Q-function reads the policy's own representations (as Digi-Q). The state message mirrors the actor's
observation (task, current tab URL, executed-action history, screenshot) without its fixed system prompt, which
would dominate a mean over tokens. Outputs float16 .npy plus a JSON index; rows keep transitions.jsonl order.
"""
import argparse
import base64
import functools
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# BOQ_EMBEDDER selects the frozen VLM: our actor (first Best-of-Q attempt) or the paper's Qwen2.5-VL-3B-Instruct.
MODEL = os.environ.get('BOQ_EMBEDDER', '/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')


@functools.lru_cache(maxsize=4)
def _sample_images(path):
    return json.loads(Path(path).read_text())['images']


def screenshot(row):
    """A row's screenshot: a PNG path (collection rows) or [sample JSON, step] (Best-of-Q run rows)."""
    if row.get('screenshot_ref'):
        path, step = row['screenshot_ref']
        img = _sample_images(path)[step][0]
        return Image.open(io.BytesIO(base64.b64decode(img.split(',', 1)[1] if img.startswith('data:') else img))).convert('RGB')
    return Image.open(row['screenshot']).convert('RGB')


def load_model(device='cuda'):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = 'right'
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    return processor, model


def state_message(row):
    history = '\n'.join(f'{i + 1}. {a}' for i, a in enumerate(row['history'])) or '(none)'
    return [{'role': 'user', 'content': [
        {'type': 'text', 'text': f"<user_request>\n{row['task']}\n</user_request>\n\n<action_history>\n{history}\n"
                                 f"</action_history>\n\n<observation>\nCurrent tab: {row['url']}\nscreenshot:\n"},
        {'type': 'image'},
        {'type': 'text', 'text': '\n</observation>'}]}]


def text_message(text):
    return [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]


@torch.no_grad()
def pooled(model, batch):
    # The inner model's last_hidden_state is the final (normed) layer; skips the 151k-vocab lm_head logits.
    h = model.model(**batch, return_dict=True).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).to(h.dtype)
    return ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['states', 'texts'])
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int)
    ap.add_argument('--batch', type=int, default=8)
    a = ap.parse_args()
    processor, model = load_model()
    rows = [json.loads(l) for l in a.rows.open()]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.what == 'states':
        end = min(a.end or len(rows), len(rows)); idx = list(range(a.start, end)); vecs = []
        for b in range(0, len(idx), a.batch):
            chunk = [rows[i] for i in idx[b:b + a.batch]]
            texts = [processor.apply_chat_template(state_message(r), tokenize=False, add_generation_prompt=False) for r in chunk]
            images = [screenshot(r) for r in chunk]
            batch = processor(text=texts, images=images, return_tensors='pt', padding=True).to('cuda')
            vecs.append(pooled(model, batch))
            if b % (50 * a.batch) == 0:
                print(f'states {a.start + b}/{end}', flush=True)
        np.save(a.out / f'states-{a.start:07d}-{end:07d}.npy', np.concatenate(vecs))
    else:
        actions = sorted({r['action'] for r in rows} | {c for r in rows for c in r['candidates']})
        tasks = sorted({r['task'] for r in rows})
        for name, items in (('actions', actions), ('tasks', tasks)):
            vecs = []
            tb = int(os.environ.get('BOQ_TEXT_BATCH', 64))  # long texts (thought + action) need smaller batches on a MIG slice
            for b in range(0, len(items), tb):
                texts = [processor.apply_chat_template(text_message(t), tokenize=False, add_generation_prompt=False)
                         for t in items[b:b + tb]]
                vecs.append(pooled(model, processor(text=texts, return_tensors='pt', padding=True).to('cuda')))
            np.save(a.out / f'{name}.npy', np.concatenate(vecs))
            (a.out / f'{name}.json').write_text(json.dumps(items))
            print(name, len(items), flush=True)


if __name__ == '__main__':
    main()
