#!/usr/bin/env python3
"""Scalar value-head RM scoring server.

POST /score {intent, url, history[], image_b64, candidates[{thought, action}]}
-> {"scores": [float, ...]}  (batched forward over candidates)
"""
import argparse
import base64
import io
import os

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = os.environ.get("SCALAR_BASE", "OpenWebRL/OpenWebRL-4B-SFT")
ADAPTER = os.environ.get("SCALAR_ADAPTER", "PTeterwak/OpenWebRL-4B-ScalarRM-LoRA")

SYSTEM = (
    "You are an expert web-agent action evaluator. Given the task, the current "
    "page state and screenshot, and ONE proposed next action, judge how well "
    "the action advances the task."
)

app = FastAPI()
model = processor = w = b = None


def action_prompt(intent, url, hist):
    hist_s = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(hist[-8:])) or "  (none)"
    return (
        f"Task: {intent}\nCurrent URL: {url}\nRecent actions:\n{hist_s}\n"
        f"Current page screenshot: <image>\n\nProposed action:\n(see candidate below)"
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/score")
async def score(req: dict):
    intent, url = req.get("intent", ""), req.get("url", "")
    hist = req.get("history") or []
    img_b64 = req["image_b64"]
    if "base64," in img_b64:
        img_b64 = img_b64.split("base64,", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    up = action_prompt(intent, url, hist)
    pre, post = up.split("<image>")

    texts, images = [], []
    for c in req["candidates"]:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": pre},
                {"type": "image"},
                {"type": "text", "text": post},
            ]},
            {"role": "assistant", "content":
                f"Action: {c['action'].strip()[:800]}\nReasoning: {(c.get('thought') or '')[:400]}"},
        ]
        texts.append(processor.apply_chat_template(msgs, tokenize=False))
        images.append(img)
    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1]  # [n, seq, hid]
    # score at each sequence's last non-pad token
    mask = inputs["attention_mask"]
    last = mask.sum(dim=1) - 1
    scores = []
    for i in range(h.size(0)):
        v = h[i, last[i]].to(w.dtype) @ w.T
        if b is not None:
            v = v + b
        scores.append(float(v.squeeze()))
    return {"scores": scores}


def main():
    global model, processor, w, b
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9800)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    m = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model = PeftModel.from_pretrained(m, ADAPTER)
    model.eval()
    if os.path.isdir(ADAPTER):
        vh_path = os.path.join(ADAPTER, "value_head.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        vh_path = hf_hub_download(ADAPTER, "value_head.safetensors")
    vh = load_file(vh_path)
    w = vh["v_head.summary.weight"].to(device="cuda", dtype=torch.bfloat16)
    bb = vh.get("v_head.summary.bias")
    b = bb.to(device="cuda", dtype=torch.bfloat16) if bb is not None else None
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
