#!/usr/bin/env bash
# One-off: convert a Megatron distcp checkpoint to HF format and verify the result.
# Validates the native-stack evaluation path (run_evaluate.py takes --hf_checkpoint).
# Needs a GPU: bridge.export_ckpt materializes the model on CUDA.
#
#   CONVERT_INPUT=/path/to/iter_0000001 GPUS=1 SCRIPT=scripts/convert_ckpt_to_hf.sh \
#     bash beaker/launch_train.sh
set -euo pipefail

IN="${CONVERT_INPUT:?set CONVERT_INPUT to an iter_* checkpoint dir}"
OUT="${CONVERT_OUTPUT:-${IN}/hf_converted}"
ORIGIN="${ORIGIN_HF_DIR:-/weka/oe-training-default/zixianm/OpenWebRL/models/OpenWebRL/OpenWebRL-4B-SFT}"

echo "[convert] input  = $IN"
echo "[convert] output = $OUT"
echo "[convert] origin = $ORIGIN"
nvidia-smi -L

python tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir "$IN" --output-dir "$OUT" --origin-hf-dir "$ORIGIN" -f

echo "[convert] === output contents ==="
ls -la "$OUT"

echo "[convert] === verification ==="
CONVERT_OUT="$OUT" CONVERT_ORIGIN="$ORIGIN" python - <<'PY'
import os, glob, json
import torch
from safetensors import safe_open
from transformers import AutoConfig

out, origin = os.environ["CONVERT_OUT"], os.environ["CONVERT_ORIGIN"]

cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
print("config loads OK:", type(cfg).__name__)

def tensors(d):
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"FAIL: no safetensors in {d}")
    out = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = (f, h.get_slice(k).get_shape())
    return out

a, b = tensors(out), tensors(origin)
print(f"converted tensors: {len(a)}   origin tensors: {len(b)}")

missing, extra = set(b) - set(a), set(a) - set(b)
if missing or extra:
    print(f"FAIL: key mismatch. missing={len(missing)} extra={len(extra)}")
    for k in list(missing)[:5]: print("  missing:", k)
    for k in list(extra)[:5]:   print("  extra:  ", k)
    raise SystemExit(1)
print("all tensor names match origin")

bad = [(k, a[k][1], b[k][1]) for k in a if a[k][1] != b[k][1]]
if bad:
    print("FAIL: shape mismatch, e.g.", bad[:3])
    raise SystemExit(1)
print("all tensor shapes match origin")

# The load-bearing check: the converted weights must DIFFER from the base model,
# otherwise the conversion silently produced a copy of the SFT checkpoint and every
# eval downstream would be measuring the wrong model.
changed, total, max_delta = 0, 0, 0.0
for k in sorted(a)[:80]:
    fa, fb = a[k][0], b[k][0]
    with safe_open(fa, framework="pt") as ha, safe_open(fb, framework="pt") as hb:
        ta, tb = ha.get_tensor(k).float(), hb.get_tensor(k).float()
    d = (ta - tb).abs().max().item()
    total += 1
    if d > 0:
        changed += 1
        max_delta = max(max_delta, d)
print(f"weights differ from base in {changed}/{total} sampled tensors, max |delta| = {max_delta:.3e}")
if changed == 0:
    print("FAIL: converted weights are identical to the base model — conversion did not "
          "pick up trained weights")
    raise SystemExit(1)
print("PASS: conversion produced a valid, trained HF checkpoint")
PY
echo "[convert] done"
