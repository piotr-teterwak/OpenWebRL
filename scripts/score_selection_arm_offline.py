#!/usr/bin/env python3
"""Run the released SelectionARM on saved candidate panels (one GPU, no browser).

Uses scripts/serve_arm.py's app in-process, so the canonical prompt builder,
screenshot handling and JSON-schema decoding are identical to live inference.
Writes {state_id, selected_index, raw} per panel; unparseable replies are kept
as unavailable (selected_index null), never replaced by candidate 1.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SELECTION = "/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=SELECTION)
    ap.add_argument("--base", default="/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
    ap.add_argument("--source-root", default="/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from starlette.testclient import TestClient
    from openwebrl.arm_inference import parse_selection, split_response
    from serve_arm import make_app

    app = make_app(SimpleNamespace(mode="selection", model=args.model, adapter=None, base=args.base,
                                   source_root=args.source_root, device=args.device, max_pixels=262144,
                                   context_length=32768, max_new_tokens=128))
    client = TestClient(app)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["state_id"] for line in args.output.open()}
    with open(args.panels) as stream, open(args.output, "a") as out:
        for line in stream:
            panel = json.loads(line)
            if panel["state_id"] in done:
                continue
            payload = {"mode": "selection", "task": panel["task"], "url": panel["url"],
                       "history": [split_response(r) for r in panel["history"]],
                       "candidates": [split_response(c) for c in panel["candidates"]],
                       "screenshot": base64.b64encode(Path(panel["image"]).read_bytes()).decode()}
            reply = client.post("/select", json=payload)
            reply.raise_for_status()
            result = reply.json()
            try:
                index = parse_selection(result.get("raw") or "", 5)
            except ValueError:
                index = None
            out.write(json.dumps({"state_id": panel["state_id"], "selected_index": index,
                                  "raw": result.get("raw"), "input_tokens": result.get("input_tokens")}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
