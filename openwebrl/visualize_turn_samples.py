#!/usr/bin/env python
"""
Render saved turn samples to a single self-contained HTML report.

Complements visualize_eval_trajectory.py (Jupyter, one task at a time) with a
batch view: many trajectories, full-width annotated screenshots, and a
side-by-side of the exact prompt the policy receives at different values of
--context-num-screenshots.

The turn samples written by run_evaluate.py contain everything needed:
  sample_id, total_steps, llm_input_texts, llm_response, images, status,
  terminate_reason
`images` holds one screenshot per step; `llm_input_texts` is the final turn's
prompt, already pruned to the last K screenshots by generate_browser.py.

Usage:
  python -m openwebrl.visualize_turn_samples \
      --eval-dir <.../eval_samples/turn/<ts>>  --out report.html
"""

from __future__ import annotations

import argparse
import base64
import binascii
import html
import json
import math
import os
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageDraw

# ---------------------------------------------------------------- constants

# Inserted by generate_browser.py into every user turn that carries a
# screenshot, and str.replace()d out of turns outside the last-K window.
SCREENSHOT_PLACEHOLDER = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>"
OBSERVATION_CLOSE = "\n</observation>"

_CHAT_BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_USER_REQUEST_RE = re.compile(r"<user_request>\s*(.*?)\s*</user_request>", re.DOTALL)
_ACTIVE_TAB_RE = re.compile(r"- Tab (\d+) \(active\): (\S+)")

FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]

# One colour per action family. Kept in sync with the HTML legend.
ACTION_COLORS = {
    "click": (255, 56, 96),
    "hover": (0, 200, 255),
    "scroll": (255, 176, 32),
    "drag": (177, 76, 255),
    "write": (46, 220, 140),
    "press_keys": (46, 220, 140),
}
NAV_COLOR = (138, 180, 255)
POINTLESS = {  # actions with no on-screen coordinate -> rendered as banner chips
    "write", "press_keys", "goto_url", "go_back", "new_tab", "switch_tab",
    "close_tab", "wait", "done",
}


def _font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- parsing


def _decode_data_url(data_url) -> bytes:
    """`images[i]` is the list of screenshots in turn i's context; the last is current."""
    while isinstance(data_url, (list, tuple)):
        if not data_url:
            raise ValueError("empty image entry")
        data_url = data_url[-1]
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    return base64.b64decode(payload)


def parse_tool_calls(text: str | None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text or ""):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            calls.append({"name": "<unparseable>", "arguments": {}, "_raw": raw})
    return calls


def parse_chat_blocks(text: str) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in _CHAT_BLOCK_RE.findall(text or "")]


def to_pixel(point_2d, width: int, height: int) -> tuple[int, int]:
    """point_2d is normalised to [0, 1000] on both axes."""
    x = round(float(point_2d[0]) / 1000.0 * width)
    y = round(float(point_2d[1]) / 1000.0 * height)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def summarize_call(call: dict[str, Any]) -> str:
    """One-line human label for an action, used on the overlay chip."""
    name = call.get("name", "?")
    args = call.get("arguments") or {}
    if name == "click":
        n = args.get("clicks", 1) or 1
        btn = args.get("button", "left") or "left"
        kind = {1: "click", 2: "double-click", 3: "triple-click"}.get(n, f"{n}x click")
        return kind if btn == "left" else f"{btn} {kind}"
    if name == "write":
        msg = str(args.get("message", ""))
        return f'type "{msg[:52]}{"…" if len(msg) > 52 else ""}"'
    if name == "press_keys":
        keys = args.get("keys") or []
        return " + ".join(str(k) for k in keys) if args.get("is_hotkey") else " ".join(str(k) for k in keys)
    if name == "scroll":
        return f"scroll {args.get('direction', '?')} {args.get('amount', '')}".strip()
    if name == "goto_url":
        return f"goto {str(args.get('url', ''))[:60]}"
    if name == "switch_tab":
        return f"switch to tab {args.get('tab_index')}"
    if name == "wait":
        return f"wait {args.get('seconds', '')}s"
    if name == "done":
        return "done"
    if name == "drag":
        return "drag"
    return name


# ---------------------------------------------------------------- overlays


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _chip(draw, xy, text, font, fill, fg=(10, 12, 16), pad=7, placed=None):
    """
    Rounded label chip, clamped inside the image and nudged clear of chips
    already drawn — several actions in one turn often share an anchor point.
    """
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l, b - t
    W, H = draw.im.size
    bw, bh = w + 2 * pad, h + 2 * pad
    x = max(4, min(W - bw - 4, x))
    y = max(4, min(H - bh - 4, y))
    box = (x, y, x + bw, y + bh)
    if placed is not None:
        for _ in range(10):
            hit = next((q for q in placed if _overlaps(box, q)), None)
            if hit is None:
                break
            y = hit[3] + 6
            if y + bh > H - 4:                     # ran out of room below
                y = max(4, box[1] - bh - 6)
            box = (x, y, x + bw, y + bh)
        placed.append(box)
    try:
        draw.rounded_rectangle(box, radius=7, fill=fill)
    except AttributeError:  # very old Pillow
        draw.rectangle(box, fill=fill)
    draw.text((x + pad - l, y + pad - t), text, font=font, fill=fg)
    return box


def _target(draw, x, y, color, radius=36, width=5):
    """Concentric ring + crosshair + solid centre: readable on any background."""
    for r, w in ((radius, width), (radius // 2, max(2, width - 2))):
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=w)
    arm = radius + 16
    for dx, dy in ((1, 0), (0, 1)):
        draw.line((x - arm * dx, y - arm * dy, x + arm * dx, y + arm * dy), fill=color, width=2)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def _dashed_ring(draw, x, y, color, radius=30, width=4, segments=16):
    for i in range(0, segments, 2):
        a0, a1 = 360 * i / segments, 360 * (i + 1) / segments
        draw.arc((x - radius, y - radius, x + radius, y + radius), a0, a1, fill=color, width=width)


def _arrow(draw, p0, p1, color, width=6, head=22):
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for sign in (+1, -1):
        a = ang + sign * math.radians(150)
        draw.line((p1[0], p1[1], p1[0] + head * math.cos(a), p1[1] + head * math.sin(a)),
                  fill=color, width=width)


_SCROLL_VEC = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def annotate(img_bytes: bytes, calls: list[dict[str, Any]]) -> tuple[bytes, list[str]]:
    """
    Draw every action in one assistant response onto the screenshot it acted on.

    Coordinate actions get a target/arrow at their point; coordinate-free ones
    (type, keys, navigation) are anchored to the most recent point in the same
    response, or stacked as banner chips when there is none.
    """
    image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    W, H = image.size
    f_lab, f_num = _font(23), _font(21)
    labels: list[str] = []
    last_pt: tuple[int, int] | None = None
    banner_y = 14
    placed: list[tuple[int, int, int, int]] = []

    for i, call in enumerate(calls, start=1):
        name = call.get("name", "?")
        args = call.get("arguments") or {}
        color = ACTION_COLORS.get(name, NAV_COLOR)
        text = f"{i}. {summarize_call(call)}"
        pt = None

        if isinstance(args.get("point_2d"), list) and len(args["point_2d"]) == 2:
            pt = to_pixel(args["point_2d"], W, H)

        if name == "drag" and isinstance(args.get("start_point_2d"), list):
            p0 = to_pixel(args["start_point_2d"], W, H)
            p1 = to_pixel(args.get("end_point_2d", args["start_point_2d"]), W, H)
            _arrow(draw, p0, p1, color)
            _target(draw, *p0, color=color, radius=24, width=4)
            _dashed_ring(draw, *p1, color=color, radius=24)
            _chip(draw, (p1[0] + 30, p1[1] - 16), text, f_lab, color, placed=placed)
            labels.append(f"{text} — {args['start_point_2d']} → {args.get('end_point_2d')}")
            last_pt = p1
            continue

        if name == "scroll":
            origin = pt or (W // 2, H // 2)
            dx, dy = _SCROLL_VEC.get(str(args.get("direction", "down")).lower(), (0, 1))
            span = 130
            _arrow(draw, (origin[0] - dx * span // 2, origin[1] - dy * span // 2),
                   (origin[0] + dx * span // 2, origin[1] + dy * span // 2), color, width=7)
            _chip(draw, (origin[0] + 26, origin[1] + 26), text, f_lab, color, placed=placed)
            labels.append(text + (f" at {args['point_2d']}" if pt else " (page)"))
            last_pt = origin
            continue

        if pt is not None:
            if name == "hover":
                _dashed_ring(draw, *pt, color=color)
            else:
                _target(draw, *pt, color=color)
            _chip(draw, (pt[0] + 52, pt[1] - 18), text, f_lab, color, placed=placed)
            labels.append(f"{text} — point_2d={args['point_2d']} → px{pt}")
            last_pt = pt
            continue

        # No coordinate of its own.
        if name in POINTLESS and last_pt is not None and name in ("write", "press_keys"):
            _chip(draw, (last_pt[0] + 52, last_pt[1] + 26), text, f_lab, color, placed=placed)
            labels.append(text + " (anchored to previous point)")
        else:
            box = _chip(draw, (14, banner_y), text, f_lab, color, placed=placed)
            banner_y = box[3] + 8
            labels.append(text)

    # Step-order key in the corner so multi-call responses stay readable.
    if len(calls) > 1:
        _chip(draw, (W - 252, H - 48), f"{len(calls)} actions this turn", f_num, (32, 38, 48),
              fg=(230, 237, 243), placed=placed)

    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue(), labels


def to_jpeg_data_uri(img_bytes: bytes, max_width: int, quality: int) -> tuple[str, int]:
    image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    if image.width > max_width:
        h = round(image.height * max_width / image.width)
        image = image.resize((max_width, h), PILImage.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    raw = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), len(raw)


# ---------------------------------------------------------------- prompt


def rebuild_prompt_at_k(text: str, k: int) -> tuple[str, int]:
    """
    Reconstruct the prompt as it would be at --context-num-screenshots=k.

    generate_browser.py prunes by str.replace()ing SCREENSHOT_PLACEHOLDER out of
    user turns outside the last-K window; the placeholder always sits directly
    before the observation close tag, so re-inserting there is exact.
    """
    blocks = _CHAT_BLOCK_RE.findall(text)
    obs_idx = [i for i, (role, c) in enumerate(blocks)
               if role == "user" and OBSERVATION_CLOSE in c]
    keep = set(obs_idx[-k:]) if k > 0 else set()

    out, n_imgs = [], 0
    for i, (role, content) in enumerate(blocks):
        if i in obs_idx:
            has = SCREENSHOT_PLACEHOLDER in content
            if i in keep and not has:
                content = content.replace(
                    OBSERVATION_CLOSE, "\n" + SCREENSHOT_PLACEHOLDER + OBSERVATION_CLOSE, 1)
            elif i not in keep and has:
                content = content.replace(SCREENSHOT_PLACEHOLDER, "")
            if i in keep:
                n_imgs += 1
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(out), n_imgs


def image_tokens(width: int, height: int, patch: int = 16, merge: int = 2) -> int:
    cell = patch * merge
    return (max(1, height // cell)) * (max(1, width // cell))


# ---------------------------------------------------------------- loading


def find_turn_dir(root: str) -> Path:
    """Accept the eval root, the eval_samples dir, or the timestamped leaf."""
    p = Path(root)
    if any(p.glob("*.json")):
        return p
    hits = sorted(p.glob("**/eval_samples/turn/*/"), key=lambda d: d.name)
    if hits:
        return hits[-1]
    hits = sorted(d for d in p.glob("**/turn/*") if d.is_dir() and any(d.glob("*.json")))
    if hits:
        return hits[-1]
    raise SystemExit(f"no turn samples found under {root}")


def load_samples(turn_dir: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted(turn_dir.glob("*.json")):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        d["_file"] = f.name
        out.append(d)
    return out


def build_steps(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Pair each screenshot with the action taken from that state.

    images[i] is the observation at step i; the assistant block that follows it
    in llm_input_texts is the response to it. The final step's response is not
    in llm_input_texts (it is llm_response), so it is appended.
    """
    blocks = parse_chat_blocks(sample.get("llm_input_texts", ""))
    assistants = [b["content"] for b in blocks if b["role"] == "assistant"]
    users = [b["content"] for b in blocks if b["role"] == "user"]
    assistants = assistants + [sample.get("llm_response", "")]

    steps = []
    for i, img in enumerate(sample.get("images", [])):
        resp = assistants[i] if i < len(assistants) else ""
        obs = users[i] if i < len(users) else ""
        tab = _ACTIVE_TAB_RE.search(obs)
        steps.append({
            "index": i,
            "image": img,
            "response": resp,
            "think": (_THINK_RE.search(resp).group(1).strip() if _THINK_RE.search(resp) else ""),
            "calls": parse_tool_calls(resp),
            "url": tab.group(2) if tab else "",
            "feedback": obs.split("<observation>")[0].strip() if "<observation>" in obs else "",
        })
    return steps


def task_text(sample: dict[str, Any]) -> str:
    m = _USER_REQUEST_RE.search(sample.get("llm_input_texts", ""))
    return m.group(1).strip() if m else "(user_request not found)"


def pick_examples(samples, n, rng):
    """
    Favour action variety and a spread of outcomes over the first n on disk:
    a report of ten identical click-only successes teaches nothing.
    """
    scored = []
    for s in samples:
        steps = build_steps(s)
        if not steps:
            continue
        kinds = {c.get("name") for st in steps for c in st["calls"]}
        rare = kinds & {"drag", "scroll", "switch_tab", "go_back", "new_tab", "hover"}
        scored.append((len(kinds) + 2 * len(rare) + rng.random(), s, steps, kinds))
    scored.sort(key=lambda t: -t[0])

    chosen, seen_status = [], {}
    for _, s, steps, kinds in scored:
        st = s.get("status", "?")
        if seen_status.get(st, 0) >= max(2, n // 2):
            continue
        seen_status[st] = seen_status.get(st, 0) + 1
        chosen.append((s, steps, kinds))
        if len(chosen) >= n:
            break
    return chosen


def thin(steps, budget):
    """Keep head and tail when a trajectory is longer than the per-example budget."""
    if len(steps) <= budget:
        return [(s, False) for s in steps]
    head = budget // 2
    tail = budget - head
    out = [(s, False) for s in steps[:head]]
    out.append((steps[head], True))  # marker carries the elision
    out += [(s, False) for s in steps[-tail:]]
    return out


# ---------------------------------------------------------------- html

CSS = """
:root{
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#eceff4; --line:#d5dce5;
  --text:#0f141b; --muted:#59636f; --accent:#e11d48;
  --ok:#187f45; --bad:#c02434; --warn:#a35a05;
  --click:#e11d48; --hover:#0071a4; --scroll:#a35a05; --drag:#7c3aed; --type:#0b7a52; --nav:#2f5fbe;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  --disp:"Archivo","IBM Plex Sans",system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#0c1016; --surface:#141a22; --surface-2:#1c232d; --line:#2a3442;
  --text:#e4ebf2; --muted:#8d99a8; --accent:#ff4d70;
  --ok:#3fb950; --bad:#f85149; --warn:#e3a008;
  --click:#ff4d70; --hover:#3ec8ff; --scroll:#ffb020; --drag:#b884ff; --type:#2edc8c; --nav:#8ab4ff;
}}
:root[data-theme="dark"]{
  --bg:#0c1016; --surface:#141a22; --surface-2:#1c232d; --line:#2a3442;
  --text:#e4ebf2; --muted:#8d99a8; --accent:#ff4d70;
  --ok:#3fb950; --bad:#f85149; --warn:#e3a008;
  --click:#ff4d70; --hover:#3ec8ff; --scroll:#ffb020; --drag:#b884ff; --type:#2edc8c; --nav:#8ab4ff;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto;padding:0 24px 96px}
a{color:var(--accent)}
h1,h2,h3{font-family:var(--disp);font-weight:700;text-wrap:balance;margin:0}
h1{font-size:clamp(30px,4.6vw,52px);letter-spacing:-.02em;line-height:1.03}
h2{font-size:clamp(20px,2.4vw,27px);letter-spacing:-.01em}
h3{font-size:17px;letter-spacing:-.005em}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--muted);max-width:66ch;font-size:16px}

header.hero{padding:56px 0 30px;border-bottom:1px solid var(--line);margin-bottom:34px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
       background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:28px}
.fact{background:var(--surface);padding:12px 14px}
.fact dt{font-family:var(--mono);font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
.fact dd{margin:3px 0 0;font-family:var(--mono);font-size:14px;font-variant-numeric:tabular-nums;word-break:break-all}

section{margin:52px 0 0;scroll-margin-top:20px}
.sec-head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.note{color:var(--muted);max-width:70ch;margin:10px 0 22px}

.legend{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0 0}
.gl{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);background:var(--surface);
    border-radius:999px;padding:6px 13px 6px 9px;font-family:var(--mono);font-size:12px}
.dot{width:11px;height:11px;border-radius:50%;flex:none}
.gl .what{color:var(--muted)}

.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:22px 0}
.card-head{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;
           align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
.task{font-size:16.5px;font-weight:600;max-width:78ch;text-wrap:pretty}
.meta{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:5px;
      display:flex;gap:14px;flex-wrap:wrap;font-variant-numeric:tabular-nums}
.pill{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;
      padding:4px 10px;border-radius:999px;border:1px solid;white-space:nowrap;flex:none}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 42%,transparent);
         background:color-mix(in srgb,var(--ok) 11%,transparent)}
.pill.bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 42%,transparent);
          background:color-mix(in srgb,var(--bad) 11%,transparent)}

.step{border-top:1px solid var(--line);padding:22px 20px}
.step:first-child{border-top:none}
.step-top{display:flex;align-items:center;gap:12px;margin-bottom:13px;flex-wrap:wrap}
.num{font-family:var(--mono);font-size:12px;font-weight:600;background:var(--surface-2);
     border:1px solid var(--line);border-radius:7px;padding:3px 9px;flex:none;font-variant-numeric:tabular-nums}
.url{font-family:var(--mono);font-size:11.5px;color:var(--muted);overflow:hidden;
     text-overflow:ellipsis;white-space:nowrap;max-width:min(100%,62ch)}
.shot{width:100%;display:block;border-radius:9px;border:1px solid var(--line);background:var(--surface-2)}
.acts{display:flex;flex-wrap:wrap;gap:8px;margin:13px 0 0}
.act{font-family:var(--mono);font-size:12px;border-radius:7px;padding:5px 11px;border:1px solid;
     background:var(--surface-2)}
.think{margin-top:13px;color:var(--muted);font-size:14px;max-width:92ch;text-wrap:pretty}
.think b{color:var(--text);font-weight:600}
details.raw{margin-top:12px}
details.raw summary{cursor:pointer;font-family:var(--mono);font-size:11.5px;color:var(--muted);
                    list-style:none;display:inline-flex;gap:6px;align-items:center}
details.raw summary::before{content:"▸";display:inline-block;transition:transform .15s}
details.raw[open] summary::before{transform:rotate(90deg)}
details.raw pre{margin:9px 0 0}
.elide{padding:13px 20px;border-top:1px dashed var(--line);font-family:var(--mono);
       font-size:11.5px;color:var(--muted);text-align:center;background:var(--surface-2)}

pre{font-family:var(--mono);font-size:12.5px;line-height:1.62;background:var(--surface-2);
    border:1px solid var(--line);border-radius:9px;padding:13px 15px;overflow-x:auto;margin:0;
    white-space:pre-wrap;word-break:break-word}

.tapes{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media (max-width:960px){.tapes{grid-template-columns:1fr}}
.tape{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tape-head{padding:14px 17px;border-bottom:1px solid var(--line)}
.tape-head .k{font-family:var(--disp);font-size:18px;font-weight:700}
.tape-head .sub{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:3px;
                font-variant-numeric:tabular-nums}
.tape-body{max-height:660px;overflow-y:auto;padding:14px}
.turn{border:1px solid var(--line);border-radius:9px;margin-bottom:10px;overflow:hidden;background:var(--bg)}
.turn-role{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
           padding:6px 11px;border-bottom:1px solid var(--line);color:var(--muted);background:var(--surface-2)}
.turn-body{padding:10px 12px;font-family:var(--mono);font-size:11.5px;line-height:1.6;
           white-space:pre-wrap;word-break:break-word;max-height:220px;overflow-y:auto}
.ph{background:color-mix(in srgb,var(--accent) 20%,transparent);
    outline:1px solid color-mix(in srgb,var(--accent) 55%,transparent);border-radius:4px;padding:1px 3px}
.ph.added{background:color-mix(in srgb,var(--ok) 22%,transparent);
          outline:1px solid color-mix(in srgb,var(--ok) 60%,transparent)}
.thumb{display:block;width:100%;max-width:330px;margin:7px 0 3px;border-radius:6px;border:1px solid var(--line)}
.cost{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:1px;background:var(--line);
      border-top:1px solid var(--line)}
.cost div{background:var(--surface);padding:10px 13px}
.cost dt{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.cost dd{margin:2px 0 0;font-family:var(--mono);font-size:15px;font-variant-numeric:tabular-nums}
.over{color:var(--bad)}

footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--line);
       font-family:var(--mono);font-size:11.5px;color:var(--muted)}
"""


def esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _render_tape(text: str, k: int, thumbs: dict[int, str], added_from: set[int]) -> str:
    """Render one prompt as a stack of chat turns, with images shown where they sit."""
    blocks = _CHAT_BLOCK_RE.findall(text)
    obs_order = [i for i, (role, c) in enumerate(blocks) if role == "user" and OBSERVATION_CLOSE in c]
    parts = []
    for i, (role, content) in enumerate(blocks):
        img_html = ""
        body = esc(content)
        if SCREENSHOT_PLACEHOLDER in content:
            cls = "ph added" if i in added_from else "ph"
            body = body.replace(esc(SCREENSHOT_PLACEHOLDER),
                                f'<span class="{cls}">{esc(SCREENSHOT_PLACEHOLDER)}</span>')
            j = obs_order.index(i)
            if j in thumbs:
                img_html = f'<img class="thumb" src="{thumbs[j]}" alt="screenshot at this turn">'
        label = role if role != "user" else f"user · observation {obs_order.index(i) + 1}" if i in obs_order else "user"
        parts.append(
            f'<div class="turn"><div class="turn-role">{esc(label)}</div>'
            f'<div class="turn-body">{body}{img_html}</div></div>')
    return "".join(parts)


def _count_tokens(tokenizer, text: str) -> tuple[int, bool]:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text)), True
        except Exception:
            pass
    return round(len(text) / 4.13), False  # measured on these prompts (3643 tok / 15032 ch)


def build_report(args) -> str:
    rng = random.Random(args.seed)
    turn_dir = find_turn_dir(args.eval_dir)
    samples = load_samples(turn_dir)
    if not samples:
        raise SystemExit(f"no turn samples in {turn_dir}")

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as exc:            # report stays useful without it
            print(f"[warn] tokenizer unavailable ({exc}); using char-based estimate")

    chosen = pick_examples(samples, args.num_examples, rng)
    out: list[str] = [f"<title>{esc(args.title)}</title>", f"<style>{CSS}</style>"]
    total_bytes = 0

    # ---- hero -----------------------------------------------------------
    n_ok = sum(1 for s in samples if s.get("status") == "completed")
    steps_all = [int(s.get("total_steps", 0) or 0) for s in samples]
    facts = [
        ("turn samples", f"{len(samples)}"),
        ("shown here", f"{len(chosen)}"),
        ("completed", f"{n_ok} / {len(samples)}"),
        ("mean steps", f"{sum(steps_all)/max(1,len(steps_all)):.1f}"),
        ("screenshots in ctx", f"K = {args.context_num_screenshots}"),
        ("run", esc(args.run_label)),
    ]
    out.append(f"""<div class="wrap"><header class="hero">
<div class="eyebrow">{esc(args.run_label)}</div>
<h1>{esc(args.title)}</h1>
<p class="lede">Every screenshot the policy saw, at the size it saw it, with the action it
took drawn on top. Then the exact prompt behind it, twice: once carrying the
{args.context_num_screenshots} screenshot{'s' if args.context_num_screenshots != 1 else ''} this run
actually kept, once carrying {args.compare_k}.</p>
<dl class="facts">{''.join(f'<div class="fact"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in facts)}</dl>
<div class="legend">""")
    for name, var, what in [("click", "click", "target + crosshair"), ("hover", "hover", "dashed ring"),
                            ("scroll", "scroll", "arrow, direction"), ("drag", "drag", "start → end"),
                            ("write / press_keys", "type", "chip at last point"),
                            ("goto_url / tabs / done", "nav", "banner chip")]:
        out.append(f'<span class="gl"><span class="dot" style="background:var(--{var})"></span>'
                   f'<b>{esc(name)}</b><span class="what">{esc(what)}</span></span>')
    out.append("</div></header>")

    # ---- trajectories ---------------------------------------------------
    out.append('<section><div class="sec-head"><h2>Trajectories</h2>'
               f'<span class="eyebrow">{len(chosen)} examples</span></div>'
               '<p class="note">Each screenshot is the observation the model received at that step; '
               'the overlay is the tool call it emitted in response. Numbered markers give the order '
               'when one response contains several actions.</p>')

    for s, steps, kinds in chosen:
        ok = s.get("status") == "completed"
        head = (f'<div class="card"><div class="card-head"><div>'
                f'<div class="task">{esc(task_text(s))}</div>'
                f'<div class="meta"><span>{esc(s.get("sample_id", "")[:16])}</span>'
                f'<span>{esc(str(s.get("total_steps")))} steps</span>'
                f'<span>{esc(str(s.get("terminate_reason", "")))}</span>'
                f'<span>{len(kinds)} action types</span></div></div>'
                f'<span class="pill {"ok" if ok else "bad"}">{esc(str(s.get("status", "?")))}</span>'
                f'</div>')
        out.append(head)

        for step, elided in thin(steps, args.max_steps_per_example):
            if elided:
                skipped = len(steps) - args.max_steps_per_example
                out.append(f'<div class="elide">… {skipped} intermediate step'
                           f'{"s" if skipped != 1 else ""} not shown …</div>')
                continue
            try:
                raw = _decode_data_url(step["image"])
                png, labels = annotate(raw, step["calls"])
                uri, nbytes = to_jpeg_data_uri(png, args.img_width, args.jpeg_quality)
                total_bytes += nbytes
            except (binascii.Error, OSError, ValueError, IndexError) as exc:
                out.append(f'<div class="step"><div class="step-top"><span class="num">'
                           f'{step["index"] + 1}</span></div><pre>screenshot unavailable: '
                           f'{esc(type(exc).__name__)}</pre></div>')
                continue

            chips = "".join(
                f'<span class="act" style="color:var(--{_var(c.get("name"))});'
                f'border-color:color-mix(in srgb,var(--{_var(c.get("name"))}) 45%,transparent)">'
                f'{i}. {esc(summarize_call(c))}</span>'
                for i, c in enumerate(step["calls"], start=1)) or \
                '<span class="act" style="color:var(--muted)">no parsed tool call</span>'

            think = esc(step["think"])
            if len(think) > args.think_chars:
                think = think[:args.think_chars].rstrip() + " …"

            out.append(f"""<div class="step">
<div class="step-top"><span class="num">step {step['index'] + 1}</span>
<span class="url">{esc(step['url'])}</span></div>
<img class="shot" loading="lazy" src="{uri}" alt="step {step['index'] + 1} screenshot">
<div class="acts">{chips}</div>
{f'<div class="think"><b>think ·</b> {think}</div>' if think else ''}
<details class="raw"><summary>raw response</summary><pre>{esc(step['response'])}</pre></details>
</div>""")
        out.append("</div>")
    out.append("</section>")

    # ---- prompt anatomy -------------------------------------------------
    ps = _pick_prompt_example(chosen, args.compare_k)
    if ps is not None:
        sample, steps, _ = ps
        text = sample["llm_input_texts"]
        blocks_now, n_now = rebuild_prompt_at_k(text, args.context_num_screenshots)
        blocks_k, n_k = rebuild_prompt_at_k(text, args.compare_k)

        obs_idx_k = [i for i, (r, c) in enumerate(_CHAT_BLOCK_RE.findall(blocks_k))
                     if r == "user" and SCREENSHOT_PLACEHOLDER in c]
        obs_idx_now = [i for i, (r, c) in enumerate(_CHAT_BLOCK_RE.findall(blocks_now))
                       if r == "user" and SCREENSHOT_PLACEHOLDER in c]
        added = set(obs_idx_k) - set(obs_idx_now)

        thumbs = {}
        for j, step in enumerate(steps):
            try:
                thumbs[j], nb = to_jpeg_data_uri(_decode_data_url(step["image"]),
                                                 args.thumb_width, args.thumb_quality)
                total_bytes += nb
            except Exception:
                pass

        per_img = image_tokens(args.screen_w, args.screen_h)
        t_now, exact = _count_tokens(tokenizer, blocks_now)
        t_k, _ = _count_tokens(tokenizer, blocks_k)
        rows = []
        for label, txt, n_img, ttext in ((f"K = {args.context_num_screenshots}", blocks_now, n_now, t_now),
                                         (f"K = {args.compare_k}", blocks_k, n_k, t_k)):
            total = ttext + n_img * per_img
            over = total > args.max_context
            rows.append((label, txt, n_img, ttext, total, over))

        out.append(f"""<section><div class="sec-head"><h2>What the model actually reads</h2>
<span class="eyebrow">--context-num-screenshots</span></div>
<p class="note">Same trajectory, same text, one knob apart. <code>generate_browser.py</code> keeps the
screenshot on the last K observation turns and deletes the
<code>{esc(SCREENSHOT_PLACEHOLDER.splitlines()[1])}</code> span from every older one — the text of those
turns survives, the pixels do not. Slots restored at the higher K are highlighted
<span class="ph added">green</span>; the one the run actually kept is
<span class="ph">red</span>. Image cost is
{per_img:,} tokens per {args.screen_w}×{args.screen_h} screenshot
(⌊h/32⌋·⌊w/32⌋ merged patches at patch 16 / merge 2).</p>
<div class="tapes">""")

        for (label, txt, n_img, ttext, total, over) in rows:
            body = _render_tape(txt, n_img, thumbs, added if "= %d" % args.compare_k in label else set())
            out.append(f"""<div class="tape"><div class="tape-head"><div class="k">{esc(label)}</div>
<div class="sub">{n_img} screenshot{'s' if n_img != 1 else ''} in context · {len(txt):,} chars</div></div>
<div class="tape-body">{body}</div>
<dl class="cost">
<div><dt>text tokens</dt><dd>{ttext:,}{'' if exact else ' *'}</dd></div>
<div><dt>image tokens</dt><dd>{n_img * per_img:,}</dd></div>
<div><dt>total</dt><dd class="{'over' if over else ''}">{total:,}</dd></div>
<div><dt>of {args.max_context:,} budget</dt><dd class="{'over' if over else ''}">{100 * total / args.max_context:.0f}%</dd></div>
</dl></div>""")
        out.append("</div>")
        if not exact:
            out.append('<p class="note">* text tokens estimated at 4.13 chars/token; pass '
                       '<code>--tokenizer &lt;model dir&gt;</code> for exact counts.</p>')
        out.append("</section>")

    out.append(f'<footer>generated by openwebrl/visualize_turn_samples.py from '
               f'{esc(str(turn_dir))} · {len(samples)} samples on disk · '
               f'embedded imagery {total_bytes / 1e6:.1f} MB</footer></div>')
    return "\n".join(out)


def _var(name: str | None) -> str:
    return {"click": "click", "hover": "hover", "scroll": "scroll", "drag": "drag",
            "write": "type", "press_keys": "type"}.get(name or "", "nav")


def _pick_prompt_example(chosen, compare_k):
    """
    A trajectory long enough that K actually bites.

    Picking a short one makes the two tapes nearly identical and hides the
    point: the context cost of K only shows up once there are >= K observations
    to keep. Prefer the longest trajectory available, capped at compare_k since
    beyond that the tapes stop diverging.
    """
    if not chosen:
        return None
    ranked = sorted(chosen, key=lambda t: -min(len(t[1]), compare_k))
    return ranked[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True,
                   help="eval root, eval_samples dir, or the timestamped turn dir")
    p.add_argument("--out", default="turn_samples.html")
    p.add_argument("--title", default="Rollout Inspector")
    p.add_argument("--run-label", default="browser eval")
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--max-steps-per-example", type=int, default=6)
    p.add_argument("--img-width", type=int, default=1120, help="stored width; display is full-bleed")
    p.add_argument("--jpeg-quality", type=int, default=68)
    p.add_argument("--thumb-width", type=int, default=300)
    p.add_argument("--thumb-quality", type=int, default=55)
    p.add_argument("--think-chars", type=int, default=520)
    p.add_argument("--context-num-screenshots", type=int, default=1,
                   help="K the run used (run_evaluation.sh defaults to 1)")
    p.add_argument("--compare-k", type=int, default=20, help="K to reconstruct alongside it")
    p.add_argument("--max-context", type=int, default=32768,
                   help="rollout_max_context_len, for the budget bar")
    p.add_argument("--screen-w", type=int, default=1280)
    p.add_argument("--screen-h", type=int, default=1000)
    p.add_argument("--tokenizer", default=None, help="model dir for exact token counts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    report = build_report(args)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}  ({len(report) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
