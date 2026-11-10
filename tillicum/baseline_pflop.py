#!/usr/bin/env python3
"""Parameters, PFLOP per episode and model latency for the external selectors on OpenWebRL-4B-SFT, with Zixian's
accounting (selection-recipe3-vs-cost-20261104-analysis.py): PFLOP = 2 x params x (uncached prompt + generated
tokens), summed over every model in the system.

    python3 tillicum/baseline_pflop.py

Per job (one benchmark, first run): uncached prompt tokens = the sum of `#new-token` over the SGLang server's
`Prefill batch` lines (prefix-cache hits excluded, as cached_tokens are in Zixian's actor meta); generated tokens =
the stored outputs re-tokenized (actor: the 5 candidates' thought + action per step; PRO-CUA PRM: its judgments,
stored truncated to 1,500 characters, so a slight undercount) or the servers' own usage (Dreamer-7B and value
completions). Best-of-Q's embedder runs in transformers (no prefill log): per step it embeds the state (screenshot
at the Qwen2.5-VL default resolution, 1,196 image tokens for 1280x720, plus the task / URL / history text) and the
5 candidate texts; the text part is tokenized, the image part counted from the resolution.
PFLOP / episode = PFLOP / selection step x selection steps per episode (selector calls / episodes run; timed-out
episodes record no total_steps, so the call count is the measure that includes them).
Model latency / episode = (actor time + selector time) per step x steps: selector time is measured per call; actor
time per step is Zixian's measured SFT 4B value for 5 parallel candidates (her SelectionARM run, model seconds per
step minus nothing for the actor share is not separable, so we use her single-sample SFT time x 1, see the note).
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

E = Path('/gpfs/scrubbed/piotrt/owrl/eval')
P = dict(actor=4.44e9, vl4b=4.44e9, dreamer=8.29e9, qwen25vl3b=3.75e9)
BENCH = ('online-mind2web', 'webvoyager', 'deepshop')
RUNS = {  # selector -> list of run dirs (first run; WebDreamer n=10 WebVoyager is two shards)
    'Best-of-Q': ['passboqbranchsim1/base+boq'],
    'PRO-CUA PRM': ['passprocua-t1/base+procuasel'],
    'WebDreamer n=1': ['passwebdreamer-n1/base+webdreamer', 'passwebdreamer-n1/base+webdreamer-r0-297', 'passwebdreamer-n1/base+webdreamer-r298-594'],
    'WebDreamer n=10': ['passwebdreamer2/base+webdreamer', 'passwebdreamer2/base+webdreamer-r0-297', 'passwebdreamer2/base+webdreamer-r298-594'],
}
NEW = re.compile(r'Prefill batch.*?#new-token: (\d+)')
tok = AutoTokenizer.from_pretrained('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT', local_files_only=True)
ntok = lambda texts: sum(len(x) for x in tok(list(texts), add_special_tokens=False)['input_ids']) if texts else 0


def prefill(log):
    return sum(int(m.group(1)) for line in open(log, errors='ignore') if 'Prefill batch' in line for m in [NEW.search(line)] if m) if Path(log).exists() else 0


def main():
    out = {}
    for sel, dirs in RUNS.items():
        for b in BENCH:
            fl = {'actor': 0.0, 'selector': 0.0}; steps_sel = 0; sel_seconds = 0.0; ep_steps = []
            for d in dirs:
                d = E / d
                if not (d / b).is_dir():
                    continue
                calls = [json.loads(l) for l in open(d / f'{b}-selector-calls.jsonl')] if (d / f'{b}-selector-calls.jsonl').exists() else []
                steps_sel += len(calls); sel_seconds += sum(c.get('seconds', 0) for c in calls)
                gen = []
                for f in glob.glob(str(d / b / 'selections' / '*.jsonl')):
                    for line in open(f):
                        r = json.loads(line); gen += [(c.get('thought') or '') + (c.get('action') or '') for c in r.get('candidates', [])]
                fl['actor'] += 2 * P['actor'] * (prefill(d / f'actor-server-{b}.log') + ntok(gen))
                if sel == 'PRO-CUA PRM':
                    fl['selector'] += 2 * P['vl4b'] * (prefill(d / f'vlm-server-{b}.log') + ntok([x for c in calls for x in c.get('replies', [])]))
                elif sel.startswith('WebDreamer'):
                    dc = sum((s['usage']['dreamer'] or {}).get('completion_tokens', 0) for c in calls for s in c['sims'])
                    vc = sum((s['usage']['value'] or {}).get('completion_tokens', 0) for c in calls for s in c['sims'])
                    fl['selector'] += 2 * P['dreamer'] * (prefill(d / f'dreamer-server-{b}.log') + dc) + 2 * P['vl4b'] * (prefill(d / f'vlm-server-{b}.log') + vc)
                elif sel == 'Best-of-Q':
                    # state: image tokens + text per step; actions: the 5 candidate texts (the server embeds thought + action)
                    texts = []
                    for f in glob.glob(str(d / b / 'selections' / '*.jsonl')):
                        for line in open(f):
                            r = json.loads(line); texts += [(c.get('thought') or '') + '\n' + (c.get('action') or '') for c in r.get('candidates', [])]
                    state_text = 300  # task + URL + history framing, a typical length
                    fl['selector'] += 2 * P['qwen25vl3b'] * (len(calls) * (1196 + state_text) + ntok(texts))
                for f in glob.glob(str(d / b / 'results' / '*.json')):
                    ep_steps.append(json.load(open(f)).get('total_steps') or 0)
            if not steps_sel:
                continue
            spe = steps_sel / max(1, len(ep_steps))
            per_step = (fl['actor'] + fl['selector']) / steps_sel / 1e15
            out[(sel, b)] = dict(steps=spe, pflop_step=per_step, pflop_ep=per_step * spe, actor_share=fl['actor'] / (fl['actor'] + fl['selector']),
                                 sel_s_step=sel_seconds / steps_sel)
            print(f'{sel:16} {b:16} steps/ep {spe:5.1f} | PFLOP/step {per_step:.4f} (actor {100 * out[(sel, b)]["actor_share"]:.0f}%) '
                  f'| PFLOP/episode {per_step * spe:.3f} | selector s/step {sel_seconds / steps_sel:.1f}')
    print('\nmeans over the three benchmarks')
    for sel in RUNS:
        rows = [out[(sel, b)] for b in BENCH if (sel, b) in out]
        if rows:
            print(f'{sel:16} PFLOP/episode {st.mean(r["pflop_ep"] for r in rows):.3f} | steps/ep {st.mean(r["steps"] for r in rows):.1f} '
                  f'| selector s/step {st.mean(r["sel_s_step"] for r in rows):.1f} | selector s/episode {st.mean(r["sel_s_step"] * r["steps"] for r in rows):.0f}')


if __name__ == '__main__':
    main()
