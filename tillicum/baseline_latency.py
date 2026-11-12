#!/usr/bin/env python3
"""Model seconds per episode for the external selectors on OpenWebRL-4B-SFT, in Zixian's convention
(selection-recipe3-vs-cost-20261104-analysis.py / selection-latency-head-vs-decoder-20261111.py): per-step
decision_seconds = actor generation (5 candidates) + selection; summed over the episode's steps; browser time excluded.

Our harness did not record per-step generation timing, so for Best-of-Q / PRO-CUA / WebDreamer this is an estimate:
steps per episode (selector calls / episodes) x (Zixian's measured 5-candidate SFT 4B generation, 7.957 s/step, 1 H200,
eval-1111 decoder run) + our measured selector seconds per call (server-side, *-selector-calls.jsonl). SFT 1-sample and
SelectionARM (ADM) rows are Zixian's measured model_seconds_per_episode (sft1 / piotr5 in selection-recipe3-vs-cost-20261104).
Caveat: selectors that share the actor's GPU (PRO-CUA VLM, Dreamer-7B) also slow the actor, which this estimate omits.
    python3 tillicum/baseline_latency.py
"""
import glob, json, statistics as st
from pathlib import Path
E = Path('/gpfs/scrubbed/piotrt/owrl/eval')
GEN5 = 7.957
RUNS = {'Best-of-Q': ['passboqbranchsim1/base+boq'], 'PRO-CUA PRM': ['passprocua-t1/base+procuasel'],
        'WebDreamer n=1': ['passwebdreamer-n1/base+webdreamer', 'passwebdreamer-n1/base+webdreamer-r0-297', 'passwebdreamer-n1/base+webdreamer-r298-594'],
        'WebDreamer n=10': ['passwebdreamer2/base+webdreamer', 'passwebdreamer2/base+webdreamer-r0-297', 'passwebdreamer2/base+webdreamer-r298-594']}
ZIXIAN = {'SFT (1 sample)': (35.8, 30.2, 48.0), 'our ADM (SelectionARM)': (116.8, 106.1, 186.7)}
B = ('online-mind2web', 'webvoyager', 'deepshop')
for name, v in ZIXIAN.items(): print(f'{name:22} measured  ' + '  '.join(f'{x:6.1f}' for x in v))
for name, dirs in RUNS.items():
    row = []
    for b in B:
        secs, eps = [], 0
        for d in dirs:
            c = E / d / f'{b}-selector-calls.jsonl'
            if c.exists():
                secs += [json.loads(l).get('seconds', 0) for l in open(c)]; eps += len(glob.glob(str(E / d / b / 'results' / '*.json')))
        row.append(len(secs) / eps * (GEN5 + st.mean(secs)) if secs else float('nan'))
    print(f'{name:22} estimated ' + '  '.join(f'{x:6.1f}' for x in row))
