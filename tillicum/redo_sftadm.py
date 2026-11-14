#!/usr/bin/env python3
"""Plan reruns of invalid non-Wolfram episodes of Zixian's SFT (head-k5-20261101 sft1) and SelectionARM
(controls-sameday-20261103 piotr5) runs, 3 runs x 3 benchmarks, for tillicum/eval_redo_sftadm.sbatch (our arm_eval
harness, her sampling: T 1.0 / top-p 0.9 / 1,024 tokens / 30 steps / seed 42, 8 browsers, step 600 s / task 3600 s).
A task already redone validly under pass<zx-...>-redo2 is not planned again.
    python3 tillicum/redo_sftadm.py      # kind  pass  bench  n  indices
"""
import glob, json, sys
WOLFRAM = '--wolfram' in sys.argv  # plan only the Wolfram Alpha tasks
from pathlib import Path
Z = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'); E = Path('/gpfs/scrubbed/piotrt/owrl/eval')
REPO = Path('/gpfs/home/piotrt/OpenWebRL')
FILES = {'online_mind2web': 'openwebrl/data/eval/online-mind2web.jsonl', 'webvoyager': 'openwebrl/data/eval/webvoyager_fara.jsonl',
         'deepshop': 'openwebrl/data/eval/deepshop.jsonl'}
RUNS = {'sft': ('head-k5-20261101', 'sft1', ['run1', 'rerun2', 'rerun3']), 'adm': ('controls-sameday-20261103', 'piotr5', ['run1', 'run2', 'run3'])}
def tid(row): return str(row.get('task_id') or (row.get('metadata') or {}).get('task_id') or row.get('id'))
def valid(r): return str(r.get('valid')) == 'True'
for kind, (root, cond, runs) in RUNS.items():
    for run in runs:
        for b, f in FILES.items():
            index = {tid(json.loads(l)): i for i, l in enumerate((REPO / f).read_text().splitlines())}
            state = {}
            for p in glob.glob(str(Z / root / run / b / cond / b / 'collection' / 'records' / '*.json')):
                r = json.loads(Path(p).read_text()); state[str(r['task_id'])] = valid(r)
            ps = f'zx-{cond}-{run}-redo2'
            for p in glob.glob(str(E / f'pass{ps}' / cond / b.replace('_', '-') / 'results' / '*.json')):
                r = json.loads(Path(p).read_text()); state[str(r['task_id'])] = state.get(str(r['task_id'])) or bool(r.get('valid'))
            todo = sorted(index[t] for t in index if t.startswith('Wolfram') == WOLFRAM and not state.get(t))
            unknown = [t for t in state if t not in index]
            if unknown: print(f'# {kind} {run} {b}: {len(unknown)} record task ids not in task file, e.g. {unknown[:3]}')
            if todo: print(f"{kind}\t{ps}\t{b.replace('_', '-')}\t{len(todo)}\t{','.join(map(str, todo))}")
