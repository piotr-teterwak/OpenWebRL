#!/usr/bin/env python3
"""Plan reruns of invalid episodes for the external-selector baselines (Best-of-Q, PRO-CUA PRM, Dreamer-7B n=1),
3 runs x OM2W / WebVoyager / DeepShop, excluding Wolfram Alpha tasks (its site blocks the cluster proxy; rerun later).

    python3 tillicum/redo_baselines.py > plan.tsv     # baseline  run  bench  pass  n  indices

Each run's current state is its original directories with later redo directories overriding per task (as the
earlier Dreamer timeout reruns did); a task is planned if it is invalid there. Reruns go to pass<orig>-redo2 with the
same harness, seeds and sampling; only the step (180 -> 600 s) and task (1800 -> 3600 s) timeouts are raised.
"""
import glob
import sys
import json
from pathlib import Path

E = Path('/gpfs/scrubbed/piotrt/owrl/eval'); REPO = Path('/gpfs/home/piotrt/OpenWebRL')
FILES = {'online-mind2web': 'openwebrl/data/eval/online-mind2web.jsonl', 'webvoyager': 'openwebrl/data/eval/webvoyager_fara.jsonl',
         'deepshop': 'openwebrl/data/eval/deepshop.jsonl'}
RUNS = {  # baseline -> per run: (pass to write the redo under, [dirs in override order])
    'boq': [('boqbranchsim1', ['passboqbranchsim1/base+boq']), ('boqbranchsim1-run2', ['passboqbranchsim1-run2/base+boq']),
            ('boqbranchsim1-run3', ['passboqbranchsim1-run3/base+boq'])],
    'procua': [('procua-t1', ['passprocua-t1/base+procuasel']), ('procua-t1-run2', ['passprocua-t1-run2/base+procuasel']),
               ('procua-t1-run3', ['passprocua-t1-run3/base+procuasel'])],
    'webdreamer': [('webdreamer-n1', ['passwebdreamer-n1/base+webdreamer', 'passwebdreamer-n1/base+webdreamer-r0-297',
                                      'passwebdreamer-n1/base+webdreamer-r298-594', 'passwebdreamer-n1-redo/base+webdreamer']),
                   ('webdreamer-n1-run2', ['passwebdreamer-n1-run2/base+webdreamer']),
                   ('webdreamer-n1-run3', ['passwebdreamer-n1-run3/base+webdreamer', 'passwebdreamer-n1-run3-redo/base+webdreamer'])],
}
WOLFRAM = '--wolfram' in sys.argv  # plan only the Wolfram Alpha tasks (once the site answers the cluster again)
LABEL = {'boq': 'base+boq', 'procua': 'base+procuasel', 'webdreamer': 'base+webdreamer'}


def task_id(row):
    return str(row.get('task_id') or (row.get('metadata') or {}).get('task_id') or row.get('id'))


def main():
    index = {b: {task_id(json.loads(l)): i for i, l in enumerate((REPO / f).read_text().splitlines())} for b, f in FILES.items()}
    for base, runs in RUNS.items():
        for ps, dirs in runs:
            for b in FILES:
                state = {}
                for d in dirs + [f'pass{ps}-redo2/{LABEL[base]}']:  # an earlier redo2 attempt counts too
                    for f in glob.glob(str(E / d / b / 'results' / '*.json')):
                        r = json.loads(Path(f).read_text()); state[str(r['task_id'])] = r
                keep = (lambda t: t.startswith('Wolfram')) if WOLFRAM else (lambda t: not t.startswith('Wolfram'))
                bad = sorted(index[b][t] for t, r in state.items() if not r.get('valid') and keep(t))
                missing = sorted(i for t, i in index[b].items() if t not in state and keep(t))
                todo = sorted(set(bad) | set(missing))
                if todo:
                    print(f"{base}\t{ps}\t{b}\tpass{ps}-redo2\t{len(todo)}\t{','.join(map(str, todo))}")


if __name__ == '__main__':
    main()
