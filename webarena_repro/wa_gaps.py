#!/usr/bin/env python3
"""List the cells a set of run_cells.sbatch jobs planned but did not produce a valid result for, and print rerun commands.

    wa_gaps.py JOB [JOB ...]

Planned tasks come from each job's 'node ... site S method M tasks: ...' line; a cell counts as done if any run with the
same (site, method, tag) has a valid summary for it (so earlier reruns are honoured). Missing cells (no summary.json:
SITE_NOT_READY, COPY_FAILED, job died) would otherwise silently shrink N in wa_report.py.
"""
import collections, glob, json, re, sys
from pathlib import Path

from wa_report import unclosed_fence

W = Path('/gpfs/scrubbed/piotrt/owrl/webarena')


def main():
    planned, running = {}, set()
    for job in sys.argv[1:]:
        outs = glob.glob(str(W / f'jobs/cells-*-{job}.out'))
        runs = glob.glob(str(W / f'runs/cells/*-{job}'))
        if not outs or not runs:
            print(f'# {job}: no output or run dir yet'); continue
        m = re.search(r'site (\S+) method (\S+) tasks: (.*)', Path(outs[0]).read_text())
        if not m:
            print(f'# {job}: no task line yet'); continue
        site, method, tasks = m.group(1), m.group(2), [int(t) for t in m.group(3).split()]
        site = {'cms': 'shopping_admin'}.get(site, site)  # job logs use the hosting name, summaries the WebArena one
        tag = '-'.join(Path(runs[0]).name.split('-')[2:-1])
        if 'CELLS_DONE' not in Path(outs[0]).read_text():
            running.add(job)
        for t in tasks:
            planned[(site, method, tag, t)] = job
    done, why = set(), {}
    for p in glob.glob(str(W / 'runs/cells/*/task-*/summary.json')):
        s = json.loads(Path(p).read_text()); tag = '-'.join(Path(p).parent.parent.name.split('-')[2:-1])
        key = (s['site'], s['method'], tag, s['task_id'])
        if s['valid'] and unclosed_fence(Path(p).parent):
            s.update(valid=False, invalid_reason='unclosed_fence')
        if s['valid']:
            done.add(key)
        else:
            why.setdefault(key, s.get('invalid_reason') or 'invalid')
    gaps = collections.defaultdict(list)
    for key, job in sorted(planned.items()):
        if key not in done and job not in running:
            gaps[key[:3]].append((key[3], why.get(key, 'missing')))
    if running:
        print(f'# still running (not counted as gaps): {" ".join(sorted(running))}')
    for (site, method, tag), cells in sorted(gaps.items()):
        print(f'# {site} {method} {tag}: ' + ', '.join(f'{t} ({r})' for t, r in cells))
        print(f'SITE={site if site != "shopping_admin" else "cms"} METHOD={method} TAG={tag} TASKS="{" ".join(str(t) for t, _ in cells)}" '
              f'sbatch --exclude=g011,g013,g015,g021 {W}/jobs/run_cells.sbatch')


if __name__ == '__main__':
    main()
