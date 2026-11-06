#!/usr/bin/env python3
"""Summarize WebArena cells: runs/cells/<site>-<method>[-<tag>]-<job>/task-<id>/summary.json.

    wa_report.py [--runs DIR] [--jobs J ...] [--tags r1 r2 r3] [--no-fuzzy] [--paper]

Per (site, method, tag): cells, valid, invalid by reason, successes, success over valid cells (Wilson 95% CI), mean
steps, episodes ended at step 0, cost; plus a pooled row over sites. Tags r1, r2, ... are repeats: a second table gives
the per-site mean +- SD over repeats. --no-fuzzy drops tasks graded by llm_fuzzy_match (ablation A10). Within one
(site, method, tag, task) the latest valid cell wins (reruns of invalid cells).
"""
import argparse, collections, json, math, re, statistics
from pathlib import Path

TASKS = Path('/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26/viewer/config_files/wa/test_webarena_lite.raw.json')
PAPER = {  # WebArbiter Table 5, GPT-4o policy: Shopping, CMS, Reddit, GitLab, Map
    'mini_prm': dict(shopping=26.67, shopping_admin=37.14, reddit=42.11, gitlab=40.00, map=19.23),
    'arbiter': dict(shopping=44.44, shopping_admin=42.86, reddit=52.63, gitlab=56.67, map=38.46),
    'no_search': dict(shopping=23.91, shopping_admin=31.43, reddit=28.57, gitlab=56.25, map=19.35),  # imported (WebShepherd)
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fuzzy_tasks():
    out = set()
    for t in json.loads(TASKS.read_text()):
        if 'fuzzy_match' in json.dumps(t['eval'].get('reference_answers') or {}) or \
           any('fuzzy_match' in json.dumps(x) for x in (t['eval'].get('program_html') or [])):
            out.add(t['task_id'])
    return out


def unclosed_fence(cell):
    """A policy reply that opened ```json without closing it reached the released parser before the 2026-11-02 fix
    (wa_protocol.JSON_FENCE): that step had no candidates, so the cell is a harness artefact and is rerun."""
    calls = cell / 'calls.jsonl'
    for line in (calls.open() if calls.exists() else []):
        if '"role": "policy"' in line:
            r = json.loads(line)
            if (r.get('reply') or '').lstrip().startswith('```') and r.get('unwrapped_fence') in (False, 'False'):
                return True
    return False


def row(label, rows):
    v = [r for r in rows if r['valid']]; k = sum(r['success'] for r in v); lo, hi = wilson(k, len(v))
    steps = sum(r['steps'] for r in v) / len(v) if v else float('nan')
    stop0 = sum(r['steps'] <= 1 and r['status'] == 'terminal' for r in v)
    inv = collections.Counter((r.get('invalid_reason') or 'unknown') for r in rows if not r['valid'])
    print(f"{label:44}{len(rows):>6}{len(v):>6}{k:>5}{(100 * k / len(v) if v else float('nan')):>6.1f}%"
          f"{f'[{100 * lo:.0f}, {100 * hi:.0f}]':>12}{steps:>7.1f}{stop0:>7}{sum(r['model_usd'] for r in rows):>8.2f}  {dict(inv)}")
    return (100 * k / len(v)) if v else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, default=Path('/gpfs/scrubbed/piotrt/owrl/webarena/runs/cells'))
    ap.add_argument('--jobs', nargs='*')
    ap.add_argument('--tags', nargs='*')
    ap.add_argument('--no-fuzzy', action='store_true')
    a = ap.parse_args()
    skip = fuzzy_tasks() if a.no_fuzzy else set()
    cells = {}
    for p in sorted(a.runs.glob('*/task-*/summary.json')):
        parts = p.parent.parent.name.split('-'); job = parts[-1]; tag = '-'.join(parts[2:-1])
        if (a.jobs and job not in a.jobs) or (a.tags is not None and tag not in a.tags):
            continue
        s = json.loads(p.read_text()); s['job'] = job; s['tag'] = tag
        if s['valid'] and unclosed_fence(p.parent):
            s.update(valid=False, invalid_reason='unclosed_fence')
        if s['task_id'] in skip:
            continue
        key = (s['site'], s['method'], tag, s['task_id'])
        if key not in cells or (s['valid'] and not cells[key]['valid']) or (s['valid'] == cells[key]['valid'] and job > cells[key]['job']):
            cells[key] = s
    groups = collections.defaultdict(list)
    for (site, method, tag, _), s in cells.items():
        groups[(site, method, tag)].append(s)
    print(f"{'site / method / tag':44}{'cells':>6}{'valid':>6}{'succ':>5}{'rate':>7}{'95% CI':>12}{'steps':>7}{'stop@0':>7}{'USD':>8}  invalid")
    rates = {}
    for (site, method, tag), rows in sorted(groups.items()):
        rates[(site, method, tag)] = row(f'{site} / {method} / {tag or "-"}', rows)
    by_mt = collections.defaultdict(list)
    for (site, method, tag), rows in groups.items():
        by_mt[(method, tag)] += rows
    for (method, tag), rows in sorted(by_mt.items()):
        if len({r['site'] for r in rows}) > 1:
            row(f'POOLED / {method} / {tag or "-"}', rows)
    reps = collections.defaultdict(list)
    for (site, method, tag), r in rates.items():
        m = re.fullmatch(r'(.*?)-?r(\d+)', tag)
        if m:
            reps[(site, method, m.group(1))].append(r)
    if reps:
        print(f"\n{'repeats: site / method / config':44}{'n':>4}{'mean':>8}{'SD':>7}{'paper':>8}")
        for (site, method, cfg), rs in sorted(reps.items()):
            sd = statistics.stdev(rs) if len(rs) > 1 else float('nan')
            label = f'{site} / {method} / ' + (cfg or 'main')
            paper = PAPER.get(method, {}).get(site, float('nan'))
            print(f"{label:44}{len(rs):>4}{statistics.mean(rs):>7.1f}%{sd:>7.1f}{paper:>7.1f}%")


if __name__ == '__main__':
    main()
