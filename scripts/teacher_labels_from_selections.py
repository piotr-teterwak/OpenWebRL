#!/usr/bin/env python3
"""Teacher labels of a DAgger-style collection (c2_collect EXECUTE=first) as a --relabel file.

    teacher_labels_from_selections.py --collection <c2-collect/NAME-shardK> [...] --out labels.jsonl

With EXECUTE=first the actor's own sample (candidate 0) is executed and the teacher's pre-execution pick is
recorded per state as teacher_index. prepare_arm_joint_data.py --relabel takes the chosen side from such a file,
so DAgger pairs are teacher pick vs another candidate at the actor's own states (--allow-unsuccessful for states
of failed episodes). One line per state: {"id": "C2:<task_id>:<turn>", "selected_index", "collection_index", "scores"}.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--collection', type=Path, nargs='+', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--executed', action='store_true', help='selector-steered collection (EXECUTE unset): the executed pick is the teacher label')
    a = ap.parse_args()
    rows, skipped = {}, 0
    for c in a.collection:
        for f in sorted((c / 'selections').glob('*.jsonl')):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if a.executed:
                    if r.get('execute') == 'first': raise ValueError(f'{f}: DAgger-style collection; drop --executed')
                    if r.get('fallback') is not None: skipped += 1; continue
                    r['teacher_index'] = r['selected_index']
                elif r.get('execute') != 'first': raise ValueError(f'{f}: not a DAgger-style (EXECUTE=first) collection')
                if r.get('teacher_index') is None: skipped += 1; continue  # selector fallback: no teacher label
                key = f"C2:{r['task_id']}:{r['turn']}"
                if key in rows: raise ValueError(f'duplicate state {key}')
                rows[key] = dict(id=key, selected_index=r['teacher_index'], collection_index=r['selected_index'], scores=r.get('scores'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in rows.values()))
    agree = sum(r['selected_index'] == r['collection_index'] for r in rows.values())
    print(json.dumps(dict(labels=len(rows), skipped_fallbacks=skipped, teacher_agrees_with_executed=agree / max(1, len(rows)), out=str(a.out))))


if __name__ == '__main__':
    main()
