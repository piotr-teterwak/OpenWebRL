#!/usr/bin/env python3
"""Derive a data-scaling point from a (matched) joint dataset.

Two knobs, applied to the training split only; validation and the image manifest are copied
unchanged so held-out numbers stay comparable across scales:

  --max-draw D    keep extra teacher draws <state>#drawN with N <= D (D=1: primary draws only).
                  Draws are ranked by the builder's seeded hash, so smaller D is a subset.
  --fraction F    keep the task groups whose seeded hash falls in the lowest F of [0, 1), so
                  F=0.25 is a subset of F=0.5 (nested), and whole tasks are kept together.

As in filter_joint_pairs.py, the last pair in id order is dropped if the final batch for the
two-rank trainer would be odd. Run verify_arm_joint_data.py and data_config afterwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def unit(value):
    return int(hashlib.sha256(('arm-joint-scale:42:' + value).encode()).hexdigest()[:15], 16) / 16 ** 15


def draw_number(row_id):
    _, _, extra = row_id.partition('#draw')
    return int(extra) if extra else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-draw', type=int, default=1)
    ap.add_argument('--fraction', type=float, default=1.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    keep = []
    with (a.src / 'joint_pairs.train.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if draw_number(r['id']) <= a.max_draw and unit(r['task_group']) < a.fraction:
                keep.append((r['id'], line))
    keep.sort()
    dropped = keep.pop()[0] if len(keep) % 32 % 2 else None
    ids = {i for i, _ in keep}
    with (a.src / 'joint_pairs.train.jsonl').open() as src, (a.out / 'joint_pairs.train.jsonl').open('w') as f:
        for line in src:  # keep the source file order
            if json.loads(line)['id'] in ids: f.write(line)
    with (a.src / 'joint_sft.train.jsonl').open() as src, (a.out / 'joint_sft.train.jsonl').open('w') as f:
        for line in src:
            if json.loads(line)['id'] in ids: f.write(line)
    for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl', 'image-manifest.json']:
        shutil.copy2(a.src / name, a.out / name)
    report = dict(source=str(a.src), max_draw=a.max_draw, fraction=a.fraction, kept_train_pairs=len(keep),
                  primary_states=sum('#' not in i for i in ids), dropped_for_even_final_batch=dropped)
    (a.out / 'subsample-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
