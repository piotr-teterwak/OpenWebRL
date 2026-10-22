#!/usr/bin/env python3
"""Derive a joint dataset whose training pairs have equal action counts in chosen and rejected.

In joint-data-v2, when the two responses differ in how many tool calls they make, the
chosen one has more 74% of the time; DPO turns that into "keep acting" and, at lr 5e-5,
into responses that never end their turn. Keeping only same-count pairs removes that
signal while every kept pair still differs in which action is taken.

Training pairs and their SFT twins are filtered together; validation and the image
manifest are copied unchanged, so held-out numbers stay comparable with v2. If the kept
count would leave an odd final batch for the two-rank trainer, the last pair in id order
is dropped. Run verify_arm_joint_data.py and train_arm_joint_sft.data_config afterwards.
"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pairs = [json.loads(line) for line in (args.src / 'joint_pairs.train.jsonl').open()]
    keep = sorted((r for r in pairs if len(r['chosen']['actions']) == len(r['rejected']['actions'])), key=lambda r: r['id'])
    dropped_for_batch = None
    if len(keep) % 32 % 2:
        dropped_for_batch = keep.pop()['id']
    ids = {r['id'] for r in keep}
    order = [r['id'] for r in pairs if r['id'] in ids]  # keep the source file order
    by_id = {r['id']: r for r in keep}
    with (args.out / 'joint_pairs.train.jsonl').open('w') as f:
        for i in order: f.write(json.dumps(by_id[i], ensure_ascii=False, separators=(',', ':')) + '\n')
    with (args.src / 'joint_sft.train.jsonl').open() as src, (args.out / 'joint_sft.train.jsonl').open('w') as f:
        for line in src:
            if json.loads(line)['id'] in ids: f.write(line)
    for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl', 'image-manifest.json']:
        shutil.copy2(args.src / name, args.out / name)
    report = dict(source=str(args.src), filter='same action count in chosen and rejected (train only)',
                  source_train_pairs=len(pairs), kept_train_pairs=len(keep), dropped_for_even_final_batch=dropped_for_batch)
    (args.out / 'filter-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
