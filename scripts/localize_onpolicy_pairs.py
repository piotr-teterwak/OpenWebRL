#!/usr/bin/env python3
"""Make Dan's on-policy ARM pairs (HF davidan03/openwebrl-arm-onpolicy-pairs) trainable here.

The release holds only joint_pairs.train.jsonl (5,206 pairs: C2 + C2p2, all from the actor's own
rollouts, labelled by the local SelectionARM) with relative image paths, plus screenshots. This
writes a dataset directory our trainer accepts without changing a single training row's content
other than image paths:

  joint_pairs.train.jsonl   the release rows, images[*].path made absolute: an existing tillicum
                            copy with the same sha256 (C2 screenshots), else the downloaded file;
                            every file's sha256 is checked against the row
  joint_sft.train.jsonl     each pair row minus rejected/rejected_index (the chosen-only view the
                            RPO SFT term and verify_arm_joint_data.py expect); still on-policy
  validation files          copied from the joint data (--val-src): held-out metrics only, never
                            trained on; the release says its panels are unchanged from these

Then run verify_arm_joint_data.py and train_arm_joint_sft.data_config on --out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, required=True, help='downloaded release directory')
    ap.add_argument('--val-src', type=Path, required=True, help='joint data dir with the validation split (e.g. joint-data/matched)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    local = {m['sha256']: m['path'] for m in json.loads((a.val_src / 'image-manifest.json').read_text())}
    checked, counts = {}, {'local_copy': 0, 'downloaded': 0}
    with (a.src / 'joint_pairs.train.jsonl').open() as src, \
         (a.out / 'joint_pairs.train.jsonl').open('w') as pairs, (a.out / 'joint_sft.train.jsonl').open('w') as sft:
        for line in src:
            row = json.loads(line)
            assert row['label_source'].startswith('SelectionARM'), row['id']  # on-policy ARM labels only
            # 277 C2p2 rows carry split=validation from the release's own pass-2 split, yet are in its
            # training file (Dan trained on them). Their task groups are not in the joint-data
            # validation split, so relabelling them train keeps his training set exactly.
            if row['split'] != 'train':
                counts['relabelled_split_train'] = counts.get('relabelled_split_train', 0) + 1
                row['split'] = 'train'
            for im in row['images']:
                want = im['sha256']
                if want not in checked:
                    candidate = local.get(want)
                    if candidate and Path(candidate).exists() and sha256(candidate) == want:
                        checked[want] = candidate; counts['local_copy'] += 1
                    else:
                        path = a.src / im['path']
                        assert path.exists() and sha256(path) == want, f'missing or wrong image {im["path"]}'
                        checked[want] = str(path.resolve()); counts['downloaded'] += 1
                im['path'] = checked[want]
            pairs.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            sft.write(json.dumps({k: v for k, v in row.items() if k not in ('rejected', 'rejected_index')},
                                 ensure_ascii=False, separators=(',', ':')) + '\n')
    for name in ['joint_pairs.validation.jsonl', 'joint_sft.validation.jsonl', 'image-manifest.json']:
        shutil.copy2(a.val_src / name, a.out / name)
    report = dict(source='HF davidan03/openwebrl-arm-onpolicy-pairs', source_train_sha256=sha256(a.src / 'joint_pairs.train.jsonl'),
                  validation_from=str(a.val_src), unique_images=len(checked), **counts)
    (a.out / 'localize-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
