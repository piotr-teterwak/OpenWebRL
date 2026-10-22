#!/usr/bin/env python3
"""Random-selection control for C2/1A: same turns, random candidate instead of the ARM pick.

Every row of the 1A dataset keeps its state (prompt tokens, images, order); only
selected_index changes, drawn uniformly over that turn's usable candidates. The
ARM's own pick stays in the draw, so ties with the ARM happen at chance rate --
that is the control. Output is laid out the way train_openwebrl_c2.py reads it:
<output>/dataset-audit.json pointing at <output>/training.jsonl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c2', type=Path, required=True, help='C2 run dir holding training.jsonl + dataset-audit.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    source_audit = json.loads((args.c2 / 'dataset-audit.json').read_text())
    source_path = Path(source_audit['dataset'])
    if sha(source_path.read_bytes()) != source_audit['dataset_sha256']: raise ValueError('Source dataset checksum mismatch')
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]

    rng = random.Random(args.seed)
    out_rows, same = [], 0
    for row in rows:
        captured = json.loads(Path(row['source']).read_text())
        usable = [i for i, f in enumerate(captured['finish_types']) if f == 'stop']
        if row['selected_index'] not in usable: raise ValueError(f"ARM pick unusable: {row['source']}")
        pick = rng.choice(usable)
        same += pick == row['selected_index']
        out_rows.append(dict(row, selected_index=pick, arm_selected_index=row['selected_index']))

    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.output / 'training.jsonl'
    dataset.write_text(''.join(json.dumps(r) + '\n' for r in out_rows))
    audit = dict(source_audit, dataset=str(dataset), dataset_sha256=sha(dataset.read_bytes()),
                 ablation='random-candidate', seed=args.seed, source_audit=str(args.c2 / 'dataset-audit.json'),
                 source_dataset_sha256=source_audit['dataset_sha256'],
                 filter=source_audit['filter'] + '; target replaced by a uniform draw over finish_type=stop candidates',
                 index_equals_arm_pick=same, retained_turns=len(out_rows))
    (args.output / 'dataset-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'{len(out_rows)} turns, {same} ({same / len(out_rows):.1%}) index-identical to the ARM pick -> {dataset}')


if __name__ == '__main__':
    main()
