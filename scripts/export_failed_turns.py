#!/usr/bin/env python3
"""Export the executed turns of *valid failed* episodes of an arm_c2 collection, in arm_c2's training.jsonl format.

    export_failed_turns.py --collection <c2-collect/NAME-shardK> --out <c2-collect/NAME-fail-shardK>

arm_c2's export keeps only valid successful episodes (reward 1). With a pre-execution relabeller (e.g. gpt-6-sol
over the stored candidates) a failed episode's states are still labelled states, so this writes them as a
collection root the pair builder reads (prepare_arm_joint_data.py --relabel ... --allow-unsuccessful):
training.jsonl + dataset-audit.json. Same per-turn eligibility as arm_c2 (openwebrl.arm_c2.eligible_turn).
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from openwebrl.arm_c2 import eligible_turn


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--collection', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    config = json.loads((a.collection / 'frozen-config.json').read_text())
    reasons, records, tasks = Counter(), [], set()
    results = sorted((a.collection / 'results').glob('*.json'))
    for p in results:
        result = json.loads(p.read_text())
        if not result.get('valid') or result.get('reward') == 1:
            reasons['not_valid_failed'] += 1; continue
        execution = json.loads((Path(result['attempt']) / 'execution.json').read_text())
        if execution['task_id'] != result['task_id']: raise ValueError('Cross-task execution join')
        for turn in execution['turns']:
            source = Path(turn['source'])
            if sha(source.read_bytes()) != turn['source_sha256']: raise ValueError('Captured turn changed')
            captured = json.loads(source.read_text())
            if captured['task_id'] != result['task_id'] or captured['turn'] != turn['turn'] or captured['attempt'] != result['attempt']:
                raise ValueError('Cross-task or cross-attempt join')
            reason = eligible_turn(captured, turn, config['training']['max_tokens'])
            if reason:
                reasons[reason] += 1; continue
            records.append(dict(task_id=result['task_id'], turn=turn['turn'], source=str(source),
                                source_sha256=turn['source_sha256'], prompt_token_ids=turn['prompt_token_ids'],
                                image_grid_thw=turn['image_grid_thw'], selected_index=captured['selected_index']))
            tasks.add(result['task_id'])
    a.out.mkdir(parents=True, exist_ok=True)
    dataset = a.out / 'training.jsonl'; dataset.write_text(''.join(json.dumps(r) + '\n' for r in records))
    audit = dict(complete_collection=True, source_collection=str(a.collection), scheduled=len(results),
                 failed_tasks_with_usable_turns=len(tasks), successful_tasks_with_usable_turns=0, retained_turns=len(records),
                 exclusions=dict(reasons), dataset=str(dataset), dataset_sha256=sha(dataset.read_bytes()),
                 filter='valid FAILED trajectories (reward != 1) + usable executed turn; for relabelled pairs only')
    (a.out / 'dataset-audit.json').write_text(json.dumps(audit, indent=2) + '\n'); print(json.dumps(audit))


if __name__ == '__main__':
    main()
