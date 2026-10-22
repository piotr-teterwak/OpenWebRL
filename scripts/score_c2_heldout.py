#!/usr/bin/env python3
"""Score a model on held-out C2 states: does it prefer the ARM's pick over the other candidates?

For each held-out state with >1 distinct usable action, compute the log-probability of
every usable candidate response given the exact collected prompt and screenshot (same
reconstruction as training, openwebrl.arm_c2.load_training_example). Compare the ARM's
pick against the candidates whose action differs from it:

  margin_sum    logp(pick) - mean logp(other actions), summed over response tokens
  margin_mean   same with per-token mean log-probs (length-normalised)
  win_rate      share of other-action candidates the pick beats on per-token mean logp

Run it on base and on each student: a student that learned the ARM's preference has a
larger margin than base; one trained on random targets should not.
"""
import argparse
import json
from pathlib import Path
import random
import re


def norm(action): return re.sub(r'\s+', ' ', action.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--states', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.arm_c2 import load_training_example

    rows = [json.loads(line) for line in args.heldout.read_text().splitlines()]
    states = []
    for row in rows:
        captured = json.loads(Path(row['source']).read_text())
        usable = [i for i, f in enumerate(captured['finish_types']) if f == 'stop']
        pick = norm(captured['candidates'][row['selected_index']]['action'])
        others = [i for i in usable if norm(captured['candidates'][i]['action']) != pick]
        if others: states.append((row, others))
    random.Random(args.seed).shuffle(states)
    states = states[:args.states]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16,
                                                        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()

    def logp(row, index):
        example = load_training_example(dict(row, selected_index=index), processor)
        example = {k: v.to('cuda:0') for k, v in example.items()}
        with torch.no_grad():
            logits = model(**{k: v for k, v in example.items() if k != 'labels'}, use_cache=False).logits[0, :-1].float()
        labels = example['labels'][0, 1:]
        mask = labels != -100
        values = torch.log_softmax(logits[mask], -1).gather(-1, labels[mask].unsqueeze(-1)).squeeze(-1)
        return values.sum().item(), values.mean().item()

    out = []
    for n, (row, others) in enumerate(states):
        pick_sum, pick_mean = logp(row, row['selected_index'])
        scored = [logp(row, i) for i in others]
        out.append(dict(task_id=row['task_id'], turn=row['turn'], pick=row['selected_index'], others=others,
                        margin_sum=pick_sum - sum(s for s, _ in scored) / len(scored),
                        margin_mean=pick_mean - sum(m for _, m in scored) / len(scored),
                        win_rate=sum(pick_mean > m for _, m in scored) / len(scored)))
        if n % 50 == 0: print(f'[score] {args.label} {n}/{len(states)}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in out))
    mean = lambda k: sum(r[k] for r in out) / len(out)
    print('SCORE', json.dumps(dict(label=args.label, model=str(args.model), states=len(out), margin_sum=mean('margin_sum'),
                                   margin_mean=mean('margin_mean'), win_rate=mean('win_rate'))), flush=True)


if __name__ == '__main__':
    main()
