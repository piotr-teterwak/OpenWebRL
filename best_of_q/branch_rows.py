#!/usr/bin/env python3
"""Best-of-Q training rows from Zixian's branch states (selection-head common SFT-action cohort, 2026-10-23).

    branch_rows.py --out DIR [--seed 0] [--fold 0 | --fold -1 for both folds]

Source: /gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021/sft-head-common/panels.jsonl,
one row per state: the saved actor prompt and screenshot, 2-5 distinct executed candidate actions, and per candidate the
recorded continuation outcomes on fit draws 0-2 (recorded / valid / successes). Fold 0 = the 1,367 training states her
heads were trained on (347 task groups); fold 4 = her 342 held-out states (85 groups), which are never used here.

One row per (state, candidate) in embed.py's format (task, url, history, screenshot path, action) plus `target`: ONE
continuation outcome per candidate, drawn uniformly from its valid recorded fit draws (equivalently Bernoulli of its
valid success rate), seeded. Candidates with no valid fit draw are dropped. Also writes `pooled` (the fit-draw success
rate her observed-outcome head used) for reference. Stdlib only.
"""
import argparse
import json
import random
import re
from pathlib import Path

PANELS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021/sft-head-common/panels.jsonl')
TOOL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def parse_prompt(prompt):
    task = re.search(r'<user_request>\s*(.*?)\s*</user_request>', prompt, re.S)
    turns = prompt.split('<|im_start|>assistant')
    # every completed assistant turn before the open one holds an executed tool call
    history = []
    for t in turns[1:-1]:
        m = TOOL.search(t)
        if m:
            history.append(f'<tool_call>\n{m.group(1)}\n</tool_call>')
    urls = re.findall(r'\(active\):\s*(\S+)', prompt)
    return (task.group(1) if task else ''), (urls[-1] if urls else ''), history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fold', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    n_states = n_rows = dropped = 0; pos = 0
    with (a.out / 'rows.jsonl').open('w') as f:
        for line in PANELS.open():
            p = json.loads(line)
            if a.fold >= 0 and p['fold'] != a.fold:
                continue
            task, url, history = parse_prompt(p['action_only_prompt'])
            n_states += 1
            for k, (action, fit) in enumerate(zip(p['actions'], p['fit'])):
                if not fit['valid']:
                    dropped += 1; continue
                target = 1.0 if rng.random() < fit['successes'] / fit['valid'] else 0.0
                act = action.replace('<|im_end|>', '').strip()
                f.write(json.dumps(dict(episode=p['state_id'], state_id=p['state_id'], task_id=p['task_id'], task=task,
                                        url=url, screenshot=p['image_path'], history=history, action=act, candidates=[act],
                                        candidate_index=k, target=target, pooled=fit['successes'] / fit['valid'],
                                        valid_draws=fit['valid'], fit_valid=fit['valid'], fit_successes=fit['successes'],
                                        holdout_rate=(p['holdout'][k]['successes'] / p['holdout'][k]['valid']) if p['holdout'][k]['valid'] else None,
                                        fold=p['fold'], actor_first=(k == p['actor_first_action_index']))) + '\n')
                n_rows += 1; pos += target
    summary = dict(states=n_states, rows=n_rows, dropped_no_valid_draw=dropped, positive_targets=pos, seed=a.seed, fold=a.fold)
    (a.out / 'build.json').write_text(json.dumps(summary, indent=1) + '\n'); print(json.dumps(summary))


if __name__ == '__main__':
    main()
