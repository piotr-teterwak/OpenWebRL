#!/usr/bin/env python3
"""Best-of-Q training data from our collections: one row per executed step of every valid episode.

    extract_transitions.py OUT.jsonl COLLECTION_DIR [COLLECTION_DIR ...]

Row: episode (attempt path), collection, task_id, step, task (the <user_request> text), url, screenshot (path),
history (executed action texts before this step), action (executed tool-call text), thought, candidates (all
proposed action texts; 1 for no-ARM collections, 5 for ARM-selection ones), selected_index, reward (episode reward
on the final step, else 0), done. Failed episodes are kept: IQL learns from both outcomes. Stdlib only.
"""
import json
from pathlib import Path
import re
import sys

USER_REQUEST = re.compile(r'<user_request>\s*(.*?)\s*</user_request>', re.S)


def action_text(raw):
    return raw.replace('<|im_end|>', '').strip()


def main():
    out, roots = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    n_ep = n_rows = n_skip = 0
    with out.open('w') as f:
        for root in roots:
            for outcome_path in sorted(root.glob('attempts/*/attempt-*/outcome.json')):
                outcome = json.loads(outcome_path.read_text())
                if not outcome.get('valid'):
                    n_skip += 1; continue
                attempt = outcome_path.parent
                turns = sorted(attempt.glob('turn-*.json'))
                if not turns:
                    n_skip += 1; continue
                history, rows = [], []
                for i, p in enumerate(turns):
                    t = json.loads(p.read_text())
                    cands = t.get('candidates') or []
                    k = t.get('selected_index') or 0
                    if not cands or k >= len(cands):
                        rows = None; break
                    task = USER_REQUEST.search(t.get('prompt', ''))
                    action = action_text(cands[k]['action'])
                    rows.append(dict(episode=str(attempt), collection=root.name, task_id=t.get('task_id'), step=i,
                                     task=task.group(1) if task else '', url=t.get('url'),
                                     screenshot=(t.get('images') or [{}])[0].get('path'), history=list(history),
                                     action=action, thought=cands[k].get('thought', ''),
                                     candidates=[action_text(c['action']) for c in cands], selected_index=k,
                                     reward=0.0, done=False))
                    history.append(action)
                if not rows:
                    n_skip += 1; continue
                rows[-1].update(reward=float(outcome.get('reward') or 0.0), done=True)
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                n_ep += 1; n_rows += len(rows)
    print(json.dumps(dict(episodes=n_ep, transitions=n_rows, skipped=n_skip, out=str(out))))


if __name__ == '__main__':
    main()
