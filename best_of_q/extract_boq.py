#!/usr/bin/env python3
"""IQL transitions from one Best-of-Q run directory (boq_run.py / arm_eval output).

    extract_boq.py RUN_DIR [RUN_DIR ...]      # writes RUN_DIR/boq/rows.jsonl for each

One row per executed step of every valid episode, in the format train_q.py and embed.py read: episode (run/task_id),
collection (run directory name), task_id, step, task (intent), url, screenshot_ref ([sample JSON path, step]: the
screenshot the actor saw at that step, verified against the step's recorded sha256), history (executed action texts
before this step), action (executed tool call), thought, candidates (all N proposed action texts), selected_index,
reward (episode reward on the final row, else 0), done. Invalid episodes (unavailable site, judge failure) are dropped;
failed episodes are kept. Stdlib only.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path


def clean(action):
    return (action or '').replace('<|im_end|>', '').strip()


def extract(run):
    samples = {}
    for p in run.glob('samples/turn/*/*.json'):
        samples.setdefault(p.name.rsplit('_', 4)[0], []).append(p)  # <task_id>_<date>_<time>_<status>_<N>steps.json
    out = run / 'boq'; out.mkdir(exist_ok=True)
    n_ep = n_rows = 0; skipped = {}
    with (out / 'rows.jsonl').open('w') as f:
        for res in sorted(run.glob('results/*.json')):
            r = json.loads(res.read_text())
            if not r.get('valid'):
                skipped['invalid'] = skipped.get('invalid', 0) + 1; continue
            tid = r['task_id']; key = hashlib.sha256(tid.encode()).hexdigest()[:20]
            sel = run / 'selections' / f'{key}.jsonl'
            files = samples.get(tid, [])
            if not sel.exists() or len(files) != 1:
                skipped['no_trace'] = skipped.get('no_trace', 0) + 1; continue
            recs = [json.loads(l) for l in sel.open()]
            starts = [i for i, x in enumerate(recs) if x['turn'] == 0]
            recs = recs[starts[-1]:] if starts else []  # a resumed task re-ran from turn 0
            images = json.loads(files[0].read_text())['images']
            if not recs or len(images) < len(recs) or [x['turn'] for x in recs] != list(range(len(recs))):
                skipped['misaligned'] = skipped.get('misaligned', 0) + 1; continue
            ok = True
            for x in recs:
                img = images[x['turn']][0]
                raw = base64.b64decode(img.split(',', 1)[1] if img.startswith('data:') else img)
                if hashlib.sha256(raw).hexdigest() != x['screenshot_sha256']:
                    ok = False; break
            if not ok:
                skipped['screenshot_mismatch'] = skipped.get('screenshot_mismatch', 0) + 1; continue
            history, rows = [], []
            for x in recs:
                k = x['selected_index']; cand = x['candidates']
                action = clean(cand[k]['action'])
                rows.append(dict(episode=f'{run.name}/{tid}', collection=run.name, task_id=tid, step=x['turn'],
                                 task=r.get('intent', ''), url=x.get('url', ''),
                                 screenshot_ref=[str(files[0]), x['turn']], history=list(history), action=action,
                                 thought=cand[k].get('thought', ''), candidates=[clean(c['action']) for c in cand],
                                 selected_index=k, reward=0.0, done=False))
                history.append(action)
            rows[-1].update(reward=float(r.get('reward') or 0.0), done=True)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_ep += 1; n_rows += len(rows)
    summary = dict(run=str(run), episodes=n_ep, transitions=n_rows, skipped=skipped)
    (out / 'extract.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        extract(Path(p))
