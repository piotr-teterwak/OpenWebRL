#!/usr/bin/env python3
"""Back up the iterative Luna RPO experiment (tillicum/luna_iter_pipeline.sh) off-cluster as pieces finish.

    python tillicum/luna_iter_backup.py            # one pass; luna_iter_backup.sh loops it

Each finished item is uploaded once (state in luna-iter/backup-state.json), all to private HF repos:
  dataset PTeterwak/openwebrl-luna-iter
    collections/<shard>/   complete collection shards: training.jsonl, audits, attempts/ and images/ (the builder's inputs),
                           selections/, results/, Luna API usage (samples/ and server logs are left out)
    data/<name>/           built pair sets (joint-data/luna-*)
    evals/<label>/<bench>/ eval summaries and per-task result JSONs (no screenshots)
  model   PTeterwak/openwebrl-4b-<run>    final LoRA adapter (no optimizer state) + frozen config, status, gate summary
Small summaries are also copied to tillicum/results/luna-iter/ and committed + pushed to the current branch.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

L = Path('/gpfs/scrubbed/piotrt/owrl/c2-collect/luna-iter'); C = L.parent
D = Path('/gpfs/scrubbed/piotrt/owrl/joint-data'); RUNS = Path('/gpfs/scrubbed/piotrt/owrl/joint-retrain/runs')
EVAL = Path('/gpfs/scrubbed/piotrt/owrl/eval/passluna-iter'); LOGS = Path('/gpfs/scrubbed/piotrt/owrl/logs')
REPO = Path('/gpfs/home/piotrt/OpenWebRL'); SUM = REPO / 'tillicum/results/luna-iter'
DS = 'PTeterwak/openwebrl-luna-iter'
RUN_NAMES = ['r2-luna-r1', 'r2-luna-joint', 'r2-luna-cont', 'r2-luna-ctrl']
SECRET = re.compile(r'sk-proj-|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_|BEGIN [A-Z ]*PRIVATE KEY')
api = HfApi()
state_path = L / 'backup-state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}


def log(*a):
    print('[backup]', *a, flush=True)


def done(key, fn):
    if state.get(key):
        return
    try:
        fn(); state[key] = True; state_path.write_text(json.dumps(state, indent=1) + '\n'); log('uploaded', key)
    except Exception as e:  # leave it for the next pass
        log('FAILED', key, repr(e)[:300])


def scan(folder, patterns):
    for p in patterns:
        for f in folder.glob(p):
            if f.is_file() and f.suffix in ('.json', '.jsonl', '.log', '.txt', '.md') and SECRET.search(f.read_text(errors='ignore')):
                raise RuntimeError(f'possible secret in {f}')


def upload_dir(folder, dest, allow, repo=DS, repo_type='dataset', ignore=None):
    scan(folder, allow)
    api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=str(folder), path_in_repo=dest,
                      allow_patterns=allow, ignore_patterns=ignore, commit_message=f'add {dest or repo}')


def main():
    api.create_repo(DS, repo_type='dataset', private=True, exist_ok=True)
    SUM.mkdir(parents=True, exist_ok=True)
    for f in ['split-report.json', 'round1-jobs.txt', 'round2-jobs.txt', 'pipeline.log']:
        if (L / f).exists(): shutil.copy2(L / f, SUM / f)
    if not state.get('splits'):
        done('splits', lambda: upload_dir(L, 'splits', ['tasks-*.jsonl', 'split-report.json']))
    # collection shards
    for audit in sorted(C.glob('luna-*-s4[23]-shard[12]/dataset-audit.json')):
        s = audit.parent
        if json.loads(audit.read_text()).get('complete_collection'):
            shutil.copy2(audit, SUM / f'{s.name}.dataset-audit.json')
            if (s / 'collection-summary.json').exists(): shutil.copy2(s / 'collection-summary.json', SUM / f'{s.name}.collection-summary.json')
            done(f'collection:{s.name}', lambda s=s: upload_dir(s, f'collections/{s.name}', [
                'training.jsonl', 'dataset-audit.json', 'collection-summary.json', 'frozen-config.json', 'export-smoke-passed.json',
                'attempts/**', 'images/**', 'selections/**', 'results/**', 'luna-api-*/usage.json', 'luna-api-*/api-requests.jsonl']))
    # built pair sets
    for cfg in sorted(D.glob('luna-*/train-config.json')):
        d = cfg.parent
        if d.name.endswith('-full'):
            continue
        shutil.copy2(cfg, SUM / f'{d.name}.train-config.json')
        if (d / 'filter-report.json').exists(): shutil.copy2(d / 'filter-report.json', SUM / f'{d.name}.filter-report.json')
        done(f'data:{d.name}', lambda d=d: upload_dir(d, f'data/{d.name}', ['*.json', '*.jsonl']))
    # trained models
    plog = (L / 'pipeline.log').read_text() if (L / 'pipeline.log').exists() else ''
    for run in RUN_NAMES:
        r = RUNS / run
        if not (r / 'student/complete.json').exists():
            continue
        u = '%06d' % json.loads((r / 'frozen-config.json').read_text())['total_updates']
        # pipeline.log is block-buffered by its sed prefixers, so find the gate job by name
        gates = subprocess.run(['sacct', '-n', '-X', '-S', '2026-11-09', '--name', f'gt-{run}', '-o', 'JobID'],
                               capture_output=True, text=True).stdout.split()
        gsum = LOGS / f'gate-{gates[-1]}.summary' if gates else None
        for f in [r / 'frozen-config.json', r / 'training-status.json']:
            if f.exists(): shutil.copy2(f, SUM / f'{run}.{f.name}')
        if gsum and gsum.exists(): shutil.copy2(gsum, SUM / f'{run}.gate.summary')

        def up(run=run, r=r, u=u, gsum=gsum):
            repo = f'PTeterwak/openwebrl-4b-{run}'
            api.create_repo(repo, private=True, exist_ok=True)
            upload_dir(r / 'student' / f'update-{u}', '', ['*'], repo=repo, repo_type='model', ignore=['optimizer.pt'])
            for f in [r / 'frozen-config.json', r / 'training-status.json', r / 'student/metrics.jsonl', gsum]:
                if f and f.exists():
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f'training/{f.name}', repo_id=repo, commit_message='training metadata')
        done(f'model:{run}', up)
    # evals (one summary per label per benchmark)
    for summ in sorted(EVAL.glob('*/*/summary.json')):
        b = summ.parent; label = b.parent.name
        shutil.copy2(summ, SUM / f'eval.{label}.{b.name}.summary.json')
        done(f'eval:{label}/{b.name}', lambda b=b, label=label: upload_dir(b, f'evals/{label}/{b.name}', ['summary.json', 'results/*.json', '*.jsonl']))
    # git: summaries only
    scan(SUM, ['*'])
    subprocess.run(['git', 'add', str(SUM)], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', str(SUM)], cwd=REPO).returncode:
        msg = ('luna-iter: results snapshot\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
               'Claude-Session: https://claude.ai/code/session_016S4mcroQkpMqianwQVfBrb\n')
        subprocess.run(['git', 'commit', '-q', '-m', msg, '--', str(SUM)], cwd=REPO, check=True)
        p = subprocess.run(['git', 'push', '-q', 'origin', 'HEAD'], cwd=REPO, capture_output=True, text=True)
        log('git push', 'ok' if p.returncode == 0 else p.stderr[-300:])


if __name__ == '__main__':
    main()
