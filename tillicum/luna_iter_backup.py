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
import os
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
RUN_NAMES = ['r2-luna-r1', 'r2-luna-joint', 'r2-luna-cont', 'r2-luna-ctrl',
             'r2-luna-r1-1ep-s1', 'r2-luna-ctrl-1ep-s1', 'r2-luna-cont-refr1-1ep',  # + stage 1 (luna_stage1.sh)
             'r2-sol-r1', 'r2-sol-r1-m0.3']  # + option C (sol_relabel_pipeline.sh)
# Key-shaped strings must start at a token boundary: browsed pages carry long opaque URL / base64 blobs that contain
# 'hf_' + 20 alphanumerics mid-string (an Amazon URL in DeepShop Amazon--134). Our real keys are also matched exactly.
SECRET = re.compile(r'(?<![A-Za-z0-9_-])(?:sk-proj-|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_)|BEGIN [A-Z ]*PRIVATE KEY')
OWN = [v for k, v in os.environ.items() if k in ('OPENAI_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY') and len(v) > 20]
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
            if f.is_file() and f.suffix in ('.json', '.jsonl', '.log', '.txt', '.md') and (SECRET.search(t := f.read_text(errors='ignore')) or any(k in t for k in OWN)):
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
    for cfg in sorted(list(D.glob('luna-*/train-config.json')) + list(D.glob('sol-*/train-config.json'))):
        d = cfg.parent
        if d.name.endswith('-full'):
            continue
        shutil.copy2(cfg, SUM / f'{d.name}.train-config.json')
        if (d / 'filter-report.json').exists(): shutil.copy2(d / 'filter-report.json', SUM / f'{d.name}.filter-report.json')
        done(f'data:{d.name}', lambda d=d: upload_dir(d, f'data/{d.name}', ['*.json', '*.jsonl']))
    # Sol relabels (option C): labels + API usage
    SOL = C / 'sol-relabel'
    for d in sorted(SOL.glob('A-s4[23]')):
        if (d / 'sol-api/usage.json').exists() and (d / 'labels.jsonl').exists() and not subprocess.run(['squeue', '-h', '-u', 'piotrt', '-n', f'sol-rl-{d.name.replace("-s", "")}'], capture_output=True, text=True).stdout.strip():
            shutil.copy2(d / 'sol-api/usage.json', SUM / f'sol-relabel.{d.name}.usage.json')
            done(f'sol-labels:{d.name}', lambda d=d: upload_dir(d, f'sol-relabel/{d.name}', ['labels.jsonl', 'sol-api/usage.json', 'sol-api/api-requests.jsonl']))
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
            upload_dir(r / 'student' / f'update-{u}', '', ['*'], repo=repo, repo_type='model', ignore=['optimizer.pt', 'README.md'])  # PEFT README's base_model is a local path, which HF rejects
            for f in [r / 'frozen-config.json', r / 'training-status.json', r / 'student/metrics.jsonl', gsum]:
                if f and f.exists():
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f'training/{f.name}', repo_id=repo, commit_message='training metadata')
        done(f'model:{run}', up)
    # evals (one summary per label per benchmark)
    # passluna-iter keeps its original keys/paths; later passes (luna-iter-e1, -s1) are prefixed by pass name
    for summ in sorted(list(EVAL.parent.glob('passluna-iter*/*/*/summary.json')) + list(EVAL.parent.glob('passsol-relabel/*/*/summary.json'))):
        b = summ.parent; label = b.parent.name; ps = b.parent.parent.name.removeprefix('pass')
        tag = '' if ps == 'luna-iter' else ps + '/'
        shutil.copy2(summ, SUM / f"eval.{tag.replace('/', '.')}{label}.{b.name}.summary.json")
        done(f'eval:{tag}{label}/{b.name}', lambda b=b, d=f'evals/{tag}{label}/{b.name}': upload_dir(b, d, ['summary.json', 'results/*.json', '*.jsonl']))
    # git: summaries only
    scan(SUM, ['*'])
    subprocess.run(['git', 'add', '-f', str(SUM)], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', str(SUM)], cwd=REPO).returncode:
        msg = ('luna-iter: results snapshot\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
               'Claude-Session: https://claude.ai/code/session_016S4mcroQkpMqianwQVfBrb\n')
        subprocess.run(['git', 'commit', '-q', '-m', msg, '--', str(SUM)], cwd=REPO, check=True)
        p = subprocess.run(['git', 'push', '-q', 'origin', 'HEAD'], cwd=REPO, capture_output=True, text=True)
        log('git push', 'ok' if p.returncode == 0 else p.stderr[-300:])


if __name__ == '__main__':
    main()
