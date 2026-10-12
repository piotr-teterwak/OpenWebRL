#!/usr/bin/env python3
"""Relabel saved RL / branch / MolmoWeb episodes with openwebrl.outcome_validity.

Read-only over run artifacts. Writes one JSON summary: per system, the native
valid/invalid counts, the unified counts, the unified ending breakdown, and the
native-vs-unified disagreement table.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.outcome_validity import classify_branch, classify_molmoweb, classify_rl_trajectory  # noqa: E402

RUNTIME = '/gpfs/scrubbed/zixianma/openwebrl-runtime'
BRANCH_GLOBS = (
    RUNTIME + '/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/fast1k-20261012/jobs/job*/seg*/attempts/*/result.json',
    RUNTIME + '/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-matched-parallel10-v1/shards/shard*/attempts/*/result.json',
)
RL_GLOB = RUNTIME + '/runs/*/completed_rollout_archive/iteration_*/group_*.json.gz'
MOLMO_GLOB = RUNTIME + '/evaluations/molmoweb-sol61-20261009/jobs/**/worker-result.json'


class Tally:
    def __init__(self):
        self.native = collections.Counter()
        self.unified = collections.Counter()
        self.endings = collections.Counter()
        self.disagree = collections.Counter()

    def add(self, native_valid, outcome, native_label):
        self.native[native_valid] += 1
        self.unified[outcome.valid] += 1
        self.endings[outcome.ending] += 1
        if bool(native_valid) != outcome.valid:
            label = re.sub(r'[0-9]+', 'N', native_label)[:70]
            self.disagree[f'{label} -> {outcome.ending}'] += 1

    def summary(self):
        n = sum(self.native.values())
        return dict(episodes=n, native_valid=self.native[True], native_invalid=n - self.native[True],
                    unified_valid=self.unified[True], unified_invalid=n - self.unified[True],
                    unified_endings=dict(self.endings.most_common()),
                    native_to_unified_changes=dict(self.disagree.most_common()))


def last_rollout(attempt_dir):
    files = sorted(glob.glob(os.path.join(attempt_dir, 'rollouts', 'turn', '*', '*.json')))
    if not files:
        return None
    value = json.loads(Path(files[-1]).read_text())
    return dict(status=value.get('status'), terminate_reason=value.get('terminate_reason'))


def branch(limit):
    tally = Tally()
    for pattern in BRANCH_GLOBS:
        for path in sorted(glob.glob(pattern))[:limit]:
            result = json.loads(Path(path).read_text())
            if result.get('mode') != 'branch' or not result.get('released'):
                continue
            outcome = classify_branch(result, last_rollout(os.path.dirname(path)))
            tally.add(result.get('valid') is True, outcome, f"native:{result.get('reason')}")
    return tally


def rl(limit):
    tally, runs = Tally(), collections.Counter()
    for path in sorted(glob.glob(RL_GLOB))[:limit]:
        group = json.load(gzip.open(path))
        runs[path.split('/runs/')[1].split('/')[0]] += 1
        for trajectory in group['trajectories']:
            turns = trajectory['turns']
            last = max(turns, key=lambda s: (s.get('metadata') or {}).get('turn_index', 0))
            native_valid = not any(s.get('remove_sample') for s in turns)  # executed rule: remove_sample only
            outcome = classify_rl_trajectory(turns)
            tally.add(native_valid, outcome, f"native:{(last.get('metadata') or {}).get('terminate_reason')}")
    return tally, runs


def molmoweb(limit):
    tally = Tally()
    for path in sorted(glob.glob(MOLMO_GLOB, recursive=True))[:limit]:
        directory = os.path.dirname(path)
        worker = json.loads(Path(path).read_text())
        judge_path = os.path.join(directory, 'judge-result.json')
        judge = json.loads(Path(judge_path).read_text()) if os.path.exists(judge_path) else None
        message = ''
        if 'ACTION_PREDICTION_ERROR' in ((worker.get('episode') or {}).get('error') or []):
            for name in glob.glob(os.path.join(directory, '**', '*.json'), recursive=True):
                match = re.search(r'"ACTION_PREDICTION_ERROR": "([^"]{0,200})', Path(name).read_text(errors='replace'))
                if match:
                    message = match.group(1)
                    break
        native_valid = worker.get('valid') is True and (judge is None or judge.get('valid') is not False)
        tally.add(native_valid, classify_molmoweb(worker, judge, message), f"native:{worker.get('status')}")
    return tally


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None, help='files per glob (testing)')
    args = parser.parse_args()
    rl_tally, rl_runs = rl(args.limit)
    report = dict(schema='outcome-validity-relabel-v1',
                  rule='openwebrl/outcome_validity.py: infrastructure endings invalid, agent endings valid; '
                       'wall-clock and model-server endings invalid',
                  branch=branch(args.limit).summary(),
                  rl=dict(rl_tally.summary(), group_files_by_run=dict(rl_runs)),
                  molmoweb=molmoweb(args.limit).summary())
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name in ('branch', 'rl', 'molmoweb'):
        r = report[name]
        print(name, r['episodes'], 'native valid', r['native_valid'], 'unified valid', r['unified_valid'],
              'changes', r['native_to_unified_changes'])


if __name__ == '__main__':
    main()
