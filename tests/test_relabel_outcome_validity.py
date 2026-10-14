import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.relabel_outcome_validity import evaluations


class EvaluationRelabelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='outcome-validity-', dir='/tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.episode = self.root / 'episode.json'
        self.episode.write_text(json.dumps(dict(task_id='fixed-task', status='truncated',
            terminate_reason='generation_length_limit', valid=False, reward=0, metadata={})))
        self.cohort = dict(expected_task_ids=['fixed-task'],
            result_artifacts=[dict(path=str(self.episode),
                sha256=hashlib.sha256(self.episode.read_bytes()).hexdigest())],
            native_documented_counts=dict(valid=0, successes=0))
        self.manifest = self.root / 'manifest.json'

    def run_manifest(self):
        self.manifest.write_text(json.dumps({'exact-experiment': self.cohort}))
        return evaluations(self.manifest)['cohorts']['exact-experiment']

    def test_agent_limit_reclassified_without_changing_success(self):
        row = self.run_manifest()
        self.assertEqual((row['native_valid'], row['unified_valid'], row['canonical_successes']), (0, 1, 0))

    def test_changed_episode_rejected(self):
        self.episode.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.run_manifest()

    def test_wrong_cohort_rejected(self):
        self.cohort['expected_task_ids'] = ['another-experiment-task']
        with self.assertRaisesRegex(ValueError, 'fixed cohort'):
            self.run_manifest()

    def test_unreproduced_native_counts_rejected(self):
        self.cohort['native_documented_counts']['valid'] = 1
        with self.assertRaisesRegex(ValueError, 'documented native counts'):
            self.run_manifest()


if __name__ == '__main__':
    unittest.main()
