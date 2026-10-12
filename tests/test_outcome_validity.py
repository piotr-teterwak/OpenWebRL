import unittest

from openwebrl.outcome_validity import classify_branch, classify_molmoweb, classify_rl_trajectory, classify_terminate_reason


def rl_turns(status, reason, **reward):
    return [dict(status=status, metadata=dict(turn_index=0)),
            dict(status=status, metadata=dict(turn_index=1, terminate_reason=reason, reward=reward))]


class TerminateReasonTest(unittest.TestCase):
    def test_agent_endings_are_valid(self):
        for status, reason, ending in [('completed', 'task_completed', 'done'),
                                       ('failed', 'max_steps_exhausted', 'step_limit'),
                                       ('truncated', 'generation_length_limit', 'context_limit'),
                                       ('truncated', 'rollout_max_context_length_exceeded', 'context_limit'),
                                       ('failed', 'format_error_failed', 'format_failure')]:
            outcome = classify_terminate_reason(status, reason)
            self.assertEqual((outcome.ending, outcome.valid), (ending, True))

    def test_infrastructure_endings_are_invalid(self):
        for reason, ending in [('env_step_error', 'browser_env'),
                               ('generation_error: local_process env_server error [500] /reset: {...}', 'browser_env'),
                               ('generation_abort', 'model_server'),
                               ('generation_error: SGLang /generate timed out after 30.0s', 'model_server'),
                               ('generation_error: rollout_task_timeout after 3600s', 'wall_timeout')]:
            outcome = classify_terminate_reason('Status.ABORTED', reason)
            self.assertEqual((outcome.ending, outcome.valid), (ending, False))

    def test_judge_failure_on_completed_episode_is_invalid(self):
        outcome = classify_rl_trajectory(rl_turns('completed', 'task_completed', judge_timeout=True))
        self.assertEqual((outcome.ending, outcome.valid), ('judge', False))
        self.assertTrue(classify_rl_trajectory(rl_turns('completed', 'task_completed', judge_timeout=False)).valid)


class BranchTest(unittest.TestCase):
    def test_action_limit_reported_as_step_error_stays_valid(self):
        result = dict(valid=True, reward=0.0, reason=None, browser_terminal_reason='action_limit')
        outcome = classify_branch(result, dict(status='aborted', terminate_reason='env_step_error'))
        self.assertEqual((outcome.ending, outcome.valid), ('action_limit', True))

    def test_step_timeout_is_browser_env(self):
        result = dict(valid=False, reward=0.0, reason='missing_fresh_terminal_evidence')
        outcome = classify_branch(result, dict(status='aborted', terminate_reason='env_step_error'))
        self.assertEqual((outcome.ending, outcome.valid), ('browser_env', False))

    def test_episode_deadline_is_wall_timeout(self):
        outcome = classify_branch(dict(valid=True, reward=0.0, policy_boundary_reason='episode_deadline'), None)
        self.assertEqual((outcome.ending, outcome.valid), ('wall_timeout', False))

    def test_completed_without_fresh_capture_is_capture(self):
        result = dict(valid=False, reward=None, reason='missing_fresh_terminal_evidence')
        outcome = classify_branch(result, dict(status='completed', terminate_reason='task_completed'))
        self.assertEqual((outcome.ending, outcome.valid), ('capture', False))


class MolmoWebTest(unittest.TestCase):
    def test_statuses(self):
        judged = dict(native=dict(judged=True))
        self.assertTrue(classify_molmoweb(dict(status='completed', valid=True), judged).valid)
        self.assertEqual(classify_molmoweb(dict(status='completed', valid=True), dict(native=dict(judged=False))).ending, 'judge')
        self.assertEqual(classify_molmoweb(dict(status='completed_timeout', valid=False)).ending, 'wall_timeout')
        self.assertEqual(classify_molmoweb(dict(status='completed_invalid', valid=False,
                                                invalid_reason=dict(classification='ordinary_browser_unavailability',
                                                                    phase='navigation'))).ending, 'browser_env')
        failed = dict(status='failed', valid=False, episode=dict(error=['ACTION_PREDICTION_ERROR']))
        self.assertEqual(classify_molmoweb(failed, None, "Error code: 503 - overloaded").ending, 'model_server')


if __name__ == '__main__':
    unittest.main()
