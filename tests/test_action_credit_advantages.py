"""Analytical tests for the offline action-credit estimators (no torch/browser)."""

from dataclasses import replace
import itertools
import math
import unittest

from openwebrl.action_credit_advantages import (
    CandidateDraw, Episode, Outcome, Step, candidate_q_advantages,
    gigpo_advantages, outcome_group_advantages, tree_grpo_advantages,
)


def outcome(i, reward, tree="tree-a", **kw):
    return Outcome(str(i), "task", "group", reward, tree_id=tree, **kw)


def draw(i, candidate, reward, **kw):
    return CandidateDraw(str(i), "task", "state", candidate, reward, **kw)


def episode(i, rewards, states=None, **kw):
    states = states or ["anchor"] * len(rewards)
    steps = tuple(Step(str(j), state, reward, j)
                  for j, (state, reward) in enumerate(zip(states, rewards)))
    return Episode(str(i), "task", "group", sum(rewards), steps, **kw)


def run(fn, rows, **kw):
    field = {outcome_group_advantages: "rollout_id", tree_grpo_advantages: "rollout_id",
             candidate_q_advantages: "draw_id", gigpo_advantages: "trajectory_id"}[fn]
    return fn(rows, expected_ids=[getattr(r, field) for r in rows], **kw)


def by_id(result):
    return {v.key: v for v in result.values}


class OutcomeTests(unittest.TestCase):
    def test_hand_computed_sample_vs_population_std(self):
        rows = [outcome("fail", 0), outcome("success", 1)]
        sample = run(outcome_group_advantages, rows)
        pop = run(outcome_group_advantages, rows, ddof=0)
        self.assertAlmostEqual(by_id(sample)[("success",)].advantage,
                               .5 / (math.sqrt(.5) + 1e-6))
        self.assertAlmostEqual(by_id(pop)[("success",)].advantage, .5 / (.5 + 1e-6))
        self.assertEqual(sample.groups[0].normalization_count, 2)
        self.assertEqual(sample.groups[0].ddof, 1)
        self.assertEqual(sample.groups[0].mean, .5)

    def test_mean_only_and_epsilon_is_outside_sqrt(self):
        rows = [outcome(0, 0), outcome(1, 2)]
        self.assertEqual([v.advantage for v in run(outcome_group_advantages, rows,
                                                 normalization="mean").values], [-1, 1])
        r = run(outcome_group_advantages, rows, epsilon=.5, ddof=0)
        self.assertAlmostEqual(r.values[1].advantage, 1 / 1.5)

    def test_ties_zeros_and_explicit_singletons(self):
        for value in (0, 1, -2):
            self.assertEqual([v.advantage for v in run(outcome_group_advantages,
                             [outcome(0, value), outcome(1, value)]).values], [0, 0])
        self.assertAlmostEqual(run(outcome_group_advantages, [outcome(0, 1)]).values[0].advantage,
                               1 / (1 + 1e-6))
        self.assertEqual(run(outcome_group_advantages, [outcome(0, 1)],
                             singleton="zero").values[0].advantage, 0)

    def test_invalid_and_missing_are_excluded_not_failures(self):
        rows = [outcome(0, 1), outcome(1, None, status="invalid"),
                outcome(2, None, status="missing")]
        result = run(outcome_group_advantages, rows, singleton="zero")
        self.assertEqual([v.advantage for v in result.values], [0, None, None])
        s = result.groups[0]
        self.assertEqual((s.planned_count, s.valid_count, s.invalid_count, s.missing_count,
                          s.normalization_count), (3, 1, 1, 1, 1))
        self.assertEqual(s.mean, 1)
        empty = run(outcome_group_advantages, rows[1:])
        self.assertIsNone(empty.groups[0].mean)
        self.assertIsNone(empty.groups[0].normalizer)

    def test_permutation_and_group_isolation(self):
        rows = [outcome(0, 0), outcome(1, 1),
                replace(outcome(2, 50), task_id="other", group_id="other-group")]
        expected = run(outcome_group_advantages, rows)
        for perm in itertools.permutations(rows):
            self.assertEqual(run(outcome_group_advantages, perm), expected)
        self.assertEqual(by_id(expected)[("0",)].advantage,
                         by_id(run(outcome_group_advantages, rows[:2]))[("0",)].advantage)

    def test_affine_reward_transform_for_mean_only(self):
        rows = [outcome(i, x) for i, x in enumerate([0, .25, 1])]
        before = run(outcome_group_advantages, rows, normalization="mean")
        after = run(outcome_group_advantages, [replace(r, reward=3*r.reward + 7) for r in rows],
                    normalization="mean")
        for a, b in zip(before.values, after.values):
            self.assertAlmostEqual(3*a.advantage, b.advantage)


class TreeTests(unittest.TestCase):
    def test_hand_computed_sum_of_two_independent_normalizations(self):
        rows = [outcome(0, 0), outcome(1, 1), outcome(2, 1, "b"), outcome(3, 1, "b")]
        result = by_id(run(tree_grpo_advantages, rows))
        # Across all four leaves: mean=.75, sample std=.5. Tree a: mean=.5, std=sqrt(.5).
        self.assertAlmostEqual(result[("0",)].advantage,
                               -.75 / (.5 + 1e-6) - .5 / (math.sqrt(.5) + 1e-6))
        self.assertAlmostEqual(result[("2",)].advantage, .25 / (.5 + 1e-6))
        self.assertEqual(dict(result[("2",)].components)["intra"], 0)

    def test_inter_tree_is_leaf_pooled_not_tree_mean_weighted(self):
        rows = [outcome(0, 0), outcome(1, 1, "b"), outcome(2, 1, "b"), outcome(3, 1, "b")]
        result = run(tree_grpo_advantages, rows, singleton="zero")
        inter = next(s for s in result.groups if s.key[0] == "inter")
        self.assertEqual(inter.mean, .75)  # Equal tree weights would incorrectly yield .5.
        self.assertEqual(inter.normalization_count, 4)
        self.assertAlmostEqual(by_id(result)[("0",)].advantage, -.75 / (.5 + 1e-6))

    def test_single_tree_doubles_outcome_advantage_without_renormalizing(self):
        rows = [outcome(0, 0), outcome(1, 1)]
        base = run(outcome_group_advantages, rows)
        tree = run(tree_grpo_advantages, rows)
        for b, t in zip(base.values, tree.values):
            self.assertEqual(t.advantage, 2*b.advantage)

    def test_upstream_singleton_is_nonzero_in_both_components(self):
        result = run(tree_grpo_advantages, [outcome(0, 1)])
        self.assertAlmostEqual(result.values[0].advantage, 2 / (1 + 1e-6))

    def test_tree_id_namespace_leakage_and_missing_tree(self):
        for rows in ([outcome(0, 0), replace(outcome(1, 1), group_id="other")],
                     [outcome(0, 0), replace(outcome(1, 1), tree_id=None)]):
            with self.assertRaises(ValueError):
                run(tree_grpo_advantages, rows)

    def test_permutation_ties_and_exclusions(self):
        rows = [outcome(0, 1), outcome(1, 1), outcome(2, None, "b", status="invalid")]
        expected = run(tree_grpo_advantages, rows)
        self.assertEqual([v.advantage for v in expected.values], [0, 0, None])
        for p in itertools.permutations(rows):
            self.assertEqual(run(tree_grpo_advantages, p), expected)


class ContinuationQTests(unittest.TestCase):
    def test_unequal_draws_center_candidates_equally(self):
        rows = [draw(0, "a", 1), draw(1, "b", 0), draw(2, "b", 0), draw(3, "b", 1)]
        result = by_id(run(candidate_q_advantages, rows))
        self.assertAlmostEqual(result[("task", "state", "a")].advantage, 1/3)
        self.assertAlmostEqual(result[("task", "state", "b")].advantage, -1/3)
        self.assertEqual(result[("task", "state", "b")].valid_count, 3)
        self.assertAlmostEqual(dict(result[("task", "state", "a")].components)["candidate_mean"], 2/3)

    def test_invalid_draws_are_not_zero_q(self):
        rows = [draw(0, "a", 1), draw(1, "a", None, status="invalid"),
                draw(2, "a", None, status="missing"), draw(3, "b", 0)]
        result = run(candidate_q_advantages, rows)
        self.assertEqual([v.advantage for v in result.values], [.5, -.5])
        self.assertEqual((result.values[0].valid_count, result.values[0].invalid_count,
                          result.values[0].missing_count), (1, 1, 1))

    def test_unavailable_candidate_blocks_entire_state_without_hiding_q(self):
        rows = [draw(0, "a", 1), draw(1, "b", None, status="missing")]
        result = run(candidate_q_advantages, rows)
        self.assertEqual([v.advantage for v in result.values], [None, None])
        self.assertEqual(dict(result.values[0].components)["q"], 1)
        center = next(s for s in result.groups if s.key[0] == "candidate_center")
        self.assertIsNone(center.mean)
        self.assertEqual((center.planned_count, center.valid_count, center.missing_count), (2, 1, 1))

    def test_singleton_ties_and_all_invalid(self):
        self.assertEqual(run(candidate_q_advantages, [draw(0, "a", 1)]).values[0].advantage, 0)
        for reward in (0, 1):
            r = run(candidate_q_advantages, [draw(0, "a", reward), draw(1, "b", reward)])
            self.assertEqual([v.advantage for v in r.values], [0, 0])
        r = run(candidate_q_advantages, [draw(0, "a", None, status="invalid")])
        self.assertIsNone(r.values[0].advantage)

    def test_permutation_and_within_candidate_replication(self):
        rows = [draw(0, "a", 1), draw(1, "b", 0), draw(2, "b", 1)]
        expected = run(candidate_q_advantages, rows)
        for p in itertools.permutations(rows):
            self.assertEqual(run(candidate_q_advantages, p), expected)
        replicated = run(candidate_q_advantages, rows + [draw(3, "b", 0), draw(4, "b", 1)])
        self.assertEqual([v.advantage for v in replicated.values], [v.advantage for v in expected.values])

    def test_holdout_and_cross_task_leakage_rejected(self):
        for rows in ([draw(0, "a", 1, partition="holdout")],
                     [draw(0, "a", 1), replace(draw(1, "b", 0), task_id="other")]):
            with self.assertRaises(ValueError):
                run(candidate_q_advantages, rows)
        with self.assertRaises(ValueError):
            run(candidate_q_advantages, [draw(0, "a", 1)], partition="holdout")


class GiGPOTests(unittest.TestCase):
    def test_discounted_terminal_return_and_joint_advantage(self):
        rows = [episode("win", [0, 1], ["start", "finish"]),
                episode("lose", [0, 0], ["start", "finish"])]
        values = by_id(run(gigpo_advantages, rows, gamma=.5))
        self.assertEqual(dict(values[("win", "0")].components),
                         {"episode": .5, "step": .25, "return": .5})
        self.assertEqual(values[("win", "0")].advantage, .75)
        self.assertEqual(values[("win", "1")].advantage, 1)
        self.assertEqual(values[("lose", "0")].advantage, -.75)

    def test_step_std_normalization_and_weight(self):
        rows = [episode("a", [0, 1], ["s", "s2"]), episode("b", [0, 0], ["s", "s2"])]
        result = by_id(run(gigpo_advantages, rows, gamma=.5, normalization="mean_std", step_weight=2))
        expected = .5/(math.sqrt(.5)+1e-6) + 2*.25/(math.sqrt(.125)+1e-6)
        self.assertAlmostEqual(result[("a", "0")].advantage, expected)

    def test_discount_zero_one_and_immediate_rewards(self):
        rows = [episode("a", [.2, .3, .5], ["s1", "s2", "s3"])]
        for gamma, expected in [(0, [.2, .3, .5]), (1, [1, .8, .5])]:
            result = run(gigpo_advantages, rows, gamma=gamma)
            for value, ret in zip(result.values, expected):
                self.assertAlmostEqual(dict(value.components)["return"], ret)
                self.assertEqual(dict(value.components)["step"], 0)

    def test_exact_state_is_scoped_within_group_and_task(self):
        rows = [episode("a", [1]), replace(episode("b", [0]), group_id="g2", task_id="t2")]
        result = run(gigpo_advantages, rows)
        self.assertEqual([dict(v.components)["step"] for v in result.values], [0, 0])
        self.assertEqual([v.valid_count for v in result.values], [1, 1])

    def test_repeated_visit_in_same_trajectory_is_in_anchor_group(self):
        result = run(gigpo_advantages, [episode("a", [0, 1])], gamma=.5, singleton="zero")
        self.assertEqual([dict(v.components)["step"] for v in result.values], [-.25, .25])
        self.assertEqual([v.valid_count for v in result.values], [2, 2])

    def test_episode_step_weighting_is_explicit_and_has_different_denominator(self):
        rows = [episode("short", [1]), episode("long", [0, 0, 0])]
        equal = run(gigpo_advantages, rows, step_weight=0)
        weighted = run(gigpo_advantages, rows, step_weight=0, episode_weighting="step")
        self.assertEqual(by_id(equal)[("short", "0")].advantage, .5)
        self.assertEqual(by_id(weighted)[("short", "0")].advantage, .75)
        es = next(s for s in equal.groups if s.key[0] == "episode")
        ws = next(s for s in weighted.groups if s.key[0] == "episode")
        self.assertEqual((es.valid_count, es.normalization_count), (2, 2))
        self.assertEqual((ws.valid_count, ws.normalization_count), (2, 4))
        self.assertNotEqual(equal.method, weighted.method)

    def test_invalid_partial_episode_and_missing_episode_are_excluded(self):
        rows = [episode("a", [1]), replace(episode("b", [0]), reward=None, status="invalid"),
                Episode("c", "task", "group", None, (), "missing")]
        result = run(gigpo_advantages, rows)
        self.assertIsNone(by_id(result)[("b", "0")].advantage)
        self.assertEqual(dict(by_id(result)[("a", "0")].components)["step"], 0)
        stat = next(s for s in result.groups if s.key[0] == "episode")
        self.assertEqual((stat.valid_count, stat.invalid_count, stat.missing_count), (1, 1, 1))

    def test_permutation_of_episodes_and_step_storage_preserves_positions(self):
        rows = [episode("a", [0, 1]), episode("b", [0, 0])]
        expected = run(gigpo_advantages, rows, gamma=.5)
        reverse = [replace(e, steps=tuple(reversed(e.steps))) for e in reversed(rows)]
        self.assertEqual(run(gigpo_advantages, reverse, gamma=.5), expected)

    def test_bad_positions_duplicate_steps_bad_reward_and_empty_valid_episode(self):
        e = episode("a", [0, 1])
        bad = [replace(e, reward=2), replace(e, steps=()),
               replace(e, steps=(e.steps[0], replace(e.steps[1], position=0))),
               replace(e, steps=(replace(e.steps[0], position=1), e.steps[1])),
               replace(e, steps=(e.steps[0], replace(e.steps[1], step_id="0"))),
               replace(e, steps=(replace(e.steps[0], reward=float("nan")), e.steps[1])),
               replace(e, steps=(replace(e.steps[0], position=True), e.steps[1]))]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                run(gigpo_advantages, [item])

    def test_tied_all_failure_has_no_fabricated_credit(self):
        result = run(gigpo_advantages, [episode("a", [0, 0]), episode("b", [0, 0])])
        self.assertTrue(all(v.advantage == 0 for v in result.values))


class ValidationTests(unittest.TestCase):
    def test_manifest_missing_extra_and_duplicate_rejected(self):
        for manifest in ([], ["0", "1"], ["other"], ["0", "0"], "0"):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                outcome_group_advantages([outcome(0, 1)], expected_ids=manifest)
        with self.assertRaises(ValueError):
            run(outcome_group_advantages, [outcome(0, 1), outcome(0, 0)])

    def test_nonfinite_missing_valid_imputed_invalid_and_boolean_reward_rejected(self):
        for reward in (float("nan"), float("inf"), -float("inf"), None, True, "1"):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                run(outcome_group_advantages, [outcome(0, reward)])
        for status in ("missing", "invalid", "unknown"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                run(outcome_group_advantages, [outcome(0, 0, status=status)])

    def test_invalid_options_and_ids(self):
        for kwargs in ({"ddof": True}, {"ddof": 2}, {"epsilon": 0}, {"epsilon": -1},
                       {"epsilon": float("nan")}, {"normalization": "bogus"}, {"singleton": "raw"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run(outcome_group_advantages, [outcome(0, 1)], **kwargs)
        for kwargs in ({"gamma": -1}, {"gamma": 1.01}, {"gamma": True},
                       {"step_weight": -1}, {"step_weight": float("inf")}, {"episode_weighting": "tokens"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run(gigpo_advantages, [episode("a", [1])], **kwargs)
        with self.assertRaises(ValueError):
            run(outcome_group_advantages, [replace(outcome(0, 1), group_id=" ")])

    def test_cross_task_group_alias_is_rejected(self):
        rows = [outcome(0, 0), replace(outcome(1, 1), task_id="different")]
        with self.assertRaises(ValueError):
            run(outcome_group_advantages, rows)

    def test_nonfinite_intermediate_arithmetic_fails_closed(self):
        with self.assertRaises(ValueError):
            run(outcome_group_advantages, [outcome(0, -1e308), outcome(1, 1e308)])

    def test_inputs_are_immutable_and_not_modified(self):
        rows = (outcome(0, 0), outcome(1, 1))
        before = repr(rows)
        run(tree_grpo_advantages, rows)
        self.assertEqual(repr(rows), before)
        with self.assertRaises(AttributeError):
            rows[0].reward = 2


if __name__ == "__main__":
    unittest.main()
