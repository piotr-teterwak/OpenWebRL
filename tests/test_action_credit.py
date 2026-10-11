"""Analytical loss/gradient and hostile-provenance tests, no browser or GPU."""
from dataclasses import replace
import math
import unittest

import torch

from openwebrl.action_credit import (CollectionContract, ResponseEdge, Leaf, ResponseGraph,
    append_local_terms, clipped_loss, digest, leaf_loss_plan, response_log_probs)


def graph_fixture():
    contract = CollectionContract(0, *[digest(x) for x in
        ("policy", "tokenizer", "processor", "tasks", "judge", "generation")], .8, ("task",), ("test",))
    def edge(key, parent, tokens=(2, 3), mask=(True, True)):
        return ResponseEdge(key, "task", "tree", parent, contract.sha256, (0, 1),
            (digest("image"),), tokens, mask, tuple(-2. for _ in tokens), contract.policy_sha256,
            digest("state:" + str(parent)), digest("replay:" + key), 30 if parent is None else 29)
    edges = (edge("prefix", None), edge("good", "prefix"), edge("bad", "prefix"))
    leaves = (Leaf("a", "task", "tree", ("prefix", "good"), "valid", 1., group_id="task-group"),
              Leaf("b", "task", "tree", ("prefix", "bad"), "valid", 0., group_id="task-group"))
    return ResponseGraph(contract, edges, leaves, ("a", "b"))


class GraphValidation(unittest.TestCase):
    def test_valid_graph(self):
        self.assertEqual(len(graph_fixture().validate()), 3)

    def test_rejects_stale_policy_protocol_and_synthetic_old_scores(self):
        graph = graph_fixture()
        for changes in ({"old_policy_sha256": digest("stale")}, {"contract_sha256": digest("stale")},
                        {"old_log_prob_source": "sglang_zero_completion"}, {"replay_evidence_sha256": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(graph, edges=(replace(graph.edges[0], **changes), *graph.edges[1:])).validate()

    def test_bad_masks_tokens_and_logps(self):
        graph = graph_fixture()
        for changes in ({"sampled_mask": (False, False)}, {"sampled_mask": (1, 1)},
                        {"old_log_probs": (-2., float("nan"))}, {"old_log_probs": (1., -2.)},
                        {"response_ids": (2, -1)}, {"response_ids": [2, 3]}, {"input_ids": ()}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(graph, edges=(replace(graph.edges[0], **changes), *graph.edges[1:])).validate()

    def test_cross_tree_broken_paths_duplicates_and_cycles(self):
        graph = graph_fixture()
        for leaf in (replace(graph.leaves[0], tree_id="other"),
                     replace(graph.leaves[0], edge_ids=("good", "prefix")),
                     replace(graph.leaves[0], edge_ids=("prefix", "good", "prefix")),
                     replace(graph.leaves[0], leaf_id="b")):
            with self.subTest(leaf=leaf), self.assertRaises(ValueError):
                replace(graph, leaves=(leaf, graph.leaves[1])).validate()

    def test_independent_chains_reject_shared_samples(self):
        with self.assertRaises(ValueError):
            graph_fixture().validate(independent_chains=True)

    def test_horizon_cannot_reset_on_child(self):
        graph = graph_fixture()
        with self.assertRaises(ValueError):
            replace(graph, edges=(graph.edges[0], replace(graph.edges[1], remaining_turns=30), graph.edges[2])).validate()

    def test_split_and_invalid_fail_closed(self):
        graph = graph_fixture()
        for leaf in (replace(graph.leaves[0], task_id="test"),
                     replace(graph.leaves[0], status="invalid", reward=0.),
                     replace(graph.leaves[0], status="valid", reward=None)):
            with self.assertRaises(ValueError):
                replace(graph, leaves=(leaf, graph.leaves[1])).validate()
        with self.assertRaises(ValueError):
            replace(graph.contract, heldout_task_ids=("task",)).validate()


class ClippedObjective(unittest.TestCase):
    def test_leaf_selection_retains_inventory_but_changes_only_fit_denominator(self):
        graph = graph_fixture()
        for reduction in ("response_mean", "trajectory_mean", "token_mean"):
            plan = leaf_loss_plan(graph, {"a": 1.}, arm_id="tree_grpo", reduction=reduction,
                                  selected_leaf_ids=("a",))
            self.assertIs(plan.graph, graph)
            self.assertEqual(tuple(x.leaf_id for x in plan.graph.leaves), ("a", "b"))
            self.assertEqual({t.source_id for t in plan.terms}, {"a"})
            self.assertEqual([t.weight for t in plan.terms], [.5, .5])
            with self.assertRaises(ValueError):
                replace(plan, terms=(*plan.terms, replace(plan.terms[-1], edge_id="bad", source_id="b"))).validate()
        for selected in (("a", "a"), ("unknown",), [], ()):
            with self.assertRaises(ValueError):
                leaf_loss_plan(graph, {}, arm_id="tree_grpo", reduction="response_mean", selected_leaf_ids=selected)

    def plan(self, reduction="response_mean"):
        return leaf_loss_plan(graph_fixture(), {"a": 1., "b": -1.},
                              arm_id="tree_outcome_sampling", reduction=reduction)

    def logps(self, ratio=1.):
        return {e: torch.full((2,), -2 + math.log(ratio), dtype=torch.float64, requires_grad=True)
                for e in ("prefix", "good", "bad")}

    def test_shared_prefix_mixed_signs_cannot_average_before_clip(self):
        plan = self.plan()
        values = self.logps(ratio=1.5)
        loss = clipped_loss(plan, values, clip_low=.2, clip_high=.2)
        # Per prefix: -(min(1.5,1.2) + min(-1.5,-1.2))/4 = .075.
        # The other two equally weighted response rows contribute another .075.
        self.assertAlmostEqual(loss.item(), .15)
        loss.backward()
        self.assertTrue(torch.all(values["prefix"].grad > 0))
        self.assertTrue(torch.all(values["good"].grad == 0))

    def test_manual_loss_plan_cannot_relabel_sources_or_change_denominators(self):
        plan = self.plan()
        for terms in ((replace(plan.terms[0], source_id="heldout"), *plan.terms[1:]),
                      (replace(plan.terms[0], weight=.5), *plan.terms[1:]),
                      (*plan.terms, plan.terms[0]), plan.terms[1:], list(plan.terms)):
            with self.assertRaises(ValueError):
                replace(plan, terms=terms).validate()

    def test_response_mean_matches_literal_duplicated_leaf_forward(self):
        plan, values = self.plan(), self.logps(1.12)
        loss = clipped_loss(plan, values, clip_low=.2, clip_high=.28)
        reference = sum(-torch.minimum((values[e] + 2).exp() * a,
            (values[e] + 2).exp().clamp(.8, 1.28) * a).mean() / 4
            for a, path in ((1., ("prefix", "good")), (-1., ("prefix", "bad"))) for e in path)
        self.assertTrue(torch.equal(loss, reference))
        self.assertEqual(len(plan.terms), 4)
        self.assertEqual(len(values), 3)

    def test_microbatch_and_uneven_dp4_gradients_equal_global_reference(self):
        plan, global_values = self.plan(), self.logps(1.15)
        clipped_loss(plan, global_values, clip_low=.2, clip_high=.28).backward()
        shards = (("prefix",), ("good", "bad"), (), ())
        rank_values = self.logps(1.15)
        # DDP mean requires world_size scaling; zero-edge ranks contribute zero.
        rank_losses = []
        for shard in shards:
            if shard:
                rank_losses.append(4 * clipped_loss(plan, {k: rank_values[k] for k in shard},
                    clip_low=.2, clip_high=.28, edge_subset=frozenset(shard)))
        (sum(rank_losses) / 4).backward()
        for e in global_values:
            torch.testing.assert_close(global_values[e].grad, rank_values[e].grad, rtol=0, atol=1e-15)

    def test_zero_local_coefficient_exact_baseline_identity(self):
        plan = replace(self.plan(), arm_id="continuation_q")
        self.assertIs(append_local_terms(plan, [{"bad": float("nan")}], coefficient=0), plan)

    def test_local_candidate_once_and_prefix_not_locally_supervised(self):
        plan = replace(self.plan(), arm_id="continuation_q")
        local = append_local_terms(plan, [{"good": .25, "bad": -.25}], coefficient=.5)
        terms = [t for t in local.terms if t.component == "local"]
        self.assertEqual([t.edge_id for t in terms], ["good", "bad"])
        self.assertEqual([t.weight for t in terms], [.25, .25])
        with self.assertRaises(ValueError):
            append_local_terms(plan, [{"prefix": .25, "bad": -.25}], coefficient=.5)

    def test_invalid_and_heldout_leaves_do_not_enter_denominator(self):
        graph = graph_fixture()
        graph = replace(graph, leaves=(*graph.leaves, Leaf("missing", "task", "tree", (), "missing", None, group_id="task-group")),
                        planned_leaf_ids=("a", "b", "missing"))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="response_mean")
        self.assertEqual([t.weight for t in plan.terms], [.25] * 4)
        with self.assertRaises(ValueError):
            leaf_loss_plan(graph, {"a": 1., "b": -1., "missing": 0.}, arm_id="tree_grpo", reduction="response_mean")

    def test_token_mean_and_trajectory_mean_are_explicit(self):
        graph = graph_fixture()
        longer = replace(graph.edges[1], response_ids=(2, 3, 4, 5), sampled_mask=(True,) * 4, old_log_probs=(-2.,) * 4)
        graph = replace(graph, edges=(graph.edges[0], longer, graph.edges[2]))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="token_mean")
        self.assertEqual([t.weight for t in plan.terms], [.2, .4, .2, .2])
        self.assertAlmostEqual(sum(t.weight for t in plan.terms), 1)
        with self.assertRaises(ValueError):
            leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="unique_edge")

    def test_dual_clip_and_standard_clip_differ_for_bad_extreme_ratios(self):
        values = self.logps(4.)
        standard = clipped_loss(self.plan(), values, clip_low=.2, clip_high=.2)
        dual = clipped_loss(self.plan(), values, clip_low=.2, clip_high=.2, dual_clip=3.)
        self.assertAlmostEqual(standard.item(), 1.4)
        self.assertAlmostEqual(dual.item(), .9)

    def test_masked_tokens_zero_gradient_and_detached_old_scores(self):
        graph = graph_fixture()
        graph = replace(graph, edges=tuple(replace(e, sampled_mask=(True, False)) for e in graph.edges))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="response_mean")
        values = self.logps(1.1)
        clipped_loss(plan, values, clip_low=.2, clip_high=.28).backward()
        for value in values.values():
            self.assertEqual(value.grad[1].item(), 0.)

    def test_nonfinite_and_overflow_fail_instead_of_silent_ratio_clamp(self):
        for invalid in (float("nan"), float("inf"), 1.):
            values = self.logps()
            values["prefix"] = torch.tensor([invalid, -2.])
            with self.assertRaises(ValueError):
                clipped_loss(self.plan(), values, clip_low=.2, clip_high=.2)
        graph = graph_fixture()
        graph = replace(graph, edges=tuple(replace(e, old_log_probs=(-1000.,) * 2) for e in graph.edges))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="response_mean")
        with self.assertRaises(ValueError):
            clipped_loss(plan, self.logps(), clip_low=.2, clip_high=.2)

    def test_sampled_token_logps_and_finite_difference_gradient(self):
        logits = torch.tensor([[.2, -.4, .1], [.6, -.2, -.8]], dtype=torch.float64, requires_grad=True)
        actual = response_log_probs(logits, (1, 2), temperature=.8)
        reference = (logits / .8).log_softmax(-1)[torch.arange(2), torch.tensor([1, 2])]
        torch.testing.assert_close(actual, reference)
        self.assertTrue(torch.autograd.gradcheck(lambda x: response_log_probs(x, (1, 2), temperature=.8), (logits,)))

    def test_tiny_policy_update_rewards_good_candidate(self):
        # End-to-end CPU optimizer check; this is not a TP/DP GPU acceptance test.
        graph = graph_fixture()
        logits = torch.nn.Parameter(torch.full((3, 2, 6), 0., dtype=torch.float64))
        old = -math.log(6.)
        graph = replace(graph, edges=tuple(replace(e, old_log_probs=(old, old)) for e in graph.edges))
        plan = leaf_loss_plan(graph, {"a": 1., "b": -1.}, arm_id="tree_grpo", reduction="response_mean")
        optimizer = torch.optim.SGD([logits], lr=.1)
        values = {e.edge_id: response_log_probs(logits[i], e.response_ids, temperature=1.) for i, e in enumerate(graph.edges)}
        loss = clipped_loss(plan, values, clip_low=.2, clip_high=.2)
        loss.backward(); optimizer.step()
        after = [response_log_probs(logits[i], e.response_ids, temperature=1.).mean().item() for i, e in enumerate(graph.edges)]
        self.assertAlmostEqual(after[0], old)
        self.assertGreater(after[1], old)
        self.assertLess(after[2], old)


if __name__ == "__main__":
    unittest.main()
