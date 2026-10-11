"""Deterministic planning tests; all replay receipts are explicitly synthetic."""
from dataclasses import replace
import unittest

from openwebrl.action_credit import CollectionContract, digest
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_sampling import (
    RequestCompletion, RoundSeal, SamplingNode, TreeSchedule, forest_sha256,
    plan_initial, plan_round, select_leaves,
)


def schedule(m=2, n=5, l=2, k=1):
    c = CollectionContract(0, *[digest(s) for s in
        ("policy", "tokenizer", "processor", "tasks", "judge", "generation")],
        .8, ("task",), ("test",))
    return TreeSchedule(c, "task", "group", m, n, l, k, 42)


def node(s, identity, tree, parent=None, remaining=3, terminal=False, round=0):
    context = digest("initial" if parent is None else "context:" + identity)
    state = digest("initial-state" if parent is None else "state:" + identity)
    snapshot = digest("synthetic snapshot manifest:" + identity)
    receipt = ReplayReceipt(identity, s.contract.sha256, state, context, remaining,
                            digest("synthetic verifier"), snapshot, True)
    n = SamplingNode(identity, "task", tree, parent, s.contract.policy_sha256,
                     context, state, snapshot, remaining, terminal, round, receipt.sha256)
    return n, receipt


def forest(s):
    nodes, receipts = [], {}
    for i, tree in enumerate(s.tree_ids):
        root, rr = node(s, f"root-{i}", tree)
        leaf, lr = node(s, f"leaf-{i}", tree, root.node_id, remaining=2, terminal=True)
        nodes.extend((root, leaf)); receipts.update({root.node_id: rr, leaf.node_id: lr})
    return tuple(nodes), receipts


def seal(s, nodes, requests, completed=()):
    completed = dict(completed)
    return RoundSeal(s.sha256, requests[0].round_index,
        forest_sha256(nodes, before_round=requests[0].round_index),
        tuple(RequestCompletion(r.request_id, "valid" if r.request_id in completed else "missing",
                                completed.get(r.request_id)) for r in requests))


class SamplingTests(unittest.TestCase):
    def test_initial_m_independent_seeds_ids_and_cost_slots(self):
        s = schedule()
        root, receipt = node(s, "initial-source", "initial", round=-1)
        requests = plan_initial(s, root, receipt)
        self.assertEqual(len(requests), s.m)
        self.assertEqual(tuple(r.tree_id for r in requests), s.tree_ids)
        self.assertEqual(len({r.request_id for r in requests}), s.m)
        self.assertEqual(len({r.generation_seed for r in requests}), s.m)
        self.assertEqual([r.cost_slot_id for r in requests], [r.request_id for r in requests])
        self.assertEqual(requests, plan_initial(s, root, receipt))
        self.assertNotEqual(requests, plan_initial(replace(s, seed=43), root, receipt))

    def test_with_replacement_duplicates_are_distinct_continuation_requests(self):
        s = schedule(n=5)
        nodes, receipts = forest(s)  # Initial chains terminated immediately; root still eligible.
        requests = plan_round(s, nodes, receipts, round_index=1)
        self.assertEqual(len(requests), s.m * s.n)
        self.assertEqual(len({r.source_node_id for r in requests}), s.m)
        self.assertEqual(len({r.request_id for r in requests}), s.m * s.n)
        for tree in s.tree_ids:
            self.assertEqual([r.draw_slot for r in requests if r.tree_id == tree], list(range(s.n)))

    def test_seed_and_input_order_invariance_without_global_rng_changes(self):
        import random
        s = schedule()
        nodes, receipts = forest(s)
        expected = plan_round(s, nodes, receipts, round_index=1)
        random.seed(99)
        before = random.getstate()
        self.assertEqual(expected, plan_round(s, tuple(reversed(nodes)), dict(reversed(list(receipts.items()))), round_index=1))
        self.assertEqual(before, random.getstate())

    def test_current_round_nodes_cannot_enter_until_next_round(self):
        s = schedule(m=1, n=30)
        nodes, receipts = forest(s)
        baseline = plan_round(s, nodes, receipts, round_index=1)
        new, receipt = node(s, "new-branch", s.tree_ids[0], "root-0", remaining=2, round=1)
        extended = nodes + (new,)
        receipts = {**receipts, new.node_id: receipt}
        self.assertEqual(plan_round(s, extended, receipts, round_index=1), baseline)
        terminal, tr = node(s, "new-terminal", s.tree_ids[0], new.node_id, remaining=1, terminal=True, round=1)
        extended += (terminal,)
        receipts[terminal.node_id] = tr
        prior = seal(s, extended, baseline, [(baseline[0].request_id, terminal.node_id)])
        second = plan_round(s, extended, receipts, round_index=2, seals=(prior,))
        self.assertIn("new-branch", {r.source_node_id for r in second})
        self.assertNotIn("leaf-0", {r.source_node_id for r in second})

    def test_leaf_sample_without_replacement_preserves_generated_inventory(self):
        s = schedule(m=1, k=2)
        nodes, receipts = forest(s)
        requests = plan_round(s, nodes, receipts, round_index=1)
        completed = []
        for i in range(4):
            leaf, receipt = node(s, "extra-" + str(i), s.tree_ids[0], "root-0", remaining=2,
                                 terminal=True, round=1)
            nodes += (leaf,); receipts[leaf.node_id] = receipt
            completed.append((requests[i].request_id, leaf.node_id))
        first = seal(s, nodes, requests, completed)
        second = seal(s, nodes, plan_round(s, nodes, receipts, round_index=2, seals=(first,)))
        seals = (first, second)
        selected, = select_leaves(s, nodes, receipts, seals=seals)
        self.assertEqual(len(selected.selected_leaf_ids), 2)
        self.assertEqual(len(set(selected.selected_leaf_ids)), 2)
        self.assertEqual(len(selected.generated_leaf_ids), 5)
        self.assertTrue(set(selected.selected_leaf_ids) <= set(selected.generated_leaf_ids))
        self.assertEqual(select_leaves(s, nodes, receipts, seals=seals),
                         select_leaves(s, tuple(reversed(nodes)), receipts, seals=seals))
        self.assertEqual(len(nodes), 6)
        with self.assertRaises(ValueError):
            short = schedule(m=1, k=6, l=0)
            select_leaves(short, *forest(short))

    def test_rounds_cannot_be_skipped_by_initial_only_forests(self):
        s = schedule(m=1)
        nodes, receipts = forest(s)
        with self.assertRaises(ValueError):
            plan_round(s, nodes, receipts, round_index=2)
        with self.assertRaises(ValueError):
            select_leaves(s, nodes, receipts)
        requests = plan_round(s, nodes, receipts, round_index=1)
        good = seal(s, nodes, requests)  # All slots closed as missing, never treated as successes.
        for bad in (replace(good, completions=good.completions[:-1]),
                    replace(good, input_forest_sha256=digest("wrong input forest")),
                    replace(good, schedule_sha256=digest("wrong schedule")),
                    replace(good, completions=(replace(good.completions[0], status="pending"), *good.completions[1:]))):
            with self.subTest(seal=bad), self.assertRaises(ValueError):
                plan_round(s, nodes, receipts, round_index=2, seals=(bad,))
        self.assertEqual(len(plan_round(s, nodes, receipts, round_index=2, seals=(good,))), s.n)

    def test_extra_round_nodes_and_reused_completion_are_rejected(self):
        s = schedule(m=1)
        nodes, receipts = forest(s)
        requests = plan_round(s, nodes, receipts, round_index=1)
        extra, er = node(s, "extra", s.tree_ids[0], "root-0", remaining=2, terminal=True, round=1)
        nodes += (extra,); receipts[extra.node_id] = er
        for completed in ([], [(requests[0].request_id, extra.node_id), (requests[1].request_id, extra.node_id)]):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                plan_round(s, nodes, receipts, round_index=2,
                           seals=(seal(s, nodes, requests, completed),))

    def test_zero_horizon_terminal_allowed_but_not_expandable(self):
        s = schedule(m=1)
        root, rr = node(s, "root", s.tree_ids[0], remaining=1)
        leaf, lr = node(s, "last-action", s.tree_ids[0], "root", remaining=0, terminal=True)
        requests = plan_round(s, (root, leaf), {root.node_id: rr, leaf.node_id: lr}, round_index=1)
        self.assertEqual({r.source_node_id for r in requests}, {"root"})
        with self.assertRaises(ValueError):
            plan_round(s, (root, replace(leaf, terminal=False)), {root.node_id: rr, leaf.node_id: lr}, round_index=1)

    def test_missing_tree_duplicate_node_incomplete_initial_and_wrong_receipt_rejected(self):
        s = schedule()
        nodes, receipts = forest(s)
        for ns, rs in [(nodes[:2], {k: v for k, v in receipts.items() if k.endswith("0")}),
                       (nodes + (nodes[0],), receipts), (nodes, {}),
                       (nodes[::2], {k: v for k, v in receipts.items() if k.startswith("root")}),
                       (nodes, {**receipts, "root-0": replace(receipts["root-0"], accepted=False)})]:
            with self.subTest(nodes=ns), self.assertRaises(ValueError):
                plan_round(s, ns, rs, round_index=1)

    def test_initial_round_rejects_multiple_continuations_per_tree(self):
        s = schedule(m=1)
        nodes, receipts = forest(s)
        for terminal in (True, False):
            branch, receipt = node(s, "extra-initial", s.tree_ids[0], "root-0",
                                   remaining=2, terminal=terminal, round=0)
            with self.subTest(terminal=terminal), self.assertRaisesRegex(ValueError, "nonbranching"):
                plan_round(s, nodes + (branch,), {**receipts, branch.node_id: receipt}, round_index=1)

    def test_initial_round_accepts_a_longer_single_chain(self):
        s = schedule(m=1)
        root, rr = node(s, "root", s.tree_ids[0], remaining=3)
        middle, mr = node(s, "middle", s.tree_ids[0], "root", remaining=2)
        leaf, lr = node(s, "leaf", s.tree_ids[0], "middle", remaining=1, terminal=True)
        requests = plan_round(s, (leaf, root, middle), {"root": rr, "middle": mr, "leaf": lr}, round_index=1)
        self.assertEqual(len(requests), s.n)
        self.assertTrue({r.source_node_id for r in requests} <= {"root", "middle"})

    def test_stale_policy_task_snapshot_context_horizon_and_round_rejected(self):
        s = schedule()
        nodes, receipts = forest(s)
        for changes in ({"policy_sha256": digest("stale")}, {"task_id": "other"},
                        {"snapshot_sha256": digest("other snapshot")}, {"context_sha256": digest("other context")},
                        {"remaining_turns": 4}, {"created_round": 3}, {"terminal": 1},
                        {"replay_receipt_sha256": digest("other receipt")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_round(s, (replace(nodes[0], **changes), *nodes[1:]), receipts, round_index=1)

    def test_cross_tree_parent_cycles_and_terminal_parent_rejected(self):
        s = schedule()
        nodes, receipts = forest(s)
        for changed in (replace(nodes[1], parent_node_id="root-1"),
                        replace(nodes[0], parent_node_id="leaf-0"),
                        replace(nodes[1], parent_node_id=nodes[1].node_id)):
            ns = tuple(changed if n.node_id == changed.node_id else n for n in nodes)
            with self.subTest(node=changed), self.assertRaises(ValueError):
                plan_round(s, ns, receipts, round_index=1)
        extra, receipt = node(s, "after-stop", s.tree_ids[0], "leaf-0", remaining=1, round=1)
        with self.assertRaises(ValueError):
            plan_round(s, nodes + (extra,), {**receipts, extra.node_id: receipt}, round_index=1)

    def test_zero_round_schedule_and_strict_frozen_parameter_types(self):
        s = schedule(l=0)
        nodes, receipts = forest(s)
        self.assertEqual(len(select_leaves(s, nodes, receipts)), s.m)
        for r in (0, 1, True):
            with self.assertRaises(ValueError):
                plan_round(s, nodes, receipts, round_index=r)
        for changes in ({"m": 0}, {"n": True}, {"l": -1}, {"k": 0},
                        {"seed": -1}, {"task_id": "test"}, {"group_id": " g "}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(s, **changes).sha256


if __name__ == "__main__":
    unittest.main()
