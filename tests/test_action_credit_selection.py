"""Synthetic sampler-to-objective selection checks; no physical replay evidence."""
from dataclasses import replace
import importlib.util
import unittest

from openwebrl.action_credit import CollectionContract, Leaf, ResponseEdge, ResponseGraph, digest
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_sampling import (
    SamplingNode, TreeSchedule, RoundSeal, RequestCompletion, forest_sha256, plan_round,
)
from openwebrl.action_credit_selection import (
    make_training_selection, validate_selection,
)


def graph_receipts(graph):
    return {e.edge_id: ReplayReceipt(e.edge_id, e.contract_sha256, e.state_sha256,
        e.context_sha256, e.remaining_turns, digest("cpu-test-verifier"), digest("cpu-test-artifact"), True)
        for e in graph.edges}


def selection_fixture():
    """Return full graph and K=2 receipt: three generated fit leaves plus one holdout."""
    c = CollectionContract(0, *[digest(s) for s in
        ("policy", "tokenizer", "processor", "tasks", "judge", "generation")],
        .8, ("task",), ("test",))
    schedule = TreeSchedule(c, "task", "g", 1, 2, 1, 2, 42)
    tree = schedule.tree_ids[0]
    def node(identity, parent, terminal, round):
        remaining = 3 if parent is None else 2
        receipt = ReplayReceipt(identity, c.sha256, digest("state:" + identity),
            digest("context:" + identity), remaining, digest("synthetic-verifier"), digest("snapshot:" + identity), True)
        return SamplingNode(identity, "task", tree, parent, c.policy_sha256,
            receipt.context_sha256, receipt.state_sha256, receipt.evidence_sha256,
            remaining, terminal, round, receipt.sha256), receipt
    root, rr = node("root", None, False, 0)
    initial, ir = node("initial-terminal", "root", True, 0)
    nodes, receipts = (root, initial), {root.node_id: rr, initial.node_id: ir}
    requests = plan_round(schedule, nodes, receipts, round_index=1)
    completions = []
    for i, request in enumerate(requests):
        terminal, receipt = node(f"expanded-terminal-{i}", "root", True, 1)
        nodes += (terminal,); receipts[terminal.node_id] = receipt
        completions.append(RequestCompletion(request.request_id, "valid", terminal.node_id))
    seal = RoundSeal(schedule.sha256, 1, forest_sha256(nodes, before_round=1), tuple(completions))
    edges, leaves, mapping = [], [], {}
    terminal_ids = sorted(n.node_id for n in nodes if n.terminal)
    for i, terminal_id in enumerate(terminal_ids + ["independent-holdout"]):
        holdout = i == len(terminal_ids)
        edge_id, leaf_id = f"edge-{i}", f"leaf-{i}"
        t, task, group = ("holdout-tree", "test", "holdout-group") if holdout else (tree, "task", "g")
        edges.append(ResponseEdge(edge_id, task, t, None, c.sha256, (0, 1), (),
            (2, 3), (True, True), (-2., -2.), c.policy_sha256, digest("edge-state"), digest("placeholder"), 3))
        leaves.append(Leaf(leaf_id, task, t, (edge_id,), "valid", float(i != 0),
                           split="holdout" if holdout else "train", group_id=group))
        if not holdout:
            mapping[terminal_id] = leaf_id
    graph = ResponseGraph(c, tuple(edges), tuple(leaves), tuple(l.leaf_id for l in leaves))
    edge_receipts = graph_receipts(graph)
    graph = replace(graph, edges=tuple(replace(e, replay_evidence_sha256=edge_receipts[e.edge_id].sha256)
                                       for e in graph.edges))
    kwargs = dict(schedule=schedule, nodes=nodes, replay_receipts=receipts, seals=(seal,), terminal_to_leaf=mapping)
    selection = make_training_selection(graph, **kwargs)
    # Make selected outcomes 0/1, unselected 1. Selection cannot depend on these labels.
    selected = selection.selected_leaf_ids
    graph = replace(graph, leaves=tuple(replace(l, reward=0. if l.leaf_id == selected[0] else 1.) for l in graph.leaves))
    return graph, make_training_selection(graph, **kwargs)


def remake(graph, selection, **changes):
    kwargs = dict(schedule=selection.schedule, nodes=selection.nodes,
        replay_receipts={r.edge_id: r for r in selection.replay_receipts},
        seals=selection.seals, terminal_to_leaf=dict(selection.terminal_to_leaf))
    kwargs.update(changes)
    return make_training_selection(graph, **kwargs)


class SelectionTests(unittest.TestCase):
    def test_full_inventory_selected_and_holdout_remain_distinct(self):
        graph, selection = selection_fixture()
        self.assertEqual(validate_selection(graph, selection), selection.selected_leaf_ids)
        self.assertEqual(len(graph.leaves), 4)
        self.assertEqual(len(selection.generated_leaf_ids), 3)
        self.assertEqual(len(selection.selected_leaf_ids), 2)
        self.assertTrue(set(selection.selected_leaf_ids) < set(selection.generated_leaf_ids))
        self.assertEqual(len([l for l in graph.leaves if l.split == "holdout"]), 1)
        self.assertTrue(all(l.split == "train" for l in graph.leaves if l.leaf_id in selection.generated_leaf_ids))
        self.assertEqual(selection.policy_sha256, digest(selection.policy))
        self.assertNotIn("selected_leaf_ids", selection.policy)

    def test_recompute_ignores_rewards_but_inventory_hash_binds_them(self):
        graph, selection = selection_fixture()
        changed = replace(graph, leaves=tuple(replace(l, reward=1-l.reward) for l in graph.leaves))
        new = remake(changed, selection)
        self.assertEqual(new.selected_leaf_ids, selection.selected_leaf_ids)
        self.assertEqual(new.policy_sha256, selection.policy_sha256)
        self.assertNotEqual(new.inventory_sha256, selection.inventory_sha256)
        with self.assertRaises(ValueError):
            validate_selection(changed, selection)

    def test_selected_invalid_outcome_remains_selected_without_imputation(self):
        graph, selection = selection_fixture()
        changed = replace(graph, leaves=tuple(replace(l, status="invalid", reward=None)
            if l.leaf_id == selection.selected_leaf_ids[0] else l for l in graph.leaves))
        new = remake(changed, selection)
        self.assertEqual(new.selected_leaf_ids, selection.selected_leaf_ids)
        self.assertEqual(validate_selection(changed, new), selection.selected_leaf_ids)

    def test_dropped_invalid_missing_or_holdout_inventory_is_detected(self):
        graph, selection = selection_fixture()
        missing = Leaf("missing", "task", selection.schedule.tree_ids[0], (), "missing", None, group_id="g")
        full = replace(graph, leaves=graph.leaves + (missing,), planned_leaf_ids=graph.planned_leaf_ids + ("missing",))
        receipt = remake(full, selection)
        self.assertEqual(receipt.selected_leaf_ids, selection.selected_leaf_ids)
        with self.assertRaises(ValueError):
            validate_selection(graph, receipt)

    def test_selected_ids_proof_and_omitted_terminal_cannot_be_forged(self):
        graph, selection = selection_fixture()
        for changed in (replace(selection, selected_leaf_ids=selection.generated_leaf_ids),
                        replace(selection, generated_leaf_ids=selection.selected_leaf_ids),
                        replace(selection, seals=()),
                        replace(selection, schedule=replace(selection.schedule, seed=43)),
                        replace(selection, terminal_to_leaf=selection.terminal_to_leaf[:-1])):
            with self.subTest(selection=changed), self.assertRaises(ValueError):
                validate_selection(graph, changed)

    def test_mapping_cannot_cross_task_group_tree_or_holdout(self):
        graph, selection = selection_fixture()
        target = selection.generated_leaf_ids[0]
        for changes in ({"group_id": "different"}, {"tree_id": "different"}, {"split": "holdout"}):
            changed = replace(graph, leaves=tuple(replace(l, **changes) if l.leaf_id == target else l for l in graph.leaves))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                remake(changed, selection)

    def test_inventory_and_proof_are_order_independent_and_immutable(self):
        graph, selection = selection_fixture()
        permuted = replace(graph, edges=tuple(reversed(graph.edges)), leaves=tuple(reversed(graph.leaves)),
                           planned_leaf_ids=tuple(reversed(graph.planned_leaf_ids)))
        self.assertEqual(validate_selection(permuted, selection), selection.selected_leaf_ids)
        self.assertEqual(remake(permuted, selection, nodes=tuple(reversed(selection.nodes))), selection)
        with self.assertRaises(AttributeError):
            selection.selected_leaf_ids = ()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "integrated fixture requires optional torch")
    def test_selected_ids_determine_advantage_pool_and_loss_denominator(self):
        from openwebrl.action_credit_objective import build_objective
        from test_action_credit_binding import config, make_binding
        graph, selection = selection_fixture()
        cfg = config("tree_outcome_sampling")
        prepared = build_objective(graph, cfg, selection=selection,
            collection_binding=make_binding(graph, cfg, selection=selection), replay_receipts=graph_receipts(graph))
        self.assertIs(prepared.plan.graph, graph)
        self.assertEqual(len(prepared.plan.graph.leaves), 4)
        self.assertEqual({t.source_id for t in prepared.plan.terms}, set(selection.selected_leaf_ids))
        self.assertEqual([t.weight for t in prepared.plan.terms], [.5, .5])
        self.assertEqual(prepared.diagnostics[0].groups[0].normalization_count, 2)
        self.assertEqual(prepared.diagnostics[0].groups[0].mean, .5)


if __name__ == "__main__":
    unittest.main()
