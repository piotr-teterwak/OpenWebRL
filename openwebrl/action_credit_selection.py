"""Outcome-blind leaf selection receipts for one task/group's sampled forest.

The full graph is retained, including generated-but-unselected training leaves,
holdout leaves and failed/missing slots. Selection and holdout are distinct roles.
The fixed schedule/seed policy must be pinned in the campaign BEFORE dispatch.
Full inventory hashes bind labels AFTER collection but never seed the selection.

Terminal-to-leaf mapping is explicit collector provenance, not inferred physical
proof: post-action SamplingNodes and pre-action ResponseEdges represent different
states. This module checks identities/membership but cannot prove browser replay
or authenticate a mapping. A single schedule/group is supported; an online
multigroup collector and its independent acceptance checks remain out of scope.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .action_credit import ResponseGraph, digest
from .action_credit_receipts import ReplayReceipt
from .action_credit_sampling import (
    RoundSeal, SamplingNode, TreeSchedule, forest_sha256, select_leaves,
)


def inventory_sha256(graph: ResponseGraph) -> str:
    """Canonical complete graph identity, independent of storage order."""
    graph.validate()
    return digest(dict(contract=graph.contract.sha256,
        edges=[asdict(e) for e in sorted(graph.edges, key=lambda e: e.edge_id)],
        leaves=[asdict(l) for l in sorted(graph.leaves, key=lambda l: l.leaf_id)],
        planned_leaf_ids=sorted(graph.planned_leaf_ids)))


@dataclass(frozen=True)
class TrainingSelectionReceipt:
    schedule: TreeSchedule
    nodes: tuple[SamplingNode, ...]
    replay_receipts: tuple[ReplayReceipt, ...]
    seals: tuple[RoundSeal, ...]
    terminal_to_leaf: tuple[tuple[str, str], ...]
    inventory_sha256: str
    generated_leaf_ids: tuple[str, ...]
    selected_leaf_ids: tuple[str, ...]

    @property
    def policy(self):
        s = self.schedule
        return dict(mode="tree_uniform_terminal_k_v1", m=s.m, n=s.n, l=s.l, k=s.k,
            seed=s.seed, rng="python_random_keyed_per_tree_v1",
            candidate_pool="generated_training_terminal_nodes", selection="without_replacement",
            task_id=s.task_id, group_id=s.group_id)

    @property
    def policy_sha256(self):
        return digest(self.policy)

    @property
    def schedule_sha256(self):
        return self.schedule.sha256

    @property
    def forest_sha256(self):
        return forest_sha256(self.nodes, before_round=self.schedule.l + 1)

    @property
    def seals_sha256(self):
        return digest([asdict(s) for s in self.seals])

    @property
    def mapping_sha256(self):
        return digest(self.terminal_to_leaf)

    @property
    def sha256(self):
        return digest(dict(schema_version=1, **asdict(self)))


def _selection(graph, schedule, nodes, replay_receipts, seals, terminal_to_leaf):
    if not isinstance(graph, ResponseGraph) or not isinstance(schedule, TreeSchedule):
        raise ValueError("typed graph and tree schedule required")
    graph.validate()
    if schedule.contract.sha256 != graph.contract.sha256:
        raise ValueError("sampler and training collection contracts differ")
    if not isinstance(terminal_to_leaf, Mapping):
        raise ValueError("explicit terminal-node to graph-leaf mapping required")
    selected = select_leaves(schedule, nodes, replay_receipts, seals=seals)
    terminal_ids = {n.node_id for n in nodes if n.terminal}
    mapping = dict(terminal_to_leaf)
    if (set(mapping) != terminal_ids or any(not isinstance(v, str) for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("mapping must bijectively cover every generated terminal node")
    leaves, node_by_id = {l.leaf_id: l for l in graph.leaves}, {n.node_id: n for n in nodes}
    for terminal, leaf_id in mapping.items():
        leaf, node = leaves.get(leaf_id), node_by_id[terminal]
        if (leaf is None or leaf.split != "train"
                or (leaf.task_id, leaf.group_id, leaf.tree_id) !=
                   (schedule.task_id, schedule.group_id, node.tree_id)):
            raise ValueError("terminal mapping changes a task/group/tree or leaks holdout")
    # A successful outcome necessarily has a generated terminal artifact. Failed
    # attempts without such an artifact remain in the full inventory, not in K's
    # candidate pool. Failure labels never choose among existing terminal nodes.
    if any(l.split == "train" and l.status == "valid" and l.leaf_id not in mapping.values()
           for l in graph.leaves):
        raise ValueError("valid training leaf omitted from generated terminal inventory")
    picked = tuple(sorted(mapping[n] for s in selected for n in s.selected_leaf_ids))
    generated = tuple(sorted(mapping.values()))
    return generated, picked


def make_training_selection(graph: ResponseGraph, *, schedule: TreeSchedule,
                            nodes: tuple[SamplingNode, ...], replay_receipts,
                            seals: tuple[RoundSeal, ...], terminal_to_leaf) -> TrainingSelectionReceipt:
    """Freeze the complete sampler proof and explicit graph mapping, without pruning."""
    generated, picked = _selection(graph, schedule, nodes, replay_receipts, seals, terminal_to_leaf)
    return TrainingSelectionReceipt(schedule, tuple(sorted(nodes, key=lambda n: n.node_id)),
        tuple(replay_receipts[k] for k in sorted(replay_receipts)),
        tuple(sorted(seals, key=lambda s: s.round_index)), tuple(sorted(terminal_to_leaf.items())),
        inventory_sha256(graph), generated, picked)


def validate_selection(graph: ResponseGraph, selection: TrainingSelectionReceipt) -> tuple[str, ...]:
    """Recompute outcome-blind selection and return selected train IDs, even invalid ones."""
    if not isinstance(selection, TrainingSelectionReceipt):
        raise ValueError("typed training selection receipt required")
    s = selection
    for value in (s.nodes, s.replay_receipts, s.seals, s.terminal_to_leaf,
                  s.generated_leaf_ids, s.selected_leaf_ids):
        if not isinstance(value, tuple):
            raise ValueError("selection proof must be immutable")
    if (any(not isinstance(r, ReplayReceipt) for r in s.replay_receipts)
            or len({r.edge_id for r in s.replay_receipts}) != len(s.replay_receipts)
            or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in s.terminal_to_leaf)
            or len(dict(s.terminal_to_leaf)) != len(s.terminal_to_leaf)):
        raise ValueError("duplicate or malformed selection proof identities")
    if s.inventory_sha256 != inventory_sha256(graph):
        raise ValueError("complete training/holdout inventory changed after selection")
    generated, picked = _selection(graph, s.schedule, s.nodes,
        {r.edge_id: r for r in s.replay_receipts}, s.seals, dict(s.terminal_to_leaf))
    if s.generated_leaf_ids != generated or s.selected_leaf_ids != picked:
        raise ValueError("selected/generated IDs differ from the pinned outcome-blind sampler")
    return picked
