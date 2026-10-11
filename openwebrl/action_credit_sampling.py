"""Deterministic offline Tree-GRPO request planning, not a browser sampler.

Pinned AMAP-ML/Tree-GRPO commit 19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59:
search_r1/llm_agent/tree_node.py SHA256
643d2ee589fce25b09e973287f8aca93a136a31d8d739f6dd928c69c2afb6b96.
get_expand_node draws WITH replacement from root + nonterminal descendants;
duplicates are separate continuation requests. generation_ts.py selects a whole
round before generation; new nodes enter the next round. sample_leaf samples K
terminal leaves without replacement. Neither selection uses outcome rewards.

Deviations: canonical ID order and keyed per-tree/round RNG replace upstream BFS
order/global RNG; sampling laws match, bitwise upstream seed mapping does not.
We preserve all leaf IDs instead of upstream destructive pruning. Snapshot means
the receipt-bound evidence/snapshot-manifest artifact, not verified browser bytes.
Receipts prove structural binding only. The caller must verify replay physically
and enforce completed-round barriers/attempt accounting in its runtime ledger.
Nodes represent post-action contexts; ResponseEdge pre-action contexts cannot be
silently reused as them. No actor execution, snapshot restore or cost is incurred.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random
from collections.abc import Mapping
from openwebrl.action_credit import CollectionContract, digest
from openwebrl.action_credit_receipts import ReplayReceipt


@dataclass(frozen=True)
class SamplingNode:
    node_id: str
    task_id: str
    tree_id: str
    parent_node_id: str | None
    policy_sha256: str
    context_sha256: str
    state_sha256: str
    snapshot_sha256: str
    remaining_turns: int
    terminal: bool
    created_round: int
    replay_receipt_sha256: str


@dataclass(frozen=True)
class TreeSchedule:
    contract: CollectionContract
    task_id: str
    group_id: str
    m: int
    n: int
    l: int
    k: int
    seed: int

    @property
    def sha256(self):
        self.contract.validate()
        for value in (self.task_id, self.group_id):
            _identity(value)
        if self.task_id not in self.contract.train_task_ids:
            raise ValueError("sampling task outside training split")
        if any(type(v) is not int or v < limit for v, limit in
               ((self.m, 1), (self.n, 1), (self.l, 0), (self.k, 1), (self.seed, 0))):
            raise ValueError("invalid frozen tree schedule")
        return digest({**self.__dict__, "contract": self.contract.sha256})

    @property
    def tree_ids(self):
        return tuple(digest([self.sha256, "tree", i]) for i in range(self.m))


@dataclass(frozen=True)
class SamplingRequest:
    request_id: str
    tree_id: str
    source_node_id: str
    context_sha256: str
    snapshot_sha256: str
    remaining_turns: int
    round_index: int
    draw_slot: int
    generation_seed: int
    cost_slot_id: str


@dataclass(frozen=True)
class LeafSelection:
    tree_id: str
    selected_leaf_ids: tuple[str, ...]
    generated_leaf_ids: tuple[str, ...]
    statistic_pool: str = "uniform_k_terminal_leaves_per_tree"


@dataclass(frozen=True)
class RequestCompletion:
    """Closed slot disposition; missing means no artifact, NEVER still pending."""
    request_id: str
    status: str
    terminal_node_id: str | None = None


@dataclass(frozen=True)
class RoundSeal:
    schedule_sha256: str
    round_index: int
    input_forest_sha256: str
    completions: tuple[RequestCompletion, ...]


def forest_sha256(nodes, *, before_round):
    """Hash every pre-round node, including nodes not selected for expansion."""
    return digest([asdict(n) for n in sorted(nodes, key=lambda n: n.node_id)
                   if n.created_round < before_round])


def _identity(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("noncanonical identity")


def _bind(schedule, node, receipt):
    if not isinstance(node, SamplingNode) or not isinstance(receipt, ReplayReceipt):
        raise ValueError("typed node and replay receipt required")
    receipt.validate()
    _identity(node.tree_id)
    if (type(node.terminal) is not bool or type(node.created_round) is not int
            or type(node.remaining_turns) is not int or node.remaining_turns < 0
            or (not node.terminal and node.remaining_turns == 0)):
        raise ValueError("invalid terminal declaration/round/horizon")
    if (not receipt.accepted or node.task_id != schedule.task_id
            or node.policy_sha256 != schedule.contract.policy_sha256
            or receipt.contract_sha256 != schedule.contract.sha256
            or (receipt.edge_id, receipt.context_sha256, receipt.state_sha256,
                receipt.evidence_sha256, receipt.remaining_turns, receipt.sha256) !=
               (node.node_id, node.context_sha256, node.state_sha256,
                node.snapshot_sha256, node.remaining_turns, node.replay_receipt_sha256)):
        raise ValueError("unverified, stale, or mismatched source snapshot")


def _request(schedule, node, tree_id, round_index, slot):
    identity = digest([schedule.sha256, node.node_id, tree_id, round_index, slot,
                       node.context_sha256, node.snapshot_sha256, node.replay_receipt_sha256])
    return SamplingRequest(identity, tree_id, node.node_id, node.context_sha256,
        node.snapshot_sha256, node.remaining_turns, round_index, slot,
        int(digest([identity, "generation"])[:15], 16), identity)


def plan_initial(schedule: TreeSchedule, source: SamplingNode, receipt: ReplayReceipt):
    """M independent initial continuations, each with its own tree/cost/seed slot."""
    schedule.sha256
    _bind(schedule, source, receipt)
    if source.parent_node_id is not None or source.created_round != -1 or source.terminal:
        raise ValueError("initial source must be an unconsumed nonterminal root snapshot")
    return tuple(_request(schedule, source, t, 0, i) for i, t in enumerate(schedule.tree_ids))


def _forest(schedule, nodes, receipts, cutoff):
    schedule.sha256
    if not isinstance(nodes, tuple) or not isinstance(receipts, Mapping):
        raise ValueError("immutable nodes and receipt mapping required")
    ids = [n.node_id for n in nodes if isinstance(n, SamplingNode)]
    if len(ids) != len(nodes) or len(set(ids)) != len(nodes) or set(ids) != set(receipts):
        raise ValueError("duplicate node or receipt manifest mismatch")
    indexed, roots = dict(zip(ids, nodes)), {}
    for node in nodes:
        _bind(schedule, node, receipts[node.node_id])
        if node.tree_id not in schedule.tree_ids or not 0 <= node.created_round <= cutoff:
            raise ValueError("unexpected tree/future round")
        if node.parent_node_id is None:
            if node.tree_id in roots or node.created_round != 0 or node.terminal:
                raise ValueError("each tree needs exactly one initial nonterminal root")
            roots[node.tree_id] = node
        else:
            parent = indexed.get(node.parent_node_id)
            if (parent is None or parent.tree_id != node.tree_id or parent.terminal
                    or parent.created_round > node.created_round
                    or parent.remaining_turns != node.remaining_turns + 1):
                raise ValueError("broken/cyclic ancestry or inconsistent horizon")
    if set(roots) != set(schedule.tree_ids):
        raise ValueError("missing planned tree")
    if len({(r.context_sha256, r.state_sha256, r.remaining_turns) for r in roots.values()}) != 1:
        raise ValueError("initial trees do not share the same task starting state")
    for tree in schedule.tree_ids:
        initial = [n for n in nodes if n.tree_id == tree and n.created_round == 0]
        children = {n.node_id: 0 for n in initial}
        for n in initial:
            if n.parent_node_id is not None:
                children[n.parent_node_id] += 1
        if (sum(n.terminal for n in initial) != 1
                or any(children[n.node_id] != (0 if n.terminal else 1) for n in initial)):
            raise ValueError("each initial tree must be one complete nonbranching continuation")
    return nodes


def _requests(schedule, nodes, round_index):
    requests = []
    for tree in schedule.tree_ids:
        pool = sorted((n for n in nodes if n.tree_id == tree and not n.terminal
                       and n.created_round < round_index), key=lambda n: n.node_id)
        rng = random.Random(int(digest([schedule.sha256, tree, round_index, "expand"]), 16))
        requests.extend(_request(schedule, node, tree, round_index, slot)
                        for slot, node in enumerate(rng.choices(pool, k=schedule.n)))
    return tuple(requests)


def _seals(schedule, nodes, seals, last_round):
    if (not isinstance(seals, tuple) or any(not isinstance(s, RoundSeal) for s in seals)
            or len(seals) != last_round or {s.round_index for s in seals} != set(range(1, last_round + 1))):
        raise ValueError("exact completed prior-round seals required")
    indexed, assigned = {n.node_id: n for n in nodes}, set()
    for seal in seals:
        r = seal.round_index
        covered = set()
        expected = {q.request_id: q for q in _requests(schedule, nodes, r)}
        if (type(r) is not int or seal.schedule_sha256 != schedule.sha256
                or seal.input_forest_sha256 != forest_sha256(nodes, before_round=r)
                or not isinstance(seal.completions, tuple)
                or any(not isinstance(c, RequestCompletion) for c in seal.completions)
                or len(seal.completions) != len(expected)
                or {c.request_id for c in seal.completions} != set(expected)):
            raise ValueError("round seal does not bind the exact frozen request manifest")
        for c in seal.completions:
            if c.status not in ("valid", "invalid", "missing"):
                raise ValueError("request is not terminally dispositioned")
            if c.terminal_node_id is None:
                if c.status == "valid":
                    raise ValueError("valid completion lacks its terminal artifact")
                continue
            leaf = indexed.get(c.terminal_node_id)
            if (c.status == "missing" or leaf is None or not leaf.terminal
                    or leaf.created_round != r or leaf.node_id in assigned):
                raise ValueError("missing/duplicate/wrong-round terminal completion")
            assigned.add(leaf.node_id)
            source = expected[c.request_id].source_node_id
            ancestor = leaf
            while ancestor.node_id != source and ancestor.parent_node_id is not None:
                if ancestor.created_round != r or ancestor.node_id in covered:
                    raise ValueError("continuations must contain distinct newly sampled nodes")
                covered.add(ancestor.node_id)
                ancestor = indexed[ancestor.parent_node_id]
            if ancestor.node_id != source or leaf.node_id == source:
                raise ValueError("completion does not descend from requested source")
        if covered != {n.node_id for n in nodes if n.created_round == r}:
            raise ValueError("round contains nodes without a bound completion")


def plan_round(schedule: TreeSchedule, nodes: tuple[SamplingNode, ...], receipts,
               *, round_index: int, seals: tuple[RoundSeal, ...] = ()):
    """N requests per tree; require sealed exact manifests of all prior rounds."""
    if type(round_index) is not int or not 1 <= round_index <= schedule.l:
        raise ValueError("round outside frozen schedule")
    nodes = _forest(schedule, nodes, receipts, round_index)
    _seals(schedule, nodes, seals, round_index - 1)
    return _requests(schedule, nodes, round_index)


def select_leaves(schedule: TreeSchedule, nodes: tuple[SamplingNode, ...], receipts,
                  *, seals: tuple[RoundSeal, ...] = ()):
    """Select K terminal leaves per tree without replacement; never prune artifacts."""
    nodes = _forest(schedule, nodes, receipts, schedule.l)
    _seals(schedule, nodes, seals, schedule.l)
    selections = []
    for tree in schedule.tree_ids:
        pool = sorted(n.node_id for n in nodes if n.tree_id == tree and n.terminal)
        if len(pool) < schedule.k:
            raise ValueError("fewer than K generated terminal leaves in a tree")
        rng = random.Random(int(digest([schedule.sha256, tree, "leaf"]), 16))
        selections.append(LeafSelection(tree, tuple(rng.sample(pool, schedule.k)), tuple(pool)))
    return tuple(selections)
