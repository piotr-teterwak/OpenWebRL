"""Bind numerical objectives to an independently sealed complete draw ledger.

This is an offline identity check, not browser/GPU verification or permission to
execute an optimizer. A snapshot preserves fit, holdout, invalid and missing
slots. Updating a graph cannot silently redefine the prespecified denominator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .action_credit import CollectionContract, ResponseGraph
from .action_credit_state import CampaignStore, StateError, digest


def _require(condition, message):
    if not condition:
        raise StateError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def token_mask_contract():
    """The implemented response-domain mask, not physical alignment evidence."""
    return dict(domain="complete_sampled_response_including_reasoning_and_action",
                sampled_mask="response_aligned_boolean_mask",
                excluded="prompt_observation_image_and_transport_padding")


def shared_prefix_contract(reduction):
    return dict(reduction=reduction, mode="separate_leaf_terms_before_clipping")


def _check_contract(manifest, contract):
    splits = {task: "train" for task in contract.train_task_ids}
    splits.update({task: "holdout" for task in contract.heldout_task_ids})
    _require(manifest["task_splits"] == splits, "Graph task partition differs from campaign")
    # Require explicit, unique pins; generic source/prompt hashes cannot replace
    # the complete protocol hashes named by the graph contract.
    pins = ("tokenizer_sha256", "processor_sha256", "task_manifest_sha256",
            "judge_protocol_sha256", "generation_sha256", "likelihood_temperature",
            "likelihood_convention")
    for name in pins:
        values = [settings[name] for settings in manifest["science"].values() if name in settings]
        _require(len(values) == 1 and values[0] == getattr(contract, name),
                 "Missing, duplicate or mismatched campaign scientific pin: " + name)
    mask = manifest["science"]["token_mask"]
    _require(mask.get("contract") == token_mask_contract()
             and mask["protocol_sha256"] == digest(token_mask_contract()),
             "Campaign token mask must pin the implemented complete-response convention")


@dataclass(frozen=True, init=False)
class SealedCollection:
    """Canonical immutable snapshot; only ``from_store`` constructs one.

    Its hashes attest identity and completeness, not the external truth of a
    saved outcome. Python object construction is not a security boundary.
    """

    canonical_json: str

    def __init__(self, *args, **kwargs):
        raise TypeError("Use SealedCollection.from_store with the complete sealed ledger")

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @classmethod
    def from_store(cls, store, graph_contract, leaf_to_draw):
        _require(type(store) is CampaignStore and type(graph_contract) is CollectionContract,
                 "Require a CampaignStore and typed CollectionContract")
        graph_contract.validate()
        state = store.snapshot()
        manifest = state["manifest"]
        _require(digest(manifest) == state["manifest_sha256"] == store.manifest.sha256,
                 "Campaign manifest digest mismatch")
        _check_contract(manifest, graph_contract)
        update_id = graph_contract.update_id
        update = state["updates"].get(str(update_id))
        _require(update is not None and update["phase"] in ("sealed", "training_pending", "consumed"),
                 "Collection must be sealed before graph binding")
        _require(update["policy_sha256"] == graph_contract.policy_sha256,
                 "Graph policy differs from sealed collection")
        expected = update["expected_draws"]
        _require(digest(expected) == update["plan_sha256"], "Planned inventory digest mismatch")
        _require(isinstance(leaf_to_draw, dict) and bool(leaf_to_draw)
                 and all(isinstance(k, str) and k and k.strip() == k for k in leaf_to_draw)
                 and all(isinstance(v, str) for v in leaf_to_draw.values())
                 and len(set(leaf_to_draw.values())) == len(leaf_to_draw)
                 and set(leaf_to_draw.values()) == set(expected),
                 "Leaf mapping must bijectively cover ALL planned fit and holdout draws")
        # Use the same export implementation against this single locked snapshot;
        # do not combine reads from different store revisions.
        store._export(state, update_id, None)
        inventory = {}
        for leaf_id, draw_id in sorted(leaf_to_draw.items()):
            identity = expected[draw_id]
            _require(digest(identity) == draw_id and identity["update_id"] == update_id
                     and identity["policy_sha256"] == graph_contract.policy_sha256,
                     "Draw identity differs from update")
            draw = state["draws"].get(draw_id)
            _require(draw is not None and all(draw.get(k) == v for k, v in identity.items()),
                     "Reserved draw differs from planned identity")
            _require(draw["status"] in ("valid", "invalid", "missing"), "Unresolved planned draw")
            selected = draw["selected_attempt"]
            completion = None
            if selected is not None:
                _require(selected in draw["attempts"], "Selected attempt is outside draw")
                attempt = state["attempts"][selected]
                completion = attempt["completion"]
                _require(attempt["draw_id"] == draw_id and completion is not None
                         and attempt["status"] == completion["status"] == draw["status"],
                         "Selected attempt does not match terminal draw")
            else:
                _require(draw["status"] == "missing" and not draw["attempts"] and draw["missing_reason"],
                         "Terminal draw lacks a selected attempt or explicit missing record")
            _require(not any(state["attempts"][a]["status"] == "active" for a in draw["attempts"]),
                     "Sealed draw still has an active attempt")
            outcome = completion["outcome"] if completion else None
            reward = outcome.get("reward") if outcome else None
            _require((reward is not None) == (draw["status"] == "valid"),
                     "Outcome reward/status mismatch")
            inventory[leaf_id] = dict(identity, draw_id=draw_id, status=draw["status"], reward=reward,
                outcome=outcome, reason=completion["reason"] if completion else draw["missing_reason"],
                sampling_terminal_node_id=completion.get("sampling_terminal_node_id") if completion else None)
        payload = dict(schema_version=1, phase=update["phase"], manifest=manifest,
            manifest_sha256=state["manifest_sha256"], plan_sha256=update["plan_sha256"],
            export_sha256=update["export_sha256"], update_id=update_id,
            policy_sha256=graph_contract.policy_sha256, graph_contract_sha256=graph_contract.sha256,
            inventory=inventory)
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_json", _canonical(payload))
        return instance


def validate_collection_binding(graph, config, panels, binding, *, selection=None):
    """Reject changed outcomes, roles, inventory, candidates or complete panels.

    An empty panels tuple permits the zero-local ablation, while nonzero local
    credit requires the complete candidate panel inventory.
    """
    from .action_credit_objective import ObjectiveConfig
    _require(type(binding) is SealedCollection and type(graph) is ResponseGraph
             and type(config) is ObjectiveConfig,
             "Objective requires a typed sealed collection binding")
    config.validate()
    graph.validate()
    bound = binding.to_dict()
    _require(binding.canonical_json == _canonical(bound), "Binding must be canonical JSON")
    _require(bound["graph_contract_sha256"] == graph.contract.sha256
             and bound["update_id"] == graph.contract.update_id
             and bound["policy_sha256"] == graph.contract.policy_sha256,
             "Graph contract differs from sealed collection")
    _require(bound["manifest"]["arm_id"] == config.arm_id, "Objective arm differs from campaign")
    science = bound["manifest"]["science"]
    _require(science["algorithm"].get("objective_config_sha256") == digest(asdict(config)),
             "Complete objective configuration differs from campaign")
    selection_policy = {"mode": "all_fit_inventory"}
    if selection is not None:
        from .action_credit_selection import TrainingSelectionReceipt
        _require(type(selection) is TrainingSelectionReceipt
                 and config.arm_id in ("tree_outcome_sampling", "tree_grpo"),
                 "Only tree arms accept a typed outcome-blind selection receipt")
        selection_policy = selection.policy
        _require(selection.policy_sha256 == digest(selection_policy), "Selection policy digest mismatch")
    _require(science["algorithm"].get("leaf_selection") == selection_policy,
             "Leaf selection schedule, seed or policy differs from prespecified campaign")
    prefix = science["shared_prefix_normalization"]
    expected_prefix = shared_prefix_contract(config.reduction)
    _require(prefix.get("contract") == expected_prefix
             and prefix["protocol_sha256"] == digest(expected_prefix),
             "Shared-prefix reduction differs from campaign")
    inventory = bound["inventory"]
    _require(set(inventory) == set(graph.planned_leaf_ids)
             == {leaf.leaf_id for leaf in graph.leaves}, "Graph omits or adds a planned draw")
    for leaf in graph.leaves:
        row = inventory[leaf.leaf_id]
        _require((leaf.task_id, leaf.group_id, leaf.tree_id, leaf.split, leaf.status, leaf.reward)
                 == (row["task_id"], row["group_id"], row["tree_id"], row["draw_role"], row["status"], row["reward"]),
                 "Graph task, group, tree, draw role, status or reward differs from sealed draw")
    if selection is not None:
        terminal_map = dict(selection.terminal_to_leaf)
        recorded = [(row["sampling_terminal_node_id"], leaf_id) for leaf_id, row in inventory.items()
                    if row["draw_role"] == "train" and row["sampling_terminal_node_id"] is not None]
        _require(len(terminal_map) == len(selection.terminal_to_leaf)
                 and len({terminal for terminal, _ in recorded}) == len(recorded)
                 and terminal_map == dict(recorded),
                 "Selection terminal mapping differs from independently sealed outcome metadata")
    _require(config.local_coefficient == 0 or bool(panels),
             "Nonzero continuation credit requires every planned candidate panel")
    if panels:
        _require(config.arm_id == "continuation_q", "Only continuation-Q accepts candidate panels")
        actual, panel_ids = {}, set()
        for panel in panels:
            _require(panel.panel_id not in panel_ids, "Duplicate continuation panel")
            panel_ids.add(panel.panel_id)
            _require(len(panel.candidate_edges) == len(panel.draw_leaf_ids)
                     and len(set(panel.candidate_edges)) == len(panel.candidate_edges),
                     "Candidate inventory mismatch")
            for action_id, leaves in zip(panel.candidate_edges, panel.draw_leaf_ids):
                _require(bool(leaves), "Candidate has no prespecified draws")
                for leaf_id in leaves:
                    _require(leaf_id not in actual, "Repeated draw in continuation panels")
                    actual[leaf_id] = (panel.panel_id, action_id)
        expected = {key: (row["state_id"], row["action_id"]) for key, row in inventory.items()}
        _require(actual == expected,
                 "Panels must exactly retain planned states, candidates and ALL draw slots")
