"""Replay receipt identity checks using explicitly synthetic CPU-only records."""
from dataclasses import replace
import unittest

from openwebrl.action_credit import CollectionContract, Leaf, ResponseEdge, ResponseGraph, digest
from openwebrl.action_credit_receipts import ReplayReceipt, validate_receipts


def fixture():
    contract = CollectionContract(0, *[digest(x) for x in
        ("policy", "tokenizer", "processor", "tasks", "judge", "generation")],
        .8, ("task",), ("test",))
    edge = ResponseEdge("e", "task", "tree", None, contract.sha256, (0, 1),
        (digest("image"),), (2, 3), (True, True), (-2., -2.), contract.policy_sha256,
        digest("state"), digest("placeholder"), 30)
    receipt = ReplayReceipt(edge.edge_id, edge.contract_sha256, edge.state_sha256,
        edge.context_sha256, edge.remaining_turns, digest("synthetic verifier"),
        digest("synthetic evidence; no physical browser validation"), True)
    edge = replace(edge, replay_evidence_sha256=receipt.sha256)
    graph = ResponseGraph(contract, (edge,),
        (Leaf("l", "task", "tree", ("e",), "valid", 1., group_id="g"),), ("l",))
    return graph, receipt


class ReceiptTests(unittest.TestCase):
    def test_matching_receipt_is_valid_and_immutable(self):
        graph, receipt = fixture()
        self.assertIsNone(validate_receipts(graph, {"e": receipt}))
        self.assertEqual(receipt.sha256, replace(receipt).sha256)
        with self.assertRaises(AttributeError):
            receipt.accepted = False

    def test_wrong_context_horizon_state_contract_or_edge_rejected(self):
        graph, receipt = fixture()
        for changes in ({"edge_id": "other"}, {"context_sha256": digest("wrong context")},
                        {"state_sha256": digest("wrong state")}, {"remaining_turns": 29},
                        {"contract_sha256": digest("wrong policy contract")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_receipts(graph, {"e": replace(receipt, **changes)})

    def test_rejected_verdict_is_preserved_but_never_admitted(self):
        graph, receipt = fixture()
        rejected = replace(receipt, accepted=False)
        self.assertNotEqual(receipt.sha256, rejected.sha256)
        # Even a graph bound to the rejected receipt must not accept it.
        graph = replace(graph, edges=(replace(graph.edges[0], replay_evidence_sha256=rejected.sha256),))
        with self.assertRaises(ValueError):
            validate_receipts(graph, {"e": rejected})

    def test_evidence_verifier_digest_and_receipt_hash_tampering_rejected(self):
        graph, receipt = fixture()
        for field in ("evidence_sha256", "verifier_sha256"):
            changed = replace(receipt, **{field: digest("tampered")})
            self.assertNotEqual(changed.sha256, receipt.sha256)
            with self.assertRaises(ValueError):
                validate_receipts(graph, {"e": changed})
        with self.assertRaises(ValueError):
            validate_receipts(replace(graph, edges=(replace(graph.edges[0],
                replay_evidence_sha256=digest("wrong receipt bytes")),)), {"e": receipt})

    def test_actor_context_and_policy_changes_invalidate_old_receipt(self):
        graph, receipt = fixture()
        for changes in ({"input_ids": (5, 6)}, {"media_sha256": (digest("other image"),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_receipts(replace(graph, edges=(replace(graph.edges[0], **changes),)), {"e": receipt})
        contract = replace(graph.contract, policy_sha256=digest("new policy"))
        edge = replace(graph.edges[0], contract_sha256=contract.sha256,
                       old_policy_sha256=contract.policy_sha256)
        with self.assertRaises(ValueError):
            validate_receipts(replace(graph, contract=contract, edges=(edge,)), {"e": receipt})

    def test_missing_extra_and_untyped_receipts_rejected(self):
        graph, receipt = fixture()
        for receipts in ({}, {"e": receipt, "extra": receipt}, {"e": receipt.__dict__},
                         {"wrong-key": receipt}, [("e", receipt)]):
            with self.subTest(receipts=receipts), self.assertRaises(ValueError):
                validate_receipts(graph, receipts)

    def test_noncanonical_schema_fails(self):
        _, receipt = fixture()
        for changes in ({"accepted": 1}, {"remaining_turns": True}, {"remaining_turns": -1},
                        {"remaining_turns": 1.5}, {"edge_id": " e "}, {"edge_id": ""},
                        {"verifier_sha256": "abc"}, {"evidence_sha256": "A"*64},
                        {"context_sha256": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(receipt, **changes).sha256


if __name__ == "__main__":
    unittest.main()
