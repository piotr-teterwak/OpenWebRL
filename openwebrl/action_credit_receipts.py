"""Typed replay attestations for the offline action-credit objective adapter.

This module validates consistency and identity, not physical browser truth or
the authority of whoever issued a receipt. The verifier and evidence digests
identify external artifacts; they are not signatures and do not prove those
artifacts were examined. Independent replay acceptance remains a readiness gate.
The collection contract transitively pins actor policy, tokenizer/processor,
task manifest, generation and judge protocols. No files or networks are touched.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from openwebrl.action_credit import ResponseGraph, digest


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class ReplayReceipt:
    edge_id: str
    contract_sha256: str
    state_sha256: str
    context_sha256: str
    remaining_turns: int
    verifier_sha256: str
    evidence_sha256: str
    accepted: bool

    def validate(self) -> None:
        """Validate schema, allowing accepted=False for preserved failed attempts."""
        if (not isinstance(self.edge_id, str) or not self.edge_id
                or self.edge_id != self.edge_id.strip()):
            raise ValueError("receipt edge_id must be a nonempty canonical identity")
        for field in ("contract_sha256", "state_sha256", "context_sha256",
                      "verifier_sha256", "evidence_sha256"):
            _sha(getattr(self, field), field)
        if type(self.remaining_turns) is not int or self.remaining_turns < 0:
            raise ValueError("receipt remaining_turns must be a nonnegative integer")
        if type(self.accepted) is not bool:
            raise ValueError("receipt accepted must be an explicit boolean")

    @property
    def sha256(self) -> str:
        """Canonical content digest, including schema version and verdict."""
        self.validate()
        return digest({"schema_version": 1, **asdict(self)})


def validate_receipts(graph: ResponseGraph,
                      receipts: Mapping[str, ReplayReceipt]) -> None:
    """Require one accepted, content-bound receipt for EVERY graph edge.

    Reject omitted/extra receipts and bind the receipt's exact actor context,
    state, horizon and collection contract to its edge. Changing the evidence,
    verifier or acceptance verdict changes the edge's required receipt digest.
    Do not infer an independent replay success from passing this structural check.
    """
    if not isinstance(graph, ResponseGraph):
        raise ValueError("expected ResponseGraph")
    edges = graph.validate()
    if not isinstance(receipts, Mapping) or set(receipts) != set(edges):
        raise ValueError("replay receipts must cover exactly the graph edges")
    for edge_id, edge in edges.items():
        receipt = receipts[edge_id]
        if not isinstance(receipt, ReplayReceipt):
            raise ValueError("expected typed ReplayReceipt")
        receipt.validate()
        if not receipt.accepted:
            raise ValueError(f"replay was not accepted for {edge_id}")
        if (receipt.edge_id != edge_id
                or receipt.contract_sha256 != edge.contract_sha256
                or receipt.state_sha256 != edge.state_sha256
                or receipt.context_sha256 != edge.context_sha256
                or receipt.remaining_turns != edge.remaining_turns):
            raise ValueError(f"replay receipt context/state/horizon/contract mismatch: {edge_id}")
        if receipt.sha256 != edge.replay_evidence_sha256:
            raise ValueError(f"replay receipt content digest mismatch: {edge_id}")
