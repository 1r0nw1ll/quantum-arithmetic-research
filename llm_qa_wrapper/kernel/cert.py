# noqa: FIREWALL-2 (cert record + hashing — no mev/loop content)
"""
cert.py — CertRecord and SHA-256 hashing.

Maps 1:1 to the TLA+ record type in cert_gate.tla and the Lean
CertRecord struct in LedgerInvariants.lean. Every field is
integer-typed or bytes; no float state.
"""
from __future__ import annotations

QA_COMPLIANCE = {
    "observer": "LLM_QA_WRAPPER_CERT",
    "state_alphabet": "SHA-256 32-byte hashes, integer counters, "
                      "canonical JSON payloads",
    "rationale": "CertRecord is the single unit of auditable work. "
                 "Hash is deterministic over canonical JSON so that "
                 "the same (agent, tool, payload, prev, counter) "
                 "tuple always produces the same self_hash — "
                 "required for Lean hash_chain_binds_contents.",
}

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict


#: Genesis hash — the prev of the first ledger entry. 32 zero bytes.
GENESIS: bytes = bytes(32)


def canonical_json(obj: Any) -> bytes:
    """Return the canonical JSON encoding of obj.

    Deterministic: sorted keys, no whitespace, UTF-8 bytes.
    Used for payload hashing and cert self-hashing.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def payload_hash(payload: Any) -> bytes:
    """SHA-256 of the canonical JSON of a payload."""
    return hashlib.sha256(canonical_json(payload)).digest()


def _compute_self_hash(
    agent: str,
    tool: str,
    payload_hash_bytes: bytes,
    prev: bytes,
    counter: int,
) -> bytes:
    """Compute the SHA-256 binding for a CertRecord.

    Matches the TLA+ Hash function signature and is invertible only
    via collision attack on SHA-256.
    """
    if not isinstance(counter, int):
        raise TypeError("counter must be int (A1/S2 compliance)")
    if counter < 0:
        raise ValueError("counter must be non-negative")
    if len(payload_hash_bytes) != 32:
        raise ValueError("payload_hash must be 32 bytes (SHA-256)")
    if len(prev) != 32:
        raise ValueError("prev must be 32 bytes")

    h = hashlib.sha256()
    h.update(agent.encode("utf-8"))
    h.update(b"\x00")  # field separator
    h.update(tool.encode("utf-8"))
    h.update(b"\x00")
    h.update(payload_hash_bytes)
    h.update(b"\x00")
    h.update(prev)
    h.update(b"\x00")
    h.update(counter.to_bytes(8, "big"))
    return h.digest()


@dataclass(frozen=True)
class CertRecord:
    """A single ledger entry.

    Matches TLA+ cert structure: {agent, tool, payload, ch, prev, ctr}
    where `ch = Hash(agent, tool, payload, prev, ctr)`.

    In Python the payload is hashed separately so that the cert itself
    only carries the fixed-size payload_hash, not the variable-size
    payload bytes. This keeps ledger entries fixed-size and the
    Ledger.append critical section bounded.
    """

    agent: str
    tool: str
    payload_hash_bytes: bytes  # SHA-256 of canonical JSON of payload
    prev: bytes                # 32 bytes; GENESIS for first cert
    counter: int               # monotonic per-kernel
    self_hash: bytes           # SHA-256 binding of all other fields

    def __post_init__(self) -> None:
        # Verify self_hash matches at construction time; any tampering
        # produces an error at load time.
        expected = _compute_self_hash(
            self.agent,
            self.tool,
            self.payload_hash_bytes,
            self.prev,
            self.counter,
        )
        if expected != self.self_hash:
            raise ValueError(
                "CertRecord self_hash does not match computed hash — "
                "record is tampered or malformed"
            )

    @classmethod
    def create(
        cls,
        agent: str,
        tool: str,
        payload: Any,
        prev: bytes,
        counter: int,
    ) -> "CertRecord":
        """Factory: compute payload_hash and self_hash, return record."""
        ph = payload_hash(payload)
        sh = _compute_self_hash(agent, tool, ph, prev, counter)
        return cls(
            agent=agent,
            tool=tool,
            payload_hash_bytes=ph,
            prev=prev,
            counter=counter,
            self_hash=sh,
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize for ledger write. Bytes → hex strings."""
        return {
            "agent": self.agent,
            "tool": self.tool,
            "payload_hash": self.payload_hash_bytes.hex(),
            "prev": self.prev.hex(),
            "counter": self.counter,
            "self_hash": self.self_hash.hex(),
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "CertRecord":
        """Deserialize from a ledger line. Verifies self_hash on load."""
        return cls(
            agent=obj["agent"],
            tool=obj["tool"],
            payload_hash_bytes=bytes.fromhex(obj["payload_hash"]),
            prev=bytes.fromhex(obj["prev"]),
            counter=obj["counter"],
            self_hash=bytes.fromhex(obj["self_hash"]),
        )
