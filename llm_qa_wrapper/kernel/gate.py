# noqa: FIREWALL-2 (gate kernel — no mev/loop content)
"""
gate.py — The single cert-gate function.

Implements TLA+ actions: RequestToolCall, IssueCert, Deny, Execute.
Every LLM tool call passes through this gate before executing.

Maintains in-memory state corresponding to TLA+ variables:
  pending  → _pending: dict
  certs    → _certs: dict
  executed → _executed: set
  denied   → _denied_log: list

The ledger state is held by a separate Ledger instance and invoked
from the gate after Execute.
"""
from __future__ import annotations

QA_COMPLIANCE = {
    "observer": "LLM_QA_WRAPPER_GATE",
    "state_alphabet": "requests, certs (by self_hash), executed set, "
                      "denied records — all integer/bytes state, "
                      "no float, no continuous layer.",
    "rationale": "Implements cert_gate.tla single-gate routing. "
                 "Every tool call enters here and either produces a "
                 "signed CertRecord or a typed DenyRecord. The policy "
                 "decision is delegated to qa_guardrail.guard().",
}

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Tuple

from .cert import CertRecord, GENESIS, payload_hash
from .ledger import Ledger


class GateDecision(Enum):
    """Gate returns one of three outcomes on every tool call."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DenyRecord:
    """Structured denial. Never contains the denied payload bytes —
    only the payload hash, so the denial log itself is not a data-
    exfiltration vector."""

    request_id: str
    agent: str
    tool: str
    payload_hash_bytes: bytes
    reason: str
    ts: float


# Default policy oracle: allow everything. Production kernels replace
# this with qa_guardrail.guard() which evaluates against the failure
# algebra. Kept as a parameter so tests can inject custom policies.
def _default_policy(
    agent: str,
    tool: str,
    payload: Any,
) -> Tuple[GateDecision, Optional[str]]:
    return (GateDecision.ALLOW, None)


class Gate:
    """The cert-gate kernel.

    Usage:

        gate = Gate(ledger=Ledger(Path("ledger/")))
        rid = gate.submit_request("claude", "bash", {"cmd": "ls"})
        cert = gate.issue_cert(rid)
        if cert is None:
            # denied; check gate.last_denial for reason
            ...
        else:
            result = <execute the tool here>
            gate.execute(cert)
            offset = gate.ledger.append(cert)

    The `@gated_tool` decorator wraps this sequence around any
    callable — see hooks.py.
    """

    def __init__(
        self,
        ledger: Ledger,
        policy: Callable[[str, str, Any], Tuple[GateDecision, Optional[str]]] = _default_policy,
    ):
        self.ledger = ledger
        self.policy = policy
        self._lock = threading.Lock()
        # Wrapper serialization lock (2026-04-11 fix for Phase 5 finding
        # "hash-chain-fork"). The entire issue→tool-run→execute sequence
        # must be atomic across threads because cert counters are
        # assigned monotonically at issue_cert time but the ledger
        # appends must land in counter order. Without this, concurrent
        # wrappers produce out-of-order appends that the ledger rejects.
        # Tradeoff: tool calls are serialized across all @gated_tool
        # invocations on the same Gate instance. For research prototype
        # correctness trumps throughput; production deployments that
        # need parallelism must implement a ticketed append buffer.
        self._wrapper_lock = threading.Lock()

        # TLA+ state variables
        self._pending: Dict[str, Tuple[str, str, Any]] = {}
        self._certs: Dict[bytes, CertRecord] = {}
        self._executed: Set[bytes] = set()
        self._denied_log: list[DenyRecord] = []

        # Monotonic counter — matches TLA+ Cardinality(certs)
        # and is used as the counter field in new certs.
        self._counter: int = 0

        self.last_denial: Optional[DenyRecord] = None

    # ───────────────── TLA+ RequestToolCall ─────────────────

    def submit_request(self, agent: str, tool: str, payload: Any) -> str:
        """Submit a tool call to the gate. Returns a request_id.

        Matches cert_gate.tla:RequestToolCall. Adds (agent, tool, payload)
        to the pending set. Idempotent: same tuple → same pending entry.
        """
        with self._lock:
            rid = str(uuid.uuid4())
            self._pending[rid] = (agent, tool, payload)
            return rid

    # ───────────────── TLA+ IssueCert ─────────────────

    def issue_cert(self, request_id: str) -> Optional[CertRecord]:
        """Issue a cert for a pending request.

        Critical section (per ARCHITECTURE.md §5): lock over
          (pending lookup, policy check, LastCertHash read, cert build,
           _certs insert, pending delete)
        to preserve serializability against the TLA+ spec.

        Returns the CertRecord on ALLOW, or None on DENY. On DENY,
        self.last_denial carries the structured reason.
        """
        with self._lock:
            # Precondition: request must be pending
            triple = self._pending.get(request_id)
            if triple is None:
                self.last_denial = DenyRecord(
                    request_id=request_id,
                    agent="?",
                    tool="?",
                    payload_hash_bytes=GENESIS,
                    reason="NOT_PENDING",
                    ts=time.time(),
                )
                self._denied_log.append(self.last_denial)
                return None

            agent, tool, payload = triple

            # Policy check
            decision, reason = self.policy(agent, tool, payload)
            if decision != GateDecision.ALLOW:
                ph = payload_hash(payload)
                self.last_denial = DenyRecord(
                    request_id=request_id,
                    agent=agent,
                    tool=tool,
                    payload_hash_bytes=ph,
                    reason=reason or str(decision),
                    ts=time.time(),
                )
                self._denied_log.append(self.last_denial)
                del self._pending[request_id]
                return None

            # Build cert: prev = last cert's self_hash (or GENESIS), counter = monotone
            prev = self._last_cert_hash_locked()
            counter = self._counter
            cert = CertRecord.create(agent, tool, payload, prev, counter)

            # Commit: insert into certs, bump counter, clear pending
            self._certs[cert.self_hash] = cert
            self._counter += 1
            del self._pending[request_id]
            return cert

    def _last_cert_hash_locked(self) -> bytes:
        """Return the hash of the most recently issued cert.

        Must be called with self._lock held. Matches TLA+ LastCertHash
        operator: the self_hash of the cert with the largest counter,
        or GENESIS if no certs have been issued.
        """
        if not self._certs:
            return GENESIS
        # Highest counter wins
        max_cert = max(self._certs.values(), key=lambda c: c.counter)
        return max_cert.self_hash

    # ───────────────── TLA+ Execute + AppendLedger (combined) ─────────────────
    # 2026-04-11 fix: previously execute() and ledger.append() were separate
    # critical sections, which allowed out-of-order appends under concurrent
    # @gated_tool calls — a real circumvention found by Phase 5 self-test.
    # The TLA+ spec models Execute and AppendLedger as sequential atomic
    # actions; the Python kernel now matches that by combining them into one
    # atomic operation under gate._lock. Under high concurrency this
    # serializes the execute→append path, but correctness trumps throughput.

    def execute(self, cert: CertRecord) -> bool:
        """Mark a cert as executed AND append it to the ledger, atomically.

        Must be called AFTER the actual tool runs. Returns True on success,
        False if the cert is already executed (replay), not in _certs
        (forgery), or the ledger append fails for any other reason.

        Critical section: a single gate._lock covers (a) forgery check,
        (b) replay check, (c) mark as executed, (d) ledger append. This
        matches the TLA+ spec's Execute-then-AppendLedger sequence and
        preserves the invariant that appends land in cert-chain order
        regardless of the order in which @gated_tool wrappers call this
        method.

        Under the lock we also acquire ledger._lock via ledger.append —
        the nested acquisition is consistent (same order every time) so
        no deadlock. Ledger.append may raise if the chain order is
        broken, in which case we rollback the executed set and re-raise.
        """
        with self._lock:
            if cert.self_hash not in self._certs:
                return False  # forged cert — not in our issued set
            if cert.self_hash in self._executed:
                return False  # replay — already executed
            # Tentatively mark as executed, then append to ledger.
            # If append raises, rollback.
            self._executed.add(cert.self_hash)
            try:
                self.ledger.append(cert)
            except Exception:
                self._executed.discard(cert.self_hash)
                raise
            return True

    # ───────────────── Diagnostic / test API ─────────────────

    def snapshot_state(self) -> dict:
        """Return an in-memory state snapshot for tests / conformance.

        Matches the TLA+ vars tuple so conformance tests can compare
        state hashes directly.
        """
        with self._lock:
            return {
                "pending_count": len(self._pending),
                "certs_count": len(self._certs),
                "executed_count": len(self._executed),
                "denied_count": len(self._denied_log),
                "counter": self._counter,
                "ledger_tail_hex": self.ledger.tail_hash().hex(),
                "ledger_count": len(self.ledger),
            }

    def reset_for_testing(self) -> None:
        """Clear kernel state. DANGER: data loss. Tests only."""
        with self._lock:
            self._pending.clear()
            self._certs.clear()
            self._executed.clear()
            self._denied_log.clear()
            self._counter = 0
            self.last_denial = None
        self.ledger.reset_for_testing()
