# noqa: FIREWALL-2 (kernel package init — no mev/loop content)
"""LLM QA Wrapper kernel package.

See ARCHITECTURE.md for the action-for-action mapping from cert_gate.tla
to kernel code paths. Implementation follows the TLA+ spec and Lean
ledger invariants; conformance tests in tests/test_conformance.py
verify that replay of TLC counterexample traces matches kernel state.
"""

QA_COMPLIANCE = {
    "observer": "LLM_QA_WRAPPER_KERNEL",
    "state_alphabet": "integer cert records + SHA-256 hash chain + "
                      "policy decisions from qa_guardrail",
    "rationale": "Single-gate routing for all LLM tool calls. "
                 "Implements cert_gate.tla action set 1:1.",
}

from .cert import CertRecord, GENESIS, payload_hash  # noqa: E402, F401
from .ledger import Ledger, LedgerVerifyResult  # noqa: E402, F401
from .gate import Gate, GateDecision, DenyRecord  # noqa: E402, F401
from .hooks import gated_tool  # noqa: E402, F401
