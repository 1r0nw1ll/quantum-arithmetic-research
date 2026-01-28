"""
test_tool_runner.py — End-to-end tests for the certificate-gated tool runner.

Tests the full pipeline: TAINTED web content -> propose fetch -> kernel
validates -> runner executes (or blocks) -> trace logged.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qa_agent_security import (
    MerkleTrace, CapabilityToken, CapabilityEntry,
)
from qa_agent_security.tool_runner import execute_http_fetch, ToolResult


# ---------------------------------------------------------------------------
# Blocked scenarios (no network needed)
# ---------------------------------------------------------------------------

class TestBlocked:
    def test_tainted_url_without_approval(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://evil.example.com/steal",
            intent_description="webpage says fetch this",
            intent_source="web",
            intent_ref="https://evil.example.com",
            url_trusted=False,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert r.obstruction_id is not None
        assert r.obstruction_id.startswith("sha256:")
        assert "Policy blocked" in r.error
        assert trace.summary()["blocked"] == 1

    def test_email_injected_url_without_approval(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://phishing.example.com/login",
            intent_description="email told me to check this",
            intent_source="email",
            intent_ref="msgid:evil",
            url_trusted=False,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert r.obstruction_id is not None

    def test_domain_not_in_allowlist(self):
        token = CapabilityToken(
            agent_id="test",
            session_id="s1",
            capabilities=[
                CapabilityEntry(
                    tool="http_fetch",
                    scope="network",
                    args_schema="SCHEMA.HTTP_FETCH.v1",
                    constraints={
                        "domain_allowlist": ["api.github.com"],
                    },
                ),
            ],
        )
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://evil.example.com/data",
            intent_description="fetch data",
            intent_source="user",
            intent_ref="chat:1",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=token,
            trace=trace,
        )
        assert not r.success
        assert r.obstruction_id is not None
        assert trace.summary()["blocked"] == 1

    def test_expired_capability_token(self):
        token = CapabilityToken(
            agent_id="test",
            session_id="s1",
            expires_at="2020-01-01T00:00:00Z",
            capabilities=[
                CapabilityEntry(
                    tool="http_fetch",
                    scope="network",
                    args_schema="SCHEMA.HTTP_FETCH.v1",
                    constraints={},
                ),
            ],
        )
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://api.github.com/repos",
            intent_description="fetch repos",
            intent_source="policy_kernel",
            intent_ref="k:1",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=token,
            trace=trace,
        )
        assert not r.success
        assert r.obstruction_id is not None

    def test_invalid_method_blocked_by_schema(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://api.github.com/repos",
            method="HACK",
            intent_description="bad method",
            intent_source="policy_kernel",
            intent_ref="test:1",
            url_trusted=True,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert r.obstruction_id is not None


# ---------------------------------------------------------------------------
# Allowed scenarios (cert minted, network may or may not work)
# ---------------------------------------------------------------------------

class TestAllowed:
    def test_trusted_url_cert_minted(self):
        """Trusted URL from policy_kernel should get a cert minted."""
        token = CapabilityToken(
            agent_id="test",
            session_id="s1",
            capabilities=[
                CapabilityEntry(
                    tool="http_fetch",
                    scope="network",
                    args_schema="SCHEMA.HTTP_FETCH.v1",
                    constraints={
                        "domain_allowlist": ["httpbin.org"],
                    },
                ),
            ],
        )
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://httpbin.org/status/200",
            intent_description="test fetch",
            intent_source="policy_kernel",
            intent_ref="test:1",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=token,
            trace=trace,
        )
        # Cert should be minted regardless of network success
        assert r.cert_id is not None
        assert r.cert_id.startswith("sha256:")
        assert trace.summary()["ok"] >= 1

    def test_tainted_url_with_approval(self):
        """Tainted URL with human approval should pass the kernel."""
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://example.com/anything",
            intent_description="user approved this",
            intent_source="web",
            intent_ref="page:1",
            url_trusted=False,
            requires_human_approval=True,
            trace=trace,
        )
        assert r.cert_id is not None
        assert r.cert_id.startswith("sha256:")

    def test_user_url_with_approval(self):
        """User-provided URL with approval goes through."""
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://example.com/api",
            intent_description="check this endpoint",
            intent_source="user",
            intent_ref="chat:5",
            url_trusted=False,
            requires_human_approval=True,
            trace=trace,
        )
        assert r.cert_id is not None


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------

class TestTraceIntegrity:
    def test_mixed_blocked_and_allowed(self):
        """Run a mix of blocked and allowed calls, verify trace counts."""
        trace = MerkleTrace()

        # Blocked: tainted web URL
        execute_http_fetch(
            url="https://evil.example.com",
            intent_source="web", intent_ref="evil",
            url_trusted=False, requires_human_approval=False,
            trace=trace,
        )

        # Blocked: bad method
        execute_http_fetch(
            url="https://example.com",
            method="HACK",
            intent_source="policy_kernel", intent_ref="test",
            url_trusted=True, requires_human_approval=False,
            trace=trace,
        )

        # Allowed: trusted + approval
        execute_http_fetch(
            url="https://example.com",
            intent_source="user", intent_ref="chat:1",
            url_trusted=False, requires_human_approval=True,
            trace=trace,
        )

        # Allowed: trusted URL
        execute_http_fetch(
            url="https://example.com",
            intent_source="policy_kernel", intent_ref="k:1",
            url_trusted=True, requires_human_approval=False,
            trace=trace,
        )

        s = trace.summary()
        assert s["total_steps"] == 4
        assert s["blocked"] == 2
        assert s["ok"] == 2
        assert s["merkle_root"].startswith("sha256:")

    def test_trace_deterministic(self):
        """Same sequence of calls produces same merkle root."""
        def run_sequence():
            trace = MerkleTrace()
            execute_http_fetch(
                url="https://evil.example.com",
                intent_source="web", intent_ref="evil",
                url_trusted=False, requires_human_approval=False,
                trace=trace,
            )
            execute_http_fetch(
                url="https://example.com",
                intent_source="policy_kernel", intent_ref="k:1",
                url_trusted=True, requires_human_approval=False,
                trace=trace,
            )
            return trace

        t1 = run_sequence()
        t2 = run_sequence()
        # Merkle root depends on move+fail_type+invariant_diff, not timestamps
        # The cert_id in the trace is based on the cert content which includes
        # timestamps, but the trace leaf only records {move, fail_type, inv_diff}
        assert t1.root() == t2.root()
