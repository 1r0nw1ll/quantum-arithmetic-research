"""
test_agent_security.py — pytest suite for the QA agent security kernel.

Tests the policy kernel, taint tracking, obstruction generation, merkle trace,
and capability token enforcement.
"""
import pytest
import sys
import os

# Ensure module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qa_agent_security import (
    Prov, pv, TAINTED, TRUSTED,
    ToolSpec, CapabilityToken, CapabilityEntry,
    enforce_policy, PolicyError, obstruction_from_policy_error,
    mint_taint_flow_cert,
    MerkleTrace, MerkleLeaf, merkle_root,
    canonical_json_sha256, canonical_json_dumps,
    now_rfc3339,
)


TS = "2026-01-27T00:00:00Z"


# ---------------------------------------------------------------------------
# Provenance / taint basics
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_valid_prov(self):
        p = Prov("user", "chat:1", TAINTED, TS)
        assert p.source == "user"
        assert p.taint == TAINTED

    def test_rejects_invalid_source(self):
        with pytest.raises(ValueError, match="Invalid source"):
            Prov("hacker", "x", TAINTED, TS)

    def test_rejects_invalid_taint(self):
        with pytest.raises(ValueError, match="Invalid taint"):
            Prov("user", "x", "MAYBE", TS)

    def test_pv_creates_tagged_value(self):
        p = Prov("web", "url:1", TAINTED, TS)
        field = pv("hello", p)
        assert field["value"] == "hello"
        assert field["prov"]["taint"] == TAINTED

    def test_trusted_field_not_tainted(self):
        p = Prov("policy_kernel", "k:1", TRUSTED, TS)
        field = pv("safe", p)
        from qa_agent_security import is_tainted
        assert not is_tainted(field)


# ---------------------------------------------------------------------------
# Taint flow certificates
# ---------------------------------------------------------------------------

class TestTaintFlow:
    def test_valid_tainted_to_tainted(self):
        inp = pv("raw", Prov("web", "url:1", TAINTED, TS))
        out = pv("summary", Prov("policy_kernel", "cert:1", TAINTED, TS))
        cert = mint_taint_flow_cert([inp], "summarize", {"max_tokens": 256}, [out])
        assert cert["schema_version"] == "TAINT_FLOW_CERT.v1"
        assert cert["cert_id"].startswith("sha256:")

    def test_blocks_taint_upgrade(self):
        inp = pv("raw", Prov("web", "url:1", TAINTED, TS))
        bad_out = pv("cleaned", Prov("policy_kernel", "cert:2", TRUSTED, TS))
        with pytest.raises(PolicyError) as exc_info:
            mint_taint_flow_cert([inp], "summarize", {}, [bad_out])
        assert exc_info.value.fail_type == "TAINT_UPGRADE_VIOLATION"

    def test_trusted_inputs_allow_trusted_outputs(self):
        inp = pv("config", Prov("system", "sys:1", TRUSTED, TS))
        out = pv("result", Prov("policy_kernel", "k:1", TRUSTED, TS))
        cert = mint_taint_flow_cert([inp], "transform", {}, [out])
        assert cert["schema_version"] == "TAINT_FLOW_CERT.v1"


# ---------------------------------------------------------------------------
# Policy kernel — web->exec blocking
# ---------------------------------------------------------------------------

class TestWebToExec:
    TOOL = ToolSpec(name="run_shell", capability_scope="exec",
                    args_schema_id="SCHEMA.RUN_SHELL.v1")

    def test_blocks_web_to_exec_without_approval(self):
        intent = pv("do it", Prov("web", "https://evil.example", TAINTED, TS))
        args = {"command": pv("rm -rf /", Prov("web", "https://evil.example", TAINTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                           policy_rule_id="POLICY.EXEC.V1",
                           requires_human_approval=False)
        assert exc_info.value.fail_type == "CAPABILITY_ESCALATION_ATTEMPT"
        invs = {d["inv"] for d in exc_info.value.invariant_diff}
        assert "NO_WEB_TO_EXEC" in invs

    def test_allows_exec_with_approval_even_if_tainted(self):
        intent = pv("run", Prov("user", "chat:1", TAINTED, TS))
        args = {"command": pv("echo hello", Prov("user", "chat:1", TAINTED, TS))}
        cert = enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                              policy_rule_id="POLICY.EXEC.V1",
                              requires_human_approval=True)
        assert cert["schema_version"] == "TOOL_CALL_CERT.v1"
        assert cert["requires_human_approval"] is True

    def test_allows_trusted_exec_without_approval(self):
        intent = pv("list", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args = {"command": pv("ls /tmp", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        cert = enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                              policy_rule_id="POLICY.EXEC.V1",
                              requires_human_approval=False)
        assert cert["schema_version"] == "TOOL_CALL_CERT.v1"


# ---------------------------------------------------------------------------
# Email recipient gating
# ---------------------------------------------------------------------------

class TestEmailRecipient:
    TOOL = ToolSpec(name="send_email", capability_scope="write_limited",
                    args_schema_id="SCHEMA.SEND_EMAIL.v1")

    def test_blocks_tainted_recipient(self):
        intent = pv("fwd email", Prov("email", "msg:1", TAINTED, TS))
        args = {
            "to": pv("attacker@evil.example", Prov("email", "msg:1", TAINTED, TS)),
            "body": pv("hi", Prov("policy_kernel", "d:1", TAINTED, TS)),
        }
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                           policy_rule_id="POLICY.EMAIL.SEND.V1",
                           requires_human_approval=False)
        invs = {d["inv"] for d in exc_info.value.invariant_diff}
        assert "TO_TRUSTED" in invs

    def test_allows_trusted_recipient(self):
        intent = pv("send report", Prov("user", "chat:3", TAINTED, TS))
        args = {
            "to": pv("team@safe.com", Prov("policy_kernel", "k:1", TRUSTED, TS)),
            "body": pv("report text", Prov("policy_kernel", "k:2", TAINTED, TS)),
        }
        cert = enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                              policy_rule_id="POLICY.EMAIL.SEND.V1",
                              requires_human_approval=False)
        assert cert["schema_version"] == "TOOL_CALL_CERT.v1"


# ---------------------------------------------------------------------------
# Obstruction generation
# ---------------------------------------------------------------------------

class TestObstruction:
    def test_obstruction_has_valid_cert_id(self):
        tool = ToolSpec(name="run_shell", capability_scope="exec",
                        args_schema_id="SCHEMA.RUN_SHELL.v1")
        intent = pv("do it", Prov("web", "url:1", TAINTED, TS))
        args = {"command": pv("rm -rf /", Prov("web", "url:1", TAINTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=tool, intent_pv=intent, args_pv=args,
                           policy_rule_id="TEST", requires_human_approval=False)
        obs = obstruction_from_policy_error(exc_info.value, tool)
        assert obs["schema_version"] == "PROMPT_INJECTION_OBSTRUCTION.v1"
        assert obs["cert_id"].startswith("sha256:")
        assert len(obs["cert_id"]) == 71  # sha256: + 64 hex chars

    def test_obstruction_preserves_fail_type(self):
        tool = ToolSpec(name="run_shell", capability_scope="exec",
                        args_schema_id="SCHEMA.RUN_SHELL.v1")
        intent = pv("x", Prov("web", "url:1", TAINTED, TS))
        args = {"command": pv("x", Prov("web", "url:1", TAINTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=tool, intent_pv=intent, args_pv=args,
                           policy_rule_id="TEST", requires_human_approval=False)
        obs = obstruction_from_policy_error(exc_info.value, tool)
        assert obs["fail_type"] == "CAPABILITY_ESCALATION_ATTEMPT"


# ---------------------------------------------------------------------------
# Merkle trace
# ---------------------------------------------------------------------------

class TestMerkleTrace:
    def test_records_moves(self):
        trace = MerkleTrace()
        trace.append(MerkleLeaf(move="TOOL_CALL:run_shell", fail_type="OK"))
        trace.append(MerkleLeaf(
            move="TOOL_CALL:send_email",
            fail_type="UNTRUSTED_INSTRUCTION",
            invariant_diff=[{"inv": "TO_TRUSTED", "expected": "pass", "got": "fail"}],
        ))
        s = trace.summary()
        assert s["total_steps"] == 2
        assert s["ok"] == 1
        assert s["blocked"] == 1
        assert s["merkle_root"].startswith("sha256:")

    def test_empty_trace(self):
        trace = MerkleTrace()
        s = trace.summary()
        assert s["total_steps"] == 0
        assert s["merkle_root"].startswith("sha256:")

    def test_deterministic_root(self):
        t1 = MerkleTrace()
        t2 = MerkleTrace()
        leaf = MerkleLeaf(move="A", fail_type="OK")
        t1.append(leaf)
        t2.append(leaf)
        assert t1.root() == t2.root()

    def test_leaf_hash_deterministic(self):
        l1 = MerkleLeaf(move="X", fail_type="OK", invariant_diff=[])
        l2 = MerkleLeaf(move="X", fail_type="OK", invariant_diff=[])
        assert l1.leaf_hash() == l2.leaf_hash()


# ---------------------------------------------------------------------------
# Capability token enforcement
# ---------------------------------------------------------------------------

class TestCapabilityToken:
    TOOL = ToolSpec(name="run_shell", capability_scope="exec",
                    args_schema_id="SCHEMA.RUN_SHELL.v1")

    def _make_token(self, **overrides):
        defaults = dict(
            agent_id="test-agent",
            session_id="sess-1",
            capabilities=[
                CapabilityEntry(
                    tool="run_shell",
                    scope="exec",
                    args_schema="SCHEMA.RUN_SHELL.v1",
                    constraints={
                        "command_denylist_regex": [r"\brm\b", r"\bmkfs\b"],
                    },
                ),
            ],
        )
        defaults.update(overrides)
        return CapabilityToken(**defaults)

    def test_denylist_blocks_rm(self):
        token = self._make_token()
        intent = pv("clean", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args = {"command": pv("rm -rf /tmp/junk", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                           policy_rule_id="POLICY.EXEC.V1",
                           requires_human_approval=False,
                           capability_token=token)
        assert exc_info.value.fail_type == "CONSTRAINT_VIOLATION"
        invs = {d["inv"] for d in exc_info.value.invariant_diff}
        assert "COMMAND_DENYLIST" in invs

    def test_allows_safe_command(self):
        token = self._make_token()
        intent = pv("list", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args = {"command": pv("ls /tmp", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        cert = enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                              policy_rule_id="POLICY.EXEC.V1",
                              requires_human_approval=False,
                              capability_token=token)
        assert cert["schema_version"] == "TOOL_CALL_CERT.v1"

    def test_expired_token_rejected(self):
        token = self._make_token(expires_at="2020-01-01T00:00:00Z")
        intent = pv("list", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args = {"command": pv("ls", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=self.TOOL, intent_pv=intent, args_pv=args,
                           policy_rule_id="POLICY.EXEC.V1",
                           requires_human_approval=False,
                           capability_token=token)
        invs = {d["inv"] for d in exc_info.value.invariant_diff}
        assert "CAPABILITY_TOKEN_VALID" in invs

    def test_missing_capability_for_tool(self):
        # Token has run_shell capability but we use http_fetch
        token = self._make_token()
        fetch_tool = ToolSpec(name="http_fetch", capability_scope="network",
                              args_schema_id="SCHEMA.HTTP_FETCH.v1")
        intent = pv("fetch", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args = {"url": pv("https://example.com", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        with pytest.raises(PolicyError) as exc_info:
            enforce_policy(tool=fetch_tool, intent_pv=intent, args_pv=args,
                           policy_rule_id="POLICY.FETCH.V1",
                           requires_human_approval=False,
                           capability_token=token)
        invs = {d["inv"] for d in exc_info.value.invariant_diff}
        assert "CAPABILITY_FOUND" in invs

    def test_to_dict(self):
        token = self._make_token()
        d = token.to_dict()
        assert d["schema_version"] == "CAPABILITY_TOKEN.v1"
        assert len(d["capabilities"]) == 1


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------

class TestCanonicalJSON:
    def test_deterministic_key_order(self):
        a = {"z": 1, "a": 2, "m": [3, 1]}
        b = {"a": 2, "m": [3, 1], "z": 1}
        assert canonical_json_sha256(a) == canonical_json_sha256(b)

    def test_no_whitespace(self):
        s = canonical_json_dumps({"a": 1})
        assert " " not in s
        assert "\n" not in s


# ---------------------------------------------------------------------------
# Integration: trace + policy kernel together
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_trace_records_blocked_and_allowed(self):
        trace = MerkleTrace()
        tool = ToolSpec(name="run_shell", capability_scope="exec",
                        args_schema_id="SCHEMA.RUN_SHELL.v1")

        # Blocked attempt
        intent = pv("hack", Prov("web", "evil.com", TAINTED, TS))
        args = {"command": pv("rm -rf /", Prov("web", "evil.com", TAINTED, TS))}
        try:
            enforce_policy(tool=tool, intent_pv=intent, args_pv=args,
                           policy_rule_id="TEST", requires_human_approval=False,
                           trace=trace)
        except PolicyError:
            pass

        # Allowed attempt
        intent2 = pv("list", Prov("policy_kernel", "k:1", TRUSTED, TS))
        args2 = {"command": pv("ls", Prov("policy_kernel", "k:2", TRUSTED, TS))}
        cert = enforce_policy(tool=tool, intent_pv=intent2, args_pv=args2,
                              policy_rule_id="TEST", requires_human_approval=False,
                              trace=trace)

        s = trace.summary()
        assert s["total_steps"] == 2
        assert s["ok"] == 1
        assert s["blocked"] == 1
        assert cert["trace"]["steps"] == 1  # Only leaves before this cert
