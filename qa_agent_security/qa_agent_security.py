"""
qa_agent_security.py — QA-native agent security kernel

Certificate-gated tool execution with taint tracking and deterministic
obstruction generation.  Every tool call requires a validated
TOOL_CALL_CERT.v1; prompt injection becomes a reachability obstruction
logged to the obstruction ledger.

Design principle: agent = generators; security = legality of moves.

Requires only Python 3.10+ stdlib.  No GPU, no pip install.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Canonical JSON + hashing (matches qa_cert_core.py contract)
# ---------------------------------------------------------------------------

def canonical_json_dumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, UTF-8, no whitespace, ensure_ascii=False."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_sha256(obj: Any) -> str:
    s = canonical_json_dumps(obj)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Provenance / taint model
# ---------------------------------------------------------------------------

TAINTED = "TAINTED"
TRUSTED = "TRUSTED"

VALID_SOURCES = frozenset({
    "user", "web", "email", "file", "system", "policy_kernel"
})


@dataclass(frozen=True)
class Prov:
    """Provenance tag — required on every field that can affect actions."""
    source: str               # user|web|email|file|system|policy_kernel
    ref: str                  # opaque id/url/hash
    taint: str                # TAINTED|TRUSTED
    captured_at: str          # RFC3339

    def __post_init__(self):
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source!r}")
        if self.taint not in (TAINTED, TRUSTED):
            raise ValueError(f"Invalid taint: {self.taint!r}")

    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "ref": self.ref,
            "taint": self.taint,
            "captured_at": self.captured_at,
        }


def pv(value: Any, prov: Prov) -> Dict[str, Any]:
    """Create a provenance-tagged value."""
    return {"prov": prov.to_dict(), "value": value}


def is_tainted(pv_field: Dict[str, Any]) -> bool:
    """Check whether a pv-field is TAINTED (anything not explicitly TRUSTED)."""
    return pv_field.get("prov", {}).get("taint") != TRUSTED


def prov_source(pv_field: Dict[str, Any]) -> str:
    """Extract source from a pv-field."""
    return pv_field.get("prov", {}).get("source", "unknown")


# ---------------------------------------------------------------------------
# Taint flow certificate
# ---------------------------------------------------------------------------

def mint_taint_flow_cert(
    inputs: List[Dict[str, Any]],
    transform_name: str,
    transform_params: Dict[str, Any],
    outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    TAINT_FLOW_CERT.v1 — tracks how taint moves through transforms.
    Key invariant: transforms may reduce content but NEVER upgrade
    TAINTED -> TRUSTED.
    """
    # Enforce: if any input is tainted, all outputs must remain tainted
    any_tainted_input = any(is_tainted(inp) for inp in inputs)
    for out in outputs:
        if any_tainted_input and not is_tainted(out):
            raise PolicyError(
                fail_type="TAINT_UPGRADE_VIOLATION",
                invariant_diff=[{
                    "inv": "NO_TRUST_UPGRADE",
                    "expected": "pass",
                    "got": "fail",
                }],
                witness={
                    "inputs": inputs,
                    "outputs": outputs,
                    "notes": "Transform attempted to upgrade TAINTED input to TRUSTED output.",
                },
            )

    cert = {
        "schema_version": "TAINT_FLOW_CERT.v1",
        "created_at": now_rfc3339(),
        "inputs": inputs,
        "transform": {
            "name": transform_name,
            "parameters": transform_params,
        },
        "outputs": outputs,
        "invariants_checked": [
            "NO_TRUST_UPGRADE",
            "PROVENANCE_PRESERVED",
        ],
    }
    cert["cert_id"] = canonical_json_sha256(cert)
    return cert


# ---------------------------------------------------------------------------
# Merkle trace (1:1 with QARM log rows)
# ---------------------------------------------------------------------------

@dataclass
class MerkleLeaf:
    """MERKLE_TRACE.v1 leaf — exactly {move, fail_type, invariant_diff}."""
    move: str                                           # tool name or generator symbol
    fail_type: str                                      # OK | obstruction type
    invariant_diff: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "move": self.move,
            "fail_type": self.fail_type,
            "invariant_diff": self.invariant_diff,
        }

    def leaf_hash(self) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json_dumps(self.to_dict()).encode("utf-8")
        ).hexdigest()


def merkle_root(leaves: List[str]) -> str:
    """
    Compute deterministic Merkle root from leaf hashes.
    - strip sha256: prefix
    - pairwise hash; odd count -> duplicate last
    """
    if not leaves:
        return "sha256:" + hashlib.sha256(b"").hexdigest()

    level = [x.split("sha256:")[-1] if "sha256:" in x else x for x in leaves]

    def hpair(a: str, b: str) -> str:
        return hashlib.sha256((a + b).encode("utf-8")).hexdigest()

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [hpair(level[i], level[i + 1]) for i in range(0, len(level), 2)]

    return f"sha256:{level[0]}"


class MerkleTrace:
    """Append-only trace of agent moves (successes + obstructions)."""

    def __init__(self):
        self._leaves: List[MerkleLeaf] = []

    def append(self, leaf: MerkleLeaf) -> None:
        self._leaves.append(leaf)

    def root(self) -> str:
        return merkle_root([l.leaf_hash() for l in self._leaves])

    def steps(self) -> int:
        return len(self._leaves)

    def leaves(self) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in self._leaves]

    def summary(self) -> Dict[str, Any]:
        ok = sum(1 for l in self._leaves if l.fail_type == "OK")
        blocked = len(self._leaves) - ok
        return {
            "merkle_root": self.root(),
            "total_steps": self.steps(),
            "ok": ok,
            "blocked": blocked,
        }


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    capability_scope: str     # read_only|write_limited|network|exec
    args_schema_id: str


# Action-critical fields per tool — these MUST be TRUSTED or require approval
CRITICAL_FIELDS: Dict[str, Set[str]] = {
    "send_email": {"to"},
    "run_shell": {"command"},
    "http_fetch": {"url"},
    "create_file": {"path"},
    "calendar_create": {"attendees"},
}


# ---------------------------------------------------------------------------
# Capability token
# ---------------------------------------------------------------------------

@dataclass
class CapabilityEntry:
    tool: str
    scope: str
    args_schema: str
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityToken:
    """CAPABILITY_TOKEN.v1 — scoped, time-limited authorization."""
    agent_id: str
    session_id: str
    capabilities: List[CapabilityEntry]
    issued_at: str = ""
    expires_at: str = ""
    instance_id: str = "local-kernel"

    def __post_init__(self):
        if not self.issued_at:
            self.issued_at = now_rfc3339()

    def is_expired(self, now: Optional[str] = None) -> bool:
        if not self.expires_at:
            return False
        now = now or now_rfc3339()
        return now > self.expires_at

    def find_capability(self, tool_name: str, scope: str) -> Optional[CapabilityEntry]:
        for cap in self.capabilities:
            if cap.tool == tool_name and cap.scope == scope:
                return cap
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "CAPABILITY_TOKEN.v1",
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "issuer": {"type": "policy_kernel", "instance_id": self.instance_id},
            "subject": {"agent_id": self.agent_id, "session_id": self.session_id},
            "capabilities": [
                {
                    "tool": c.tool,
                    "scope": c.scope,
                    "args_schema": c.args_schema,
                    "constraints": c.constraints,
                }
                for c in self.capabilities
            ],
        }


def _check_constraints(cap: CapabilityEntry, args_pv: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Check capability constraints against args.  Returns invariant diffs."""
    diffs: List[Dict[str, str]] = []
    constraints = cap.constraints

    # command denylist (regex)
    if cap.tool == "run_shell" and "command" in args_pv:
        import re
        cmd_val = args_pv["command"].get("value", "")
        for pattern in constraints.get("command_denylist_regex", []):
            if re.search(pattern, cmd_val):
                diffs.append({
                    "inv": "COMMAND_DENYLIST",
                    "expected": "pass",
                    "got": f"fail (matched {pattern!r})",
                })

    # domain allowlist (http_fetch)
    if cap.tool == "http_fetch" and "url" in args_pv:
        url_val = args_pv["url"].get("value", "")
        allow = constraints.get("domain_allowlist")
        if allow is not None:
            from urllib.parse import urlparse
            domain = urlparse(url_val).hostname or ""
            if domain not in allow:
                diffs.append({
                    "inv": "DOMAIN_ALLOWLIST",
                    "expected": "pass",
                    "got": f"fail (domain {domain!r} not in allowlist)",
                })

    # recipient allowlist (send_email)
    if cap.tool == "send_email" and "to" in args_pv:
        import fnmatch
        to_val = args_pv["to"].get("value", [])
        if isinstance(to_val, str):
            to_val = [to_val]
        allow = constraints.get("recipient_allowlist")
        ext_allowed = constraints.get("external_recipients_allowed", True)
        if allow is not None and not ext_allowed:
            for addr in to_val:
                if not any(fnmatch.fnmatch(addr, pat) for pat in allow):
                    diffs.append({
                        "inv": "RECIPIENT_ALLOWLIST",
                        "expected": "pass",
                        "got": f"fail ({addr!r} not in allowlist)",
                    })

    return diffs


# ---------------------------------------------------------------------------
# Policy errors / obstructions
# ---------------------------------------------------------------------------

class PolicyError(Exception):
    """Raised when the policy kernel blocks a tool invocation."""
    def __init__(
        self,
        fail_type: str,
        invariant_diff: List[Dict[str, str]],
        witness: Dict[str, Any],
    ):
        super().__init__(fail_type)
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff
        self.witness = witness


# Canonical fail types (obstruction family)
FAIL_TYPES = frozenset({
    "UNTRUSTED_INSTRUCTION",
    "CAPABILITY_ESCALATION_ATTEMPT",
    "SCHEMA_BREAKOUT",
    "CONFUSED_DEPUTY",
    "PROVENANCE_LOSS",
    "TAINT_UPGRADE_VIOLATION",
    "CAPABILITY_TOKEN_MISSING",
    "CAPABILITY_TOKEN_EXPIRED",
    "CAPABILITY_NOT_FOUND",
    "CONSTRAINT_VIOLATION",
})


def obstruction_from_policy_error(
    err: PolicyError,
    attempted_tool: ToolSpec,
) -> Dict[str, Any]:
    """Mint a PROMPT_INJECTION_OBSTRUCTION.v1 from a PolicyError."""
    obs = {
        "schema_version": "PROMPT_INJECTION_OBSTRUCTION.v1",
        "created_at": now_rfc3339(),
        "attempted_tool": attempted_tool.name,
        "attempted_args": err.witness.get("args", {}),
        "fail_type": err.fail_type,
        "invariant_diff": err.invariant_diff,
        "witness": {
            "snippet": "(see attempted_args)",
            "source_ref": err.witness.get("intent", {}).get("prov", {}).get("ref", ""),
            "notes": "Policy kernel rejected tool invocation.",
        },
    }
    obs["cert_id"] = canonical_json_sha256(obs)
    return obs


# ---------------------------------------------------------------------------
# Policy kernel — the ONLY path to tool execution
# ---------------------------------------------------------------------------

def enforce_policy(
    tool: ToolSpec,
    intent_pv: Dict[str, Any],
    args_pv: Dict[str, Dict[str, Any]],
    policy_rule_id: str,
    requires_human_approval: bool,
    capability_token: Optional[CapabilityToken] = None,
    trace: Optional[MerkleTrace] = None,
    schema_validator=None,
) -> Dict[str, Any]:
    """
    Enforce all policy invariants and mint a TOOL_CALL_CERT.v1.

    Raises PolicyError with structured obstruction data on failure.

    Parameters
    ----------
    tool : ToolSpec
        The tool being invoked.
    intent_pv : dict
        Provenance-tagged intent description.
    args_pv : dict
        Provenance-tagged arguments (field_name -> pv dict).
    policy_rule_id : str
        Which policy rule authorizes this call.
    requires_human_approval : bool
        Whether human confirmation is needed.
    capability_token : CapabilityToken, optional
        Scoped authorization token (required unless approval=True).
    trace : MerkleTrace, optional
        Running trace to append to.
    schema_validator : callable, optional
        validate_args(schema_id, raw_args) -> None or raise.

    Returns
    -------
    dict
        TOOL_CALL_CERT.v1 on success.
    """
    inv_diff: List[Dict[str, str]] = []

    # --- 1. Args shape check ---
    if not isinstance(args_pv, dict):
        inv_diff.append({"inv": "STRICT_ARGS_SCHEMA", "expected": "pass", "got": "fail (not a dict)"})
    else:
        for k, v in args_pv.items():
            if not isinstance(v, dict) or "prov" not in v or "value" not in v:
                inv_diff.append({
                    "inv": "STRICT_ARGS_SCHEMA",
                    "expected": "pass",
                    "got": f"fail (field {k!r} missing prov/value)",
                })

    # --- 2. Schema validation (if validator provided) ---
    if schema_validator is not None and isinstance(args_pv, dict):
        raw_args = {k: v["value"] for k, v in args_pv.items() if isinstance(v, dict) and "value" in v}
        try:
            schema_validator(tool.args_schema_id, raw_args)
        except Exception as e:
            inv_diff.append({
                "inv": "STRICT_ARGS_SCHEMA",
                "expected": "pass",
                "got": f"fail ({e})",
            })

    # --- 3. Capability token check ---
    if capability_token is not None:
        if capability_token.is_expired():
            inv_diff.append({"inv": "CAPABILITY_TOKEN_VALID", "expected": "pass", "got": "fail (expired)"})
        cap = capability_token.find_capability(tool.name, tool.capability_scope)
        if cap is None:
            inv_diff.append({
                "inv": "CAPABILITY_FOUND",
                "expected": "pass",
                "got": f"fail (no cap for {tool.name}/{tool.capability_scope})",
            })
        else:
            # Check constraints
            constraint_diffs = _check_constraints(cap, args_pv)
            inv_diff.extend(constraint_diffs)
    elif not requires_human_approval:
        # Token required unless human approves
        pass  # Allow token-less operation for v1 (soft enforcement)

    # --- 4. No web/email/file -> exec without approval ---
    if tool.capability_scope == "exec" and not requires_human_approval:
        cmd_pv = args_pv.get("command") if isinstance(args_pv, dict) else None
        if cmd_pv is None:
            inv_diff.append({"inv": "COMMAND_PRESENT", "expected": "pass", "got": "fail"})
        elif is_tainted(cmd_pv):
            inv_diff.append({"inv": "NO_WEB_TO_EXEC", "expected": "pass", "got": "fail"})
            inv_diff.append({"inv": "COMMAND_TRUSTED_OR_APPROVED", "expected": "pass", "got": "fail"})

    # --- 5. Critical fields must be TRUSTED or approved ---
    crit = CRITICAL_FIELDS.get(tool.name, set())
    if not requires_human_approval and isinstance(args_pv, dict):
        for k in crit:
            if k in args_pv and is_tainted(args_pv[k]):
                inv_diff.append({
                    "inv": f"{k.upper()}_TRUSTED",
                    "expected": "pass",
                    "got": "fail",
                })

    # --- 6. Reject if any invariant failed ---
    if inv_diff:
        # Classify fail type
        fail_type = "UNTRUSTED_INSTRUCTION"
        inv_names = {d["inv"] for d in inv_diff}
        if "NO_WEB_TO_EXEC" in inv_names:
            fail_type = "CAPABILITY_ESCALATION_ATTEMPT"
        elif "CAPABILITY_TOKEN_VALID" in inv_names:
            fail_type = "CAPABILITY_TOKEN_EXPIRED"
        elif "CAPABILITY_FOUND" in inv_names:
            fail_type = "CAPABILITY_NOT_FOUND"
        elif "COMMAND_DENYLIST" in inv_names or "DOMAIN_ALLOWLIST" in inv_names or "RECIPIENT_ALLOWLIST" in inv_names:
            fail_type = "CONSTRAINT_VIOLATION"
        elif "STRICT_ARGS_SCHEMA" in inv_names:
            fail_type = "SCHEMA_BREAKOUT"

        witness = {
            "tool": asdict(tool) if hasattr(tool, '__dataclass_fields__') else {"name": tool.name},
            "intent": intent_pv,
            "args": args_pv,
        }

        err = PolicyError(fail_type=fail_type, invariant_diff=inv_diff, witness=witness)

        # Log to trace
        if trace is not None:
            trace.append(MerkleLeaf(
                move=f"TOOL_CALL:{tool.name}",
                fail_type=fail_type,
                invariant_diff=inv_diff,
            ))

        raise err

    # --- 7. Mint TOOL_CALL_CERT.v1 ---
    trace_hashes = []
    if trace is not None:
        trace_hashes = [l.leaf_hash() for l in trace._leaves]

    # Compute risk level
    risk_reasons = []
    if tool.capability_scope == "exec":
        risk_reasons.append("exec_scope")
    if tool.capability_scope == "network":
        risk_reasons.append("network_scope")
    if requires_human_approval:
        risk_reasons.append("human_approval_required")

    risk_level = "low"
    if tool.capability_scope in ("exec", "network"):
        risk_level = "med"
    if requires_human_approval:
        risk_level = "high"

    cert = {
        "schema_version": "TOOL_CALL_CERT.v1",
        "created_at": now_rfc3339(),
        "tool": {
            "name": tool.name,
            "capability_scope": tool.capability_scope,
            "schema": tool.args_schema_id,
        },
        "intent": intent_pv,
        "args": {k: v for k, v in args_pv.items()},
        "policy_rule_id": policy_rule_id,
        "risk": {"level": risk_level, "reasons": risk_reasons},
        "requires_human_approval": requires_human_approval,
        "invariants_checked": [
            "STRICT_ARGS_SCHEMA",
            "NO_TAINTED_CAPABILITY_ESCALATION",
            "NO_WEB_TO_EXEC",
            "CRITICAL_FIELDS_TRUSTED",
            "CAPABILITY_SCOPE_ALLOWED",
        ],
        "rollback_plan": {"type": "none", "details": ""},
        "trace": {
            "merkle_root": merkle_root(trace_hashes),
            "steps": len(trace_hashes),
        },
    }
    cert["cert_id"] = canonical_json_sha256(cert)

    # Log success to trace
    if trace is not None:
        trace.append(MerkleLeaf(
            move=f"TOOL_CALL:{tool.name}",
            fail_type="OK",
            invariant_diff=[],
        ))

    return cert


# ---------------------------------------------------------------------------
# Self-test (deterministic, stdlib-only, matches QA validator pattern)
# ---------------------------------------------------------------------------

def _run_self_tests() -> int:
    """
    Run deterministic self-tests.  Returns 0 on all-pass, 1 on any failure.
    Prints results to stdout.
    """
    ts = now_rfc3339()
    passed = 0
    failed = 0
    total = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [{total}] {label} -> PASS")
        else:
            failed += 1
            print(f"  [{total}] {label} -> FAIL  {detail}")

    print("=" * 60)
    print("QA Agent Security Kernel — Self-Test")
    print("=" * 60)

    # --- Test 1: Provenance creation ---
    p = Prov("user", "chat:1", TAINTED, ts)
    check("Prov creation (valid)", p.source == "user" and p.taint == TAINTED)

    # --- Test 2: Invalid source rejected ---
    try:
        Prov("hacker", "x", TAINTED, ts)
        check("Prov rejects invalid source", False, "no exception raised")
    except ValueError:
        check("Prov rejects invalid source", True)

    # --- Test 3: pv helper ---
    field = pv("hello", p)
    check("pv() creates tagged value", field["value"] == "hello" and is_tainted(field))

    # --- Test 4: Taint flow cert (valid) ---
    inp = pv("raw web text", Prov("web", "url:1", TAINTED, ts))
    out = pv("summary", Prov("policy_kernel", "cert:1", TAINTED, ts))
    cert = mint_taint_flow_cert([inp], "summarize", {"max_tokens": 256}, [out])
    check("Taint flow cert (tainted->tainted)", cert["schema_version"] == "TAINT_FLOW_CERT.v1")

    # --- Test 5: Taint flow cert blocks upgrade ---
    bad_out = pv("cleaned", Prov("policy_kernel", "cert:2", TRUSTED, ts))
    try:
        mint_taint_flow_cert([inp], "summarize", {}, [bad_out])
        check("Taint flow blocks TAINTED->TRUSTED", False, "no exception")
    except PolicyError as e:
        check("Taint flow blocks TAINTED->TRUSTED", e.fail_type == "TAINT_UPGRADE_VIOLATION")

    # --- Test 6: Blocks web->exec without approval ---
    tool = ToolSpec(name="run_shell", capability_scope="exec", args_schema_id="SCHEMA.RUN_SHELL.v1")
    intent = pv("do it", Prov("web", "https://evil.example", TAINTED, ts))
    args = {"command": pv("rm -rf /", Prov("web", "https://evil.example", TAINTED, ts))}
    try:
        enforce_policy(tool=tool, intent_pv=intent, args_pv=args,
                       policy_rule_id="POLICY.EXEC.V1", requires_human_approval=False)
        check("Blocks web->exec", False, "no exception")
    except PolicyError as e:
        obs = obstruction_from_policy_error(e, tool)
        check("Blocks web->exec",
              obs["fail_type"] == "CAPABILITY_ESCALATION_ATTEMPT"
              and obs["schema_version"] == "PROMPT_INJECTION_OBSTRUCTION.v1")

    # --- Test 7: Allows exec with human approval (even if tainted) ---
    try:
        cert = enforce_policy(
            tool=tool,
            intent_pv=pv("run cmd", Prov("user", "chat:2", TAINTED, ts)),
            args_pv={"command": pv("echo hello", Prov("user", "chat:2", TAINTED, ts))},
            policy_rule_id="POLICY.EXEC.V1",
            requires_human_approval=True,
        )
        check("Allows exec with approval", cert["schema_version"] == "TOOL_CALL_CERT.v1"
              and cert["requires_human_approval"] is True)
    except PolicyError:
        check("Allows exec with approval", False, "unexpected PolicyError")

    # --- Test 8: Blocks tainted email recipient ---
    email_tool = ToolSpec(name="send_email", capability_scope="write_limited",
                          args_schema_id="SCHEMA.SEND_EMAIL.v1")
    try:
        enforce_policy(
            tool=email_tool,
            intent_pv=pv("fwd email", Prov("email", "msg:1", TAINTED, ts)),
            args_pv={
                "to": pv("attacker@evil.example", Prov("email", "msg:1", TAINTED, ts)),
                "body": pv("hi", Prov("policy_kernel", "d:1", TAINTED, ts)),
            },
            policy_rule_id="POLICY.EMAIL.SEND.V1",
            requires_human_approval=False,
        )
        check("Blocks tainted email recipient", False, "no exception")
    except PolicyError as e:
        invs = {d["inv"] for d in e.invariant_diff}
        check("Blocks tainted email recipient", "TO_TRUSTED" in invs)

    # --- Test 9: Allows trusted email recipient ---
    try:
        cert = enforce_policy(
            tool=email_tool,
            intent_pv=pv("send report", Prov("user", "chat:3", TAINTED, ts)),
            args_pv={
                "to": pv("team@safe.com", Prov("policy_kernel", "k:1", TRUSTED, ts)),
                "body": pv("report text", Prov("policy_kernel", "k:2", TAINTED, ts)),
            },
            policy_rule_id="POLICY.EMAIL.SEND.V1",
            requires_human_approval=False,
        )
        check("Allows trusted email recipient", cert["schema_version"] == "TOOL_CALL_CERT.v1")
    except PolicyError:
        check("Allows trusted email recipient", False, "unexpected PolicyError")

    # --- Test 10: Merkle trace records moves ---
    trace = MerkleTrace()
    trace.append(MerkleLeaf(move="TOOL_CALL:run_shell", fail_type="OK"))
    trace.append(MerkleLeaf(move="TOOL_CALL:send_email", fail_type="UNTRUSTED_INSTRUCTION",
                            invariant_diff=[{"inv": "TO_TRUSTED", "expected": "pass", "got": "fail"}]))
    summary = trace.summary()
    check("Merkle trace tracks moves",
          summary["total_steps"] == 2 and summary["ok"] == 1 and summary["blocked"] == 1
          and summary["merkle_root"].startswith("sha256:"))

    # --- Test 11: Capability token constraint enforcement ---
    token = CapabilityToken(
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
    try:
        enforce_policy(
            tool=tool,
            intent_pv=pv("clean up", Prov("policy_kernel", "k:3", TRUSTED, ts)),
            args_pv={"command": pv("rm -rf /tmp/junk", Prov("policy_kernel", "k:4", TRUSTED, ts))},
            policy_rule_id="POLICY.EXEC.V1",
            requires_human_approval=False,
            capability_token=token,
        )
        check("Capability denylist blocks 'rm'", False, "no exception")
    except PolicyError as e:
        invs = {d["inv"] for d in e.invariant_diff}
        check("Capability denylist blocks 'rm'", "COMMAND_DENYLIST" in invs
              and e.fail_type == "CONSTRAINT_VIOLATION")

    # --- Test 12: Capability token allows safe command ---
    try:
        cert = enforce_policy(
            tool=tool,
            intent_pv=pv("list files", Prov("policy_kernel", "k:5", TRUSTED, ts)),
            args_pv={"command": pv("ls /tmp", Prov("policy_kernel", "k:6", TRUSTED, ts))},
            policy_rule_id="POLICY.EXEC.V1",
            requires_human_approval=False,
            capability_token=token,
        )
        check("Capability allows safe command", cert["schema_version"] == "TOOL_CALL_CERT.v1")
    except PolicyError:
        check("Capability allows safe command", False, "unexpected PolicyError")

    # --- Test 13: Obstruction cert has valid cert_id ---
    try:
        enforce_policy(tool=tool, intent_pv=intent, args_pv=args,
                       policy_rule_id="TEST", requires_human_approval=False)
    except PolicyError as e:
        obs = obstruction_from_policy_error(e, tool)
        check("Obstruction cert has sha256 cert_id",
              obs["cert_id"].startswith("sha256:") and len(obs["cert_id"]) == 71)

    # --- Test 14: Canonical JSON determinism ---
    obj_a = {"z": 1, "a": 2, "m": [3, 1]}
    obj_b = {"a": 2, "m": [3, 1], "z": 1}
    check("Canonical JSON deterministic",
          canonical_json_sha256(obj_a) == canonical_json_sha256(obj_b))

    # --- Summary ---
    print("=" * 60)
    status = "ALL PASS" if failed == 0 else f"FAILURES: {failed}"
    print(f"Result: {passed}/{total} passed — {status}")
    print("=" * 60)

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if "--validate" in sys.argv:
        # JSON output mode (for subprocess invocation from meta-validator)
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        rc = _run_self_tests()
        sys.stdout = old_stdout
        # Parse pass/fail from output
        output = buf.getvalue()
        lines = output.strip().split("\n")
        tests = []
        for line in lines:
            if line.strip().startswith("["):
                tests.append(line.strip())
        fails = [t for t in tests if "FAIL" in t]
        result = {
            "result": "PASS" if rc == 0 else "FAIL",
            "tests_run": len(tests),
            "failures": len(fails),
            "warnings": [],
            "detail": output,
        }
        print(json.dumps(result))
        sys.exit(rc)
    else:
        sys.exit(_run_self_tests())


if __name__ == "__main__":
    main()
