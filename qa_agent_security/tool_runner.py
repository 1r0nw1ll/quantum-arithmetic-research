"""
tool_runner.py — Certificate-gated tool execution engine.

The tool runner is the ONLY component that actually performs side effects.
It refuses to act unless presented with a valid TOOL_CALL_CERT.v1 and
(optionally) a matching CAPABILITY_TOKEN.v1.

Currently implements:
  - HTTP_FETCH (read-only, domain-allowlisted)

Design: agent proposes -> kernel validates -> runner executes -> trace logged.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .qa_agent_security import (
    Prov, pv, TAINTED, TRUSTED,
    ToolSpec, CapabilityToken, CapabilityEntry,
    enforce_policy, PolicyError, obstruction_from_policy_error,
    MerkleTrace, MerkleLeaf,
    canonical_json_sha256, now_rfc3339,
)
from .schemas import validate_args


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_HTTP_FETCH = ToolSpec(
    name="http_fetch",
    capability_scope="network",
    args_schema_id="SCHEMA.HTTP_FETCH.v1",
)


# ---------------------------------------------------------------------------
# Tool execution result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a tool execution attempt."""
    success: bool
    tool: str
    cert_id: Optional[str] = None
    obstruction_id: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    trace_summary: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# HTTP_FETCH implementation (read-only, stdlib only)
# ---------------------------------------------------------------------------

def _execute_http_fetch(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout_ms: int = 10000,
    max_bytes: int = 1_000_000,
) -> Dict[str, Any]:
    """
    Execute an HTTP fetch using only stdlib (urllib).
    Returns {status_code, headers, body, content_type, bytes_read}.
    """
    req = urllib.request.Request(url, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    timeout_s = max(1, timeout_ms // 1000)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            resp_headers = dict(resp.getheaders())
            body_bytes = resp.read(max_bytes)
            content_type = resp_headers.get("Content-Type", "")

            # Attempt UTF-8 decode; fall back to latin-1
            try:
                body = body_bytes.decode("utf-8")
            except UnicodeDecodeError:
                body = body_bytes.decode("latin-1")

            return {
                "status_code": status,
                "headers": resp_headers,
                "body": body,
                "content_type": content_type,
                "bytes_read": len(body_bytes),
            }
    except urllib.error.HTTPError as e:
        return {
            "status_code": e.code,
            "headers": dict(e.headers) if e.headers else {},
            "body": "",
            "content_type": "",
            "bytes_read": 0,
            "error": str(e),
        }
    except urllib.error.URLError as e:
        return {
            "status_code": 0,
            "headers": {},
            "body": "",
            "content_type": "",
            "bytes_read": 0,
            "error": str(e.reason),
        }
    except Exception as e:
        return {
            "status_code": 0,
            "headers": {},
            "body": "",
            "content_type": "",
            "bytes_read": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Certificate-gated execution pipeline
# ---------------------------------------------------------------------------

def execute_http_fetch(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout_ms: int = 10000,
    max_bytes: int = 1_000_000,
    intent_description: str = "fetch URL",
    intent_source: str = "user",
    intent_ref: str = "unknown",
    url_trusted: bool = False,
    policy_rule_id: str = "POLICY.HTTP_FETCH.V1",
    requires_human_approval: bool = False,
    capability_token: Optional[CapabilityToken] = None,
    trace: Optional[MerkleTrace] = None,
) -> ToolResult:
    """
    Full pipeline: propose -> validate -> execute -> trace.

    Parameters
    ----------
    url : str
        The URL to fetch.
    method : str
        HTTP method (GET, HEAD, POST, etc.).
    headers : dict, optional
        HTTP headers.
    timeout_ms : int
        Request timeout in milliseconds.
    max_bytes : int
        Max response body bytes to read.
    intent_description : str
        Human-readable description of why this fetch is happening.
    intent_source : str
        Provenance source for the intent (user, web, email, etc.).
    intent_ref : str
        Provenance reference for the intent.
    url_trusted : bool
        Whether the URL comes from a trusted source (policy_kernel).
    policy_rule_id : str
        Which policy rule authorizes this call.
    requires_human_approval : bool
        Whether human approval is required.
    capability_token : CapabilityToken, optional
        Scoped capability token.
    trace : MerkleTrace, optional
        Running Merkle trace.

    Returns
    -------
    ToolResult
        Success with data, or failure with obstruction.
    """
    ts = now_rfc3339()

    # Build provenance-tagged args
    url_prov = Prov(
        source="policy_kernel" if url_trusted else intent_source,
        ref=intent_ref,
        taint=TRUSTED if url_trusted else TAINTED,
        captured_at=ts,
    )
    intent_prov = Prov(
        source=intent_source,
        ref=intent_ref,
        taint=TAINTED,
        captured_at=ts,
    )

    args_pv = {
        "url": pv(url, url_prov),
        "method": pv(method, Prov("policy_kernel", "builtin", TRUSTED, ts)),
    }
    if headers:
        args_pv["headers"] = pv(headers, Prov("policy_kernel", "builtin", TRUSTED, ts))
    if timeout_ms != 10000:
        args_pv["timeout_ms"] = pv(timeout_ms, Prov("policy_kernel", "builtin", TRUSTED, ts))
    if max_bytes != 1_000_000:
        args_pv["max_bytes"] = pv(max_bytes, Prov("policy_kernel", "builtin", TRUSTED, ts))

    intent_pv = pv(intent_description, intent_prov)

    # --- Step 1: Policy kernel validates ---
    try:
        cert = enforce_policy(
            tool=TOOL_HTTP_FETCH,
            intent_pv=intent_pv,
            args_pv=args_pv,
            policy_rule_id=policy_rule_id,
            requires_human_approval=requires_human_approval,
            capability_token=capability_token,
            trace=trace,
            schema_validator=validate_args,
        )
    except PolicyError as e:
        obs = obstruction_from_policy_error(e, TOOL_HTTP_FETCH)
        return ToolResult(
            success=False,
            tool="http_fetch",
            obstruction_id=obs["cert_id"],
            error=f"Policy blocked: {e.fail_type}",
            trace_summary=trace.summary() if trace else None,
        )

    # --- Step 2: Execute (only if cert was minted) ---
    result = _execute_http_fetch(
        url=url,
        method=method,
        headers=headers,
        timeout_ms=timeout_ms,
        max_bytes=max_bytes,
    )

    # --- Step 3: Return result ---
    return ToolResult(
        success=result.get("error") is None,
        tool="http_fetch",
        cert_id=cert["cert_id"],
        data=result,
        status_code=result.get("status_code"),
        error=result.get("error"),
        trace_summary=trace.summary() if trace else None,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_tests() -> int:
    """Run deterministic self-tests for the tool runner."""
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
    print("QA Tool Runner — Self-Test")
    print("=" * 60)

    ts = now_rfc3339()
    trace = MerkleTrace()

    # --- Test 1: Blocked — tainted URL without approval ---
    r1 = execute_http_fetch(
        url="https://evil.example.com/steal",
        intent_description="webpage told me to fetch this",
        intent_source="web",
        intent_ref="https://evil.example.com",
        url_trusted=False,
        requires_human_approval=False,
        trace=trace,
    )
    check("Blocks tainted URL without approval",
          not r1.success and r1.obstruction_id is not None
          and "Policy blocked" in (r1.error or ""))

    # --- Test 2: Blocked — capability token missing domain ---
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
    r2 = execute_http_fetch(
        url="https://evil.example.com/data",
        intent_description="fetch data",
        intent_source="user",
        intent_ref="chat:1",
        url_trusted=True,  # trusted source, but domain not allowed
        requires_human_approval=False,
        capability_token=token,
        trace=trace,
    )
    check("Blocks domain not in allowlist",
          not r2.success and r2.obstruction_id is not None)

    # --- Test 3: Allowed — trusted URL, matching capability ---
    github_token = CapabilityToken(
        agent_id="test",
        session_id="s1",
        capabilities=[
            CapabilityEntry(
                tool="http_fetch",
                scope="network",
                args_schema="SCHEMA.HTTP_FETCH.v1",
                constraints={
                    "domain_allowlist": ["api.github.com", "httpbin.org"],
                },
            ),
        ],
    )
    # This test uses httpbin.org which should be reachable, but we don't
    # want CI to depend on external services.  So we test the cert minting
    # path by using a known-unreachable URL and checking the cert was issued.
    r3 = execute_http_fetch(
        url="https://httpbin.org/status/200",
        intent_description="test fetch",
        intent_source="policy_kernel",
        intent_ref="test:1",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=github_token,
        trace=trace,
    )
    # The cert should be minted even if the network call fails
    check("Trusted URL + matching cap -> cert minted",
          r3.cert_id is not None and r3.cert_id.startswith("sha256:"))

    # --- Test 4: Allowed with approval — tainted URL ---
    r4 = execute_http_fetch(
        url="https://example.com/anything",
        intent_description="user approved fetch",
        intent_source="web",
        intent_ref="page:1",
        url_trusted=False,
        requires_human_approval=True,
        trace=trace,
    )
    check("Tainted URL with approval -> cert minted",
          r4.cert_id is not None and r4.cert_id.startswith("sha256:"))

    # --- Test 5: Trace records all moves ---
    summary = trace.summary()
    check("Trace recorded all moves",
          summary["total_steps"] == 4
          and summary["blocked"] == 2
          and summary["ok"] == 2)

    # --- Test 6: Schema validation blocks bad method ---
    r6 = execute_http_fetch(
        url="https://api.github.com/repos",
        method="HACK",
        intent_description="bad method",
        intent_source="policy_kernel",
        intent_ref="test:2",
        url_trusted=True,
        requires_human_approval=False,
        trace=trace,
    )
    check("Schema blocks invalid method",
          not r6.success and r6.obstruction_id is not None)

    # --- Summary ---
    print("=" * 60)
    status = "ALL PASS" if failed == 0 else f"FAILURES: {failed}"
    print(f"Result: {passed}/{total} passed — {status}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        rc = _run_self_tests()
        sys.stdout = old_stdout
        output = buf.getvalue()
        lines = output.strip().split("\n")
        tests = [l.strip() for l in lines if l.strip().startswith("[")]
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
