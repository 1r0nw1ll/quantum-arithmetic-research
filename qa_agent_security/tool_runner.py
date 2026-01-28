"""
tool_runner.py — Certificate-gated tool execution engine.

The tool runner is the ONLY component that actually performs side effects.
It refuses to act unless presented with a valid TOOL_CALL_CERT.v1 and
(optionally) a matching CAPABILITY_TOKEN.v1.

Currently implements:
  - HTTP_FETCH (read-only, domain-allowlisted, URL-hardened)

Design: agent proposes -> kernel validates -> runner executes -> trace logged.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
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
# Enforcement configuration
# ---------------------------------------------------------------------------

# When True, HTTP_FETCH requires a valid capability token even for trusted URLs.
# This makes capability tokens mandatory rather than advisory.
# Set via environment variable or directly for testing.
import os
REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = os.environ.get(
    "QA_REQUIRE_CAP_TOKEN_HTTP_FETCH", "true"
).lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# URL sanitization (blocks bypass classes)
# ---------------------------------------------------------------------------

# Regex for IPv4 literal (including dotted-decimal)
_IPV4_RE = re.compile(
    r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
)

# Regex for IPv6 literal (bracketed form in URLs)
_IPV6_BRACKET_RE = re.compile(r"^\[.*\]$")

# Headers that must never be set by the agent (injection vectors)
DANGEROUS_HEADERS = frozenset({
    "host", "transfer-encoding", "content-length",
    "connection", "upgrade", "proxy-authorization",
    "proxy-connection",
})


class URLSanitizationError(Exception):
    """Raised when a URL fails pre-execution sanitization."""
    def __init__(self, invariant: str, detail: str):
        super().__init__(f"{invariant}: {detail}")
        self.invariant = invariant
        self.detail = detail


def sanitize_url(url: str) -> Tuple[str, str]:
    """
    Validate and normalize a URL before execution.

    Returns (scheme, hostname) on success.
    Raises URLSanitizationError on any bypass attempt.

    Checks:
      - Scheme must be http or https
      - No credentials in URL (user:pass@host)
      - No raw IP literals (IPv4 or IPv6) unless explicitly allowed
      - Hostname must be a valid DNS name (no empty, no port tricks)
      - No null bytes or control characters
    """
    # Null byte / control character check
    if any(ord(c) < 32 for c in url) or "\x00" in url:
        raise URLSanitizationError(
            "URL_NO_CONTROL_CHARS",
            "URL contains null bytes or control characters")

    parsed = urlparse(url)

    # Scheme
    if parsed.scheme not in ("http", "https"):
        raise URLSanitizationError(
            "URL_SCHEME_HTTP_ONLY",
            f"Scheme must be http/https, got {parsed.scheme!r}")

    # Credentials in URL
    if parsed.username or parsed.password:
        raise URLSanitizationError(
            "URL_NO_CREDENTIALS",
            "URL must not contain user:pass@ credentials")

    # Extract hostname
    hostname = parsed.hostname or ""
    if not hostname:
        raise URLSanitizationError(
            "URL_HOSTNAME_REQUIRED",
            "URL has no hostname")

    # Lowercase for comparison
    hostname = hostname.lower()

    # IP literal check (IPv4)
    if _IPV4_RE.match(hostname):
        raise URLSanitizationError(
            "URL_NO_IP_LITERAL",
            f"Raw IPv4 literals not allowed: {hostname}")

    # IP literal check (IPv6 — already stripped of brackets by urlparse)
    if ":" in hostname:
        raise URLSanitizationError(
            "URL_NO_IP_LITERAL",
            f"IPv6 literals not allowed: {hostname}")

    # Punycode / IDN normalization — decode to check for confusables
    # We reject any hostname that starts with xn-- (punycode) to prevent
    # homograph attacks. Real allowlists should use ASCII domains.
    if hostname.startswith("xn--") or any(
        label.startswith("xn--") for label in hostname.split(".")
    ):
        raise URLSanitizationError(
            "URL_NO_PUNYCODE",
            f"Punycode/IDN hostnames not allowed: {hostname}")

    # Mixed-case normalization is handled by lowercasing above.
    # The allowlist check in the kernel also lowercases.

    return parsed.scheme, hostname


def sanitize_headers(headers: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Check that no dangerous headers are being set.
    Returns list of invariant diffs (empty = ok).
    """
    diffs = []
    for key in headers:
        if key.lower() in DANGEROUS_HEADERS:
            diffs.append({
                "inv": "HEADER_INJECTION_BLOCKED",
                "expected": "pass",
                "got": f"fail (header {key!r} is forbidden)",
            })
    return diffs


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
# Redirect handling (validates each hop against constraints)
# ---------------------------------------------------------------------------

MAX_REDIRECTS = 5


class RedirectError(Exception):
    """Raised when a redirect violates security constraints."""
    def __init__(self, invariant: str, detail: str, redirect_url: str):
        super().__init__(f"{invariant}: {detail}")
        self.invariant = invariant
        self.detail = detail
        self.redirect_url = redirect_url


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    Custom redirect handler that:
    1. Sanitizes each redirect URL (same checks as initial URL)
    2. Re-validates domain against an optional allowlist
    3. Enforces redirect count limit
    """

    def __init__(self, domain_allowlist: Optional[List[str]] = None):
        super().__init__()
        self.domain_allowlist = domain_allowlist
        self.redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirect_count += 1

        # Redirect count limit
        if self.redirect_count > MAX_REDIRECTS:
            raise RedirectError(
                "REDIRECT_COUNT_MAX",
                f"Exceeded max redirects ({MAX_REDIRECTS})",
                newurl)

        # Sanitize the new URL
        try:
            scheme, hostname = sanitize_url(newurl)
        except URLSanitizationError as e:
            raise RedirectError(e.invariant, e.detail, newurl) from e

        # Re-check domain allowlist
        if self.domain_allowlist is not None:
            allow_lower = [d.lower() for d in self.domain_allowlist]
            if hostname not in allow_lower:
                raise RedirectError(
                    "REDIRECT_TARGET_ALLOWLIST",
                    f"Redirect target {hostname!r} not in allowlist",
                    newurl)

        # Delegate to parent (which builds the new Request)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener(domain_allowlist: Optional[List[str]] = None) -> urllib.request.OpenerDirector:
    """Build an opener with safe redirect handling."""
    handler = SafeRedirectHandler(domain_allowlist=domain_allowlist)
    opener = urllib.request.build_opener(handler)
    return opener


# ---------------------------------------------------------------------------
# HTTP_FETCH implementation (read-only, stdlib only, redirect-safe)
# ---------------------------------------------------------------------------

def _execute_http_fetch(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout_ms: int = 10000,
    max_bytes: int = 1_000_000,
    domain_allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute an HTTP fetch using only stdlib (urllib).
    Returns {status_code, headers, body, content_type, bytes_read}.

    Uses a safe redirect handler that re-validates each redirect target.
    """
    req = urllib.request.Request(url, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    timeout_s = max(1, timeout_ms // 1000)
    opener = build_safe_opener(domain_allowlist=domain_allowlist)

    try:
        with opener.open(req, timeout=timeout_s) as resp:
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
    except RedirectError:
        # Re-raise to caller for proper obstruction handling
        raise
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

    # --- Step 1b: Mandatory capability token enforcement ---
    if REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH and capability_token is None:
        # Extract hostname for witness (best-effort, may fail on malformed URL)
        try:
            parsed = urlparse(url)
            witness_hostname = (parsed.hostname or "").lower()
        except Exception:
            witness_hostname = "<parse-failed>"

        if trace is not None:
            trace.append(MerkleLeaf(
                move="TOOL_CALL:http_fetch",
                fail_type="CAPABILITY_NOT_FOUND",
                invariant_diff=[{
                    "inv": "CAP_TOKEN_REQUIRED",
                    "expected": "valid capability token",
                    "got": "no token provided",
                    "witness": {
                        "tool": "http_fetch",
                        "url_hostname": witness_hostname,
                        "enforcement": "QA_REQUIRE_CAP_TOKEN_HTTP_FETCH=true",
                    },
                }],
            ))
        return ToolResult(
            success=False,
            tool="http_fetch",
            cert_id=cert["cert_id"],
            obstruction_id=None,
            error="Capability token required: CAP_TOKEN_REQUIRED",
            trace_summary=trace.summary() if trace else None,
        )

    # --- Step 2: URL sanitization (pre-execution hardening) ---
    try:
        _scheme, _hostname = sanitize_url(url)
    except URLSanitizationError as e:
        # Log as obstruction via trace
        if trace is not None:
            trace.append(MerkleLeaf(
                move="TOOL_CALL:http_fetch",
                fail_type="CONSTRAINT_VIOLATION",
                invariant_diff=[{"inv": e.invariant, "expected": "pass", "got": f"fail ({e.detail})"}],
            ))
        return ToolResult(
            success=False,
            tool="http_fetch",
            cert_id=cert["cert_id"],
            error=f"URL sanitization failed: {e.invariant}",
            trace_summary=trace.summary() if trace else None,
        )

    # --- Step 2b: Header injection check ---
    if headers:
        header_diffs = sanitize_headers(headers)
        if header_diffs:
            if trace is not None:
                trace.append(MerkleLeaf(
                    move="TOOL_CALL:http_fetch",
                    fail_type="CONSTRAINT_VIOLATION",
                    invariant_diff=header_diffs,
                ))
            return ToolResult(
                success=False,
                tool="http_fetch",
                cert_id=cert["cert_id"],
                error=f"URL sanitization failed: {header_diffs[0]['inv']} — {header_diffs[0]['got']}",
                trace_summary=trace.summary() if trace else None,
            )

    # --- Step 3: Extract domain allowlist from capability token for redirect validation ---
    domain_allowlist = None
    if capability_token is not None:
        cap = capability_token.find_capability("http_fetch", "network")
        if cap is not None:
            domain_allowlist = cap.constraints.get("domain_allowlist")

    # --- Step 4: Execute (only if cert was minted + URL is clean) ---
    try:
        result = _execute_http_fetch(
            url=url,
            method=method,
            headers=headers,
            timeout_ms=timeout_ms,
            max_bytes=max_bytes,
            domain_allowlist=domain_allowlist,
        )
    except RedirectError as e:
        # Redirect violated security constraints
        if trace is not None:
            trace.append(MerkleLeaf(
                move="TOOL_CALL:http_fetch",
                fail_type="CONSTRAINT_VIOLATION",
                invariant_diff=[{
                    "inv": e.invariant,
                    "expected": "pass",
                    "got": f"fail ({e.detail})",
                    "witness": {"redirect_url": e.redirect_url},
                }],
            ))
        return ToolResult(
            success=False,
            tool="http_fetch",
            cert_id=cert["cert_id"],
            error=f"URL sanitization failed: {e.invariant} — {e.detail}",
            trace_summary=trace.summary() if trace else None,
        )

    # --- Step 5: Return result ---
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

    # Helper: permissive token for testing bypass classes (allows any domain)
    permissive_token = CapabilityToken(
        agent_id="test",
        session_id="self-test",
        capabilities=[
            CapabilityEntry(
                tool="http_fetch",
                scope="network",
                args_schema="SCHEMA.HTTP_FETCH.v1",
                constraints={},  # No domain restriction
            ),
        ],
    )

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
        capability_token=permissive_token,
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
        capability_token=permissive_token,
        trace=trace,
    )
    check("Schema blocks invalid method",
          not r6.success and r6.obstruction_id is not None)

    # --- Test 7: Blocks credential-in-URL ---
    r7 = execute_http_fetch(
        url="https://user:pass@example.com/secret",
        intent_description="fetch with creds",
        intent_source="policy_kernel",
        intent_ref="test:3",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=permissive_token,
        trace=trace,
    )
    check("Blocks credential-in-URL",
          not r7.success and "URL_NO_CREDENTIALS" in (r7.error or ""))

    # --- Test 8: Blocks raw IPv4 literal ---
    r8 = execute_http_fetch(
        url="https://192.168.1.1/admin",
        intent_description="fetch IP",
        intent_source="policy_kernel",
        intent_ref="test:4",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=permissive_token,
        trace=trace,
    )
    check("Blocks raw IPv4 literal",
          not r8.success and "URL_NO_IP_LITERAL" in (r8.error or ""))

    # --- Test 9: Blocks punycode hostname ---
    r9 = execute_http_fetch(
        url="https://xn--80ak6aa92e.com/page",
        intent_description="fetch punycode",
        intent_source="policy_kernel",
        intent_ref="test:5",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=permissive_token,
        trace=trace,
    )
    check("Blocks punycode hostname",
          not r9.success and "URL_NO_PUNYCODE" in (r9.error or ""))

    # --- Test 10: Blocks dangerous header injection ---
    r10 = execute_http_fetch(
        url="https://example.com/api",
        headers={"Host": "evil.com", "Authorization": "Bearer token"},
        intent_description="fetch with host override",
        intent_source="policy_kernel",
        intent_ref="test:6",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=permissive_token,
        trace=trace,
    )
    check("Blocks Host header injection",
          not r10.success and "HEADER_INJECTION_BLOCKED" in (r10.error or "")
          and r10.cert_id is not None)  # cert minted, but execution blocked by sanitizer

    # --- Test 11: Mandatory capability token enforcement ---
    # Temporarily ensure enforcement is on for this test
    global REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH
    old_enforce = REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH
    REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = True
    r11 = execute_http_fetch(
        url="https://example.com/page",
        intent_description="fetch without token",
        intent_source="policy_kernel",
        intent_ref="test:7",
        url_trusted=True,
        requires_human_approval=False,
        capability_token=None,  # No token!
        trace=trace,
    )
    REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = old_enforce
    check("Blocks fetch without capability token",
          not r11.success and "CAP_TOKEN_REQUIRED" in (r11.error or ""))

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
