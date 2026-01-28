"""
test_security_regression.py — Security regression tests for HTTP_FETCH bypass classes.

Tests classic prompt-injection and URL-manipulation bypass attempts:
  - Credential-in-URL (user:pass@host)
  - Raw IP literals (IPv4, IPv6)
  - Punycode / IDN homograph attacks
  - Mixed-case domain allowlist bypass
  - Dangerous header injection (Host, Transfer-Encoding)
  - File scheme injection
  - Control characters / null bytes in URL

Each blocked attempt must produce a deterministic obstruction with the
correct fail_type and invariant_diff.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qa_agent_security import MerkleTrace, CapabilityToken, CapabilityEntry
from qa_agent_security.tool_runner import (
    execute_http_fetch,
    sanitize_url,
    sanitize_headers,
    URLSanitizationError,
)


# ---------------------------------------------------------------------------
# URL sanitization unit tests
# ---------------------------------------------------------------------------

class TestSanitizeURL:
    def test_valid_https(self):
        scheme, host = sanitize_url("https://api.github.com/repos")
        assert scheme == "https"
        assert host == "api.github.com"

    def test_valid_http(self):
        scheme, host = sanitize_url("http://example.com")
        assert scheme == "http"
        assert host == "example.com"

    def test_rejects_file_scheme(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("file:///etc/passwd")
        assert exc.value.invariant == "URL_SCHEME_HTTP_ONLY"

    def test_rejects_ftp_scheme(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("ftp://files.example.com/data")
        assert exc.value.invariant == "URL_SCHEME_HTTP_ONLY"

    def test_rejects_credentials_in_url(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://admin:secret@example.com/")
        assert exc.value.invariant == "URL_NO_CREDENTIALS"

    def test_rejects_username_only(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://admin@example.com/")
        assert exc.value.invariant == "URL_NO_CREDENTIALS"

    def test_rejects_ipv4_literal(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://192.168.1.1/admin")
        assert exc.value.invariant == "URL_NO_IP_LITERAL"

    def test_rejects_ipv4_loopback(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://127.0.0.1:8080/")
        assert exc.value.invariant == "URL_NO_IP_LITERAL"

    def test_rejects_ipv6_literal(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://[::1]/admin")
        assert exc.value.invariant == "URL_NO_IP_LITERAL"

    def test_rejects_punycode(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://xn--80ak6aa92e.com/page")
        assert exc.value.invariant == "URL_NO_PUNYCODE"

    def test_rejects_punycode_subdomain(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://xn--nxasmq6b.example.com/page")
        assert exc.value.invariant == "URL_NO_PUNYCODE"

    def test_rejects_null_bytes(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://example.com/\x00evil")
        assert exc.value.invariant == "URL_NO_CONTROL_CHARS"

    def test_rejects_control_characters(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https://example.com/\r\nevil")
        assert exc.value.invariant == "URL_NO_CONTROL_CHARS"

    def test_normalizes_to_lowercase(self):
        scheme, host = sanitize_url("https://API.GITHUB.COM/repos")
        assert host == "api.github.com"

    def test_rejects_empty_hostname(self):
        with pytest.raises(URLSanitizationError) as exc:
            sanitize_url("https:///path")
        assert exc.value.invariant == "URL_HOSTNAME_REQUIRED"


# ---------------------------------------------------------------------------
# Header sanitization unit tests
# ---------------------------------------------------------------------------

class TestSanitizeHeaders:
    def test_safe_headers_pass(self):
        diffs = sanitize_headers({
            "Authorization": "Bearer token",
            "Accept": "application/json",
            "User-Agent": "QA-Agent/1.0",
        })
        assert diffs == []

    def test_blocks_host_header(self):
        diffs = sanitize_headers({"Host": "evil.com"})
        assert len(diffs) == 1
        assert diffs[0]["inv"] == "HEADER_INJECTION_BLOCKED"

    def test_blocks_transfer_encoding(self):
        diffs = sanitize_headers({"Transfer-Encoding": "chunked"})
        assert len(diffs) == 1
        assert diffs[0]["inv"] == "HEADER_INJECTION_BLOCKED"

    def test_blocks_connection_header(self):
        diffs = sanitize_headers({"Connection": "keep-alive"})
        assert len(diffs) == 1

    def test_blocks_multiple_dangerous(self):
        diffs = sanitize_headers({
            "Host": "evil.com",
            "Transfer-Encoding": "chunked",
            "Accept": "text/html",
        })
        assert len(diffs) == 2

    def test_case_insensitive(self):
        diffs = sanitize_headers({"HOST": "evil.com"})
        assert len(diffs) == 1


# ---------------------------------------------------------------------------
# End-to-end runner regression tests
# ---------------------------------------------------------------------------

class TestHTTPFetchBypassRegression:
    """
    Each test simulates a classic bypass and verifies the runner blocks it
    with the correct obstruction/error.
    """

    def _make_github_token(self):
        return CapabilityToken(
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

    def test_credential_url_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://user:pass@api.github.com/repos",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=self._make_github_token(),
            trace=trace,
        )
        assert not r.success
        assert "URL_NO_CREDENTIALS" in (r.error or "")
        # Cert was minted (policy passed) but URL sanitization caught it
        assert r.cert_id is not None

    def test_ipv4_literal_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://10.0.0.1/internal",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert "URL_NO_IP_LITERAL" in (r.error or "")

    def test_ipv6_loopback_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://[::1]/admin",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert "URL_NO_IP_LITERAL" in (r.error or "")

    def test_punycode_homograph_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://xn--80ak6aa92e.com/page",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            trace=trace,
        )
        assert not r.success
        assert "URL_NO_PUNYCODE" in (r.error or "")

    def test_mixed_case_domain_still_matches_allowlist(self):
        """Mixed-case should normalize and match the allowlist."""
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://API.GITHUB.COM/repos",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=self._make_github_token(),
            trace=trace,
        )
        # Should pass: domain normalizes to api.github.com
        assert r.cert_id is not None

    def test_subdomain_spoofing_blocked_by_allowlist(self):
        """api.github.com.evil.com should NOT match api.github.com."""
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://api.github.com.evil.com/repos",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=self._make_github_token(),
            trace=trace,
        )
        assert not r.success
        # The domain allowlist should reject this
        assert r.obstruction_id is not None

    def test_host_header_injection_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="https://api.github.com/repos",
            headers={"Host": "evil.com", "Accept": "application/json"},
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=self._make_github_token(),
            trace=trace,
        )
        assert not r.success
        assert "HEADER_INJECTION_BLOCKED" in (r.error or "")

    def test_file_scheme_blocked(self):
        trace = MerkleTrace()
        r = execute_http_fetch(
            url="file:///etc/passwd",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            trace=trace,
        )
        # Schema validation should reject non-http URL pattern
        assert not r.success

    def test_trace_records_all_bypass_attempts(self):
        """Run multiple bypasses and verify trace integrity."""
        trace = MerkleTrace()
        urls = [
            "https://user:pass@example.com/",
            "https://192.168.1.1/",
            "https://xn--80ak6aa92e.com/",
        ]
        for url in urls:
            execute_http_fetch(
                url=url,
                intent_source="policy_kernel",
                intent_ref="test",
                url_trusted=True,
                requires_human_approval=False,
                trace=trace,
            )
        s = trace.summary()
        # All should be blocked (some at policy, some at sanitization)
        assert s["total_steps"] >= 3
        assert s["merkle_root"].startswith("sha256:")
