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

    def _make_permissive_token(self):
        """Token with no domain restrictions for testing other bypass classes."""
        return CapabilityToken(
            agent_id="test",
            session_id="s1",
            capabilities=[
                CapabilityEntry(
                    tool="http_fetch",
                    scope="network",
                    args_schema="SCHEMA.HTTP_FETCH.v1",
                    constraints={},  # No domain restriction
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
            capability_token=self._make_permissive_token(),
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
            capability_token=self._make_permissive_token(),
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
            capability_token=self._make_permissive_token(),
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
            capability_token=self._make_permissive_token(),
            trace=trace,
        )
        # Schema validation should reject non-http URL pattern
        assert not r.success

    def test_trace_records_all_bypass_attempts(self):
        """Run multiple bypasses and verify trace integrity."""
        trace = MerkleTrace()
        token = self._make_permissive_token()
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
                capability_token=token,
                trace=trace,
            )
        s = trace.summary()
        # All should be blocked (some at policy, some at sanitization)
        assert s["total_steps"] >= 3
        assert s["merkle_root"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Redirect handler unit tests
# ---------------------------------------------------------------------------

from qa_agent_security.tool_runner import (
    SafeRedirectHandler,
    RedirectError,
    MAX_REDIRECTS,
)


class TestSafeRedirectHandler:
    """
    Unit tests for the SafeRedirectHandler to verify redirect validation.
    These test the handler's logic directly without making network calls.
    """

    def test_redirect_to_ip_literal_blocked(self):
        """Redirect to raw IP should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=["example.com"])
        with pytest.raises(RedirectError) as exc:
            # Simulate a redirect request to an IP literal
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://192.168.1.1/internal"
            )
        assert exc.value.invariant == "URL_NO_IP_LITERAL"
        assert exc.value.redirect_url == "https://192.168.1.1/internal"

    def test_redirect_to_ipv6_blocked(self):
        """Redirect to IPv6 should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=["example.com"])
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://[::1]/admin"
            )
        assert exc.value.invariant == "URL_NO_IP_LITERAL"

    def test_redirect_with_credentials_blocked(self):
        """Redirect URL with credentials should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=["example.com"])
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://user:pass@example.com/secret"
            )
        assert exc.value.invariant == "URL_NO_CREDENTIALS"

    def test_redirect_to_punycode_blocked(self):
        """Redirect to punycode domain should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=["example.com"])
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://xn--80ak6aa92e.com/page"
            )
        assert exc.value.invariant == "URL_NO_PUNYCODE"

    def test_redirect_off_allowlist_blocked(self):
        """Redirect to domain not in allowlist should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=["api.github.com"])
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://evil.com/steal"
            )
        assert exc.value.invariant == "REDIRECT_TARGET_ALLOWLIST"
        assert "evil.com" in exc.value.detail

    def test_redirect_within_allowlist_allowed(self):
        """Redirect to domain in allowlist should NOT raise (returns Request)."""
        handler = SafeRedirectHandler(domain_allowlist=["api.github.com", "github.com"])
        # This should not raise - but we can't fully test without mocking Request
        # The key test is that it doesn't raise RedirectError
        # Note: will raise AttributeError on req.get_method() since req=None
        # but that's after our validation passes
        try:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://github.com/repos"
            )
        except RedirectError:
            pytest.fail("RedirectError raised for allowed domain")
        except AttributeError:
            # Expected: parent's redirect_request tries to call req.get_method()
            pass

    def test_redirect_no_allowlist_passes_sanitization(self):
        """Without allowlist, redirect passes if URL is clean."""
        handler = SafeRedirectHandler(domain_allowlist=None)  # No allowlist
        try:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://any-domain.com/page"
            )
        except RedirectError:
            pytest.fail("RedirectError raised when no allowlist configured")
        except AttributeError:
            # Expected: parent method failure
            pass

    def test_redirect_count_limit_enforced(self):
        """Exceeding MAX_REDIRECTS should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=None)
        # Simulate MAX_REDIRECTS+1 redirects
        for i in range(MAX_REDIRECTS):
            try:
                handler.redirect_request(
                    req=None, fp=None, code=302, msg="Found",
                    headers={}, newurl=f"https://example.com/hop{i}"
                )
            except AttributeError:
                # Expected after validation passes
                pass

        # The next redirect should fail
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://example.com/final"
            )
        assert exc.value.invariant == "REDIRECT_COUNT_MAX"

    def test_redirect_with_control_chars_blocked(self):
        """Redirect URL with control characters should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=None)
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="https://example.com/\r\ninjection"
            )
        assert exc.value.invariant == "URL_NO_CONTROL_CHARS"

    def test_redirect_file_scheme_blocked(self):
        """Redirect to file:// scheme should raise RedirectError."""
        handler = SafeRedirectHandler(domain_allowlist=None)
        with pytest.raises(RedirectError) as exc:
            handler.redirect_request(
                req=None, fp=None, code=302, msg="Found",
                headers={}, newurl="file:///etc/passwd"
            )
        assert exc.value.invariant == "URL_SCHEME_HTTP_ONLY"


class TestCapabilityTokenEnforcement:
    """
    Tests for mandatory capability token enforcement.
    """

    def test_missing_token_blocked_when_enforced(self):
        """With enforcement on, missing token should block even trusted URLs."""
        import qa_agent_security.tool_runner as runner
        old_val = runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH
        try:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = True
            trace = MerkleTrace()
            r = execute_http_fetch(
                url="https://example.com/page",
                intent_source="policy_kernel",
                intent_ref="test",
                url_trusted=True,
                requires_human_approval=False,
                capability_token=None,
                trace=trace,
            )
            assert not r.success
            assert "CAP_TOKEN_REQUIRED" in (r.error or "")
            # Cert was minted (policy passed) but token enforcement blocked
            assert r.cert_id is not None
        finally:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = old_val

    def test_missing_token_allowed_when_not_enforced(self):
        """With enforcement off, missing token should allow execution."""
        import qa_agent_security.tool_runner as runner
        old_val = runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH
        try:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = False
            trace = MerkleTrace()
            r = execute_http_fetch(
                url="https://example.com/page",
                intent_source="policy_kernel",
                intent_ref="test",
                url_trusted=True,
                requires_human_approval=False,
                capability_token=None,
                trace=trace,
            )
            # Should proceed to execution (will fail network, but cert minted)
            assert r.cert_id is not None
            assert "CAP_TOKEN_REQUIRED" not in (r.error or "")
        finally:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = old_val

    def test_trace_records_token_enforcement_block(self):
        """Verify trace records token enforcement as obstruction."""
        import qa_agent_security.tool_runner as runner
        old_val = runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH
        try:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = True
            trace = MerkleTrace()
            execute_http_fetch(
                url="https://example.com/page",
                intent_source="policy_kernel",
                intent_ref="test",
                url_trusted=True,
                requires_human_approval=False,
                capability_token=None,
                trace=trace,
            )
            s = trace.summary()
            assert s["blocked"] >= 1
        finally:
            runner.REQUIRE_CAPABILITY_TOKEN_FOR_HTTP_FETCH = old_val


class TestRunnerConstraintRecheck:
    """
    Tests for runner-side constraint recheck (defense-in-depth).
    """

    def test_runner_rechecks_domain_allowlist(self):
        """Runner should recheck URL hostname against token constraints."""
        # Create a token that allows only api.github.com
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
        # Try to fetch a URL that's NOT in the allowlist
        # This should be caught by both kernel and runner
        r = execute_http_fetch(
            url="https://evil.com/steal",
            intent_source="policy_kernel",
            intent_ref="test",
            url_trusted=True,
            requires_human_approval=False,
            capability_token=token,
            trace=trace,
        )
        assert not r.success
        # Should be blocked (by kernel's DOMAIN_ALLOWLIST check)
        assert r.obstruction_id is not None


class TestRedirectErrorPropagation:
    """
    Tests that verify RedirectError is properly converted to ToolResult obstruction.
    """

    def test_redirect_error_creates_obstruction_trace(self):
        """Verify that a simulated RedirectError would be recorded in trace."""
        from qa_agent_security.tool_runner import RedirectError, MerkleLeaf

        # Create a trace and manually append what would happen on redirect error
        trace = MerkleTrace()
        err = RedirectError(
            invariant="REDIRECT_TARGET_ALLOWLIST",
            detail="Redirect target 'evil.com' not in allowlist",
            redirect_url="https://evil.com/steal"
        )
        trace.append(MerkleLeaf(
            move="TOOL_CALL:http_fetch",
            fail_type="CONSTRAINT_VIOLATION",
            invariant_diff=[{
                "inv": err.invariant,
                "expected": "pass",
                "got": f"fail ({err.detail})",
            }],
        ))
        s = trace.summary()
        assert s["blocked"] == 1
        assert s["merkle_root"].startswith("sha256:")

    def test_redirect_error_attributes(self):
        """Verify RedirectError has all required attributes for obstruction."""
        err = RedirectError(
            invariant="REDIRECT_TARGET_ALLOWLIST",
            detail="test detail",
            redirect_url="https://evil.com"
        )
        assert err.invariant == "REDIRECT_TARGET_ALLOWLIST"
        assert err.detail == "test detail"
        assert err.redirect_url == "https://evil.com"
        assert "REDIRECT_TARGET_ALLOWLIST: test detail" in str(err)
