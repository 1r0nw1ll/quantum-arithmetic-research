# QA Agent Security Kernel

Certificate-gated tool execution with taint tracking and deterministic obstruction generation.

**Design principle:** agent = generators; security = legality of moves.

## What it does

Every tool call must pass through the **policy kernel**, which:

1. Verifies provenance tags on all arguments (TAINTED vs TRUSTED)
2. Validates args against strict JSON schemas (additionalProperties=false)
3. Checks capability token constraints (denylists, allowlists, scope, TTL)
4. Enforces invariants (no web->exec, critical fields must be trusted)
5. Mints a `TOOL_CALL_CERT.v1` on success, or emits a `PROMPT_INJECTION_OBSTRUCTION.v1` on failure
6. Appends every move to a Merkle trace (1:1 with QARM log rows)

## Prompt injection as reachability obstruction

In QA terms, prompt injection is an attempt to add unauthorized generators via an untrusted edge. The kernel classifies failures into a canonical obstruction family:

| Fail Type | Meaning |
|-----------|---------|
| `UNTRUSTED_INSTRUCTION` | Action-critical field is TAINTED, no approval |
| `CAPABILITY_ESCALATION_ATTEMPT` | Web/email/file source trying to reach exec |
| `SCHEMA_BREAKOUT` | Args don't match strict JSON schema |
| `CONFUSED_DEPUTY` | High-privilege tool invoked on behalf of low-privilege input |
| `PROVENANCE_LOSS` | Cannot justify where a critical field came from |
| `CONSTRAINT_VIOLATION` | Capability token constraint (denylist/allowlist) failed |
| `CAPABILITY_TOKEN_EXPIRED` | Token TTL exceeded |
| `CAPABILITY_NOT_FOUND` | No capability entry for requested tool+scope |
| `TAINT_UPGRADE_VIOLATION` | Transform attempted TAINTED->TRUSTED upgrade |

## URL sanitization invariants (HTTP_FETCH)

The tool runner enforces additional URL-level invariants on both initial requests and redirect hops:

| Invariant | Bypass Class Blocked |
|-----------|---------------------|
| `URL_SCHEME_HTTP_ONLY` | file://, ftp://, etc. |
| `URL_NO_CREDENTIALS` | user:pass@host |
| `URL_NO_IP_LITERAL` | Raw IPv4/IPv6 addresses |
| `URL_NO_PUNYCODE` | IDN homograph attacks |
| `URL_NO_CONTROL_CHARS` | Null bytes, CRLF injection |
| `URL_HOSTNAME_REQUIRED` | Empty hostname |
| `HEADER_INJECTION_BLOCKED` | Host, Transfer-Encoding override |
| `REDIRECT_TARGET_ALLOWLIST` | Redirect to off-allowlist domain |
| `REDIRECT_COUNT_MAX` | Redirect chain > 5 hops |
| `CAP_TOKEN_REQUIRED` | Missing capability token |
| `CAP_TOKEN_CONSTRAINT_MISMATCH` | Runner-side allowlist recheck failed |

## Mandatory capability tokens

**HTTP_FETCH requires a capability token by default.** This makes capability tokens mandatory rather than advisory.

To disable in local experiments:
```bash
export QA_REQUIRE_CAP_TOKEN_HTTP_FETCH=false
```

## Certificate types

| Certificate | Purpose |
|-------------|---------|
| `TAINT_FLOW_CERT.v1` | Tracks taint through transforms; blocks TAINTED->TRUSTED upgrade |
| `TOOL_CALL_CERT.v1` | Authorizes a single tool invocation |
| `PROMPT_INJECTION_OBSTRUCTION.v1` | Deterministic failure artifact |
| `CAPABILITY_TOKEN.v1` | Scoped, time-limited authorization |
| `MERKLE_TRACE.v1` | Per-move audit leaf: `{move, fail_type, invariant_diff}` |

## Running

```bash
# Policy kernel self-test (14 checks, stdlib-only, no pip install)
python qa_agent_security/qa_agent_security.py

# Tool runner self-test (11 checks, no network needed)
python -m qa_agent_security.tool_runner

# Full pytest suite (100 tests)
python -m pytest qa_agent_security/tests/ -v \
  --override-ini="testpaths=qa_agent_security/tests" \
  --override-ini="python_files=test_*.py"
```

## File index

```
qa_agent_security/
  __init__.py                       # Re-exports all public API
  qa_agent_security.py              # Policy kernel + self-test (14 checks)
  schemas.py                        # Embedded tool schemas + validate_args()
  tool_runner.py                    # Certificate-gated tool execution + self-test (11 checks)
  README.md                         # This file
  schemas/
    TOOL_CALL_CERT.v1.schema.json
    PROMPT_INJECTION_OBSTRUCTION.v1.schema.json
    TAINT_FLOW_CERT.v1.schema.json
    CAPABILITY_TOKEN.v1.schema.json
    MERKLE_TRACE.v1.schema.json
  tests/
    test_agent_security.py          # 27 pytest tests (kernel + trace + tokens)
    test_schemas.py                 # 17 pytest tests (schema registry + validation)
    test_security_regression.py     # 46 pytest tests (bypass regression)
    test_tool_runner.py             # 10 pytest tests (end-to-end pipeline)
```

## Integration with Moltbot-style agents

Route everything through the kernel:

1. Ingest message/web/email -> TAINTED provenance
2. Model proposes a tool call as a draft
3. Kernel validates -> mints `TOOL_CALL_CERT.v1` or emits `PROMPT_INJECTION_OBSTRUCTION.v1`
4. Tool executes only if cert exists **and** capability token is present
5. Every move logged to Merkle trace

This turns prompt injection from "security whack-a-mole" into "most attempts cannot even become legal moves."
