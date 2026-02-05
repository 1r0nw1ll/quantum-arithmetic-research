# QA Guardrail

**QA Guardrail** is a production-grade security gate for autonomous agents.
It enforces deterministic, cryptographically-verifiable controls over tool execution.

Designed to sit between:

> Agent Planner → **Guardrail** → Tool Executor

and decide, in a reproducible way, whether a planned action is allowed.

## Core Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| Deny-by-default | Unknown generators rejected |
| Instruction/Content separation | IC cert + content domain enforcement |
| Crypto-bound verification | QA_IC_VERIFICATION_RECEIPT.v1 |
| Scanner pinning | scanner_id + scanner_version validation |
| Generator binding | Mandatory hash when requested |
| Tamper-evident audit | Hash-chained entries + Merkle root |
| Deterministic validation | Same input → same output |

Every decision can be independently verified.

## Components

| Component | Purpose |
|-----------|---------|
| `guard()` | Core decision engine |
| `threat_scanner` | Content + policy verification |
| `AuditLogger` | Hash-chained audit trail |
| `QA_IC_VERIFICATION_RECEIPT.v1` | Crypto receipt schema |
| Golden Fixtures | Regression + security tests |
| E2E Tests | Subprocess + OpenClaw simulation |

## Quickstart

### Install (via submodule)

```bash
git submodule add https://github.com/1r0nw1ll/quantum-arithmetic-research.git vendor/qa
cd vendor/qa
git checkout v0.1.0-guardrail
```

### Basic Usage

```python
from qa_alphageometry_ptolemy.qa_guardrail import (
    guard,
    GuardrailContext,
    AuditLogger,
    create_verification_receipt,
)

# Create context
ctx = GuardrailContext(
    active_generators={"sigma", "mu", "lambda", "nu"},
    content="user input here",
    policy={"scan_content": True, "deny_on_threats": True},
)

# Guard a move
result = guard("sigma(1)", ctx)

if result.ok:
    # Execute the tool
    pass
else:
    # Deny with reason
    print(result.fail_record)
```

### With Audit Logging

```python
logger = AuditLogger(session_id="run-001")

result = guard("sigma(1)", ctx)
entry = logger.log("sigma(1)", ctx, result)

# After session
print(logger.to_jsonl())       # JSONL export
print(logger.merkle_root())    # Tamper-evident root
print(logger.summary())        # {total, allow, deny}
```

### With Verification Receipts

```python
from qa_alphageometry_ptolemy.qa_guardrail import create_verification_receipt

content = "Please help with math"
policy = {"require_verification_receipt": True}

# Generate receipt (binds content + policy + scanner state)
receipt = create_verification_receipt(content, policy)

ctx = GuardrailContext(
    verification_receipt=receipt,
    content=content,
    policy=policy,
)

result = guard("sigma(1)", ctx)  # ALLOW if receipt valid + VERIFIED_SAFE
```

## Policy Configuration

Policies are passed via `GuardrailContext.policy`:

| Flag | Meaning |
|------|---------|
| `scan_content` | Enable threat pattern scanning |
| `deny_on_threats` | Hard deny on threat detection |
| `require_verified_ic_cert` | Require verified IC cert (legacy) |
| `require_verification_receipt` | Require crypto-bound receipt |
| `receipt_ttl_seconds` | Maximum receipt age |
| `bind_receipt_to_generators` | Bind receipt to generator set |

## Running Tests

```bash
# Self-tests (52 tests)
python -m qa_alphageometry_ptolemy.qa_guardrail

# Golden fixtures (18 fixtures)
python -m qa_alphageometry_ptolemy.qa_guardrail --fixtures

# E2E subprocess tests (12 scenarios)
python -m qa_alphageometry_ptolemy.qa_guardrail.e2e_test

# Threat scanner tests (31 tests)
python qa_alphageometry_ptolemy/qa_guardrail/threat_scanner.py
```

**Total: 113 tests**

## Security Model

QA Guardrail prevents:

- **Receipt spoofing** - Receipts bound to content/policy/patterns/scanner
- **Content substitution** - content_sha256 must match
- **Scanner impersonation** - scanner_id + scanner_version verified
- **Generator bypass** - Mandatory generator_set_sha256 when requested
- **Policy drift** - policy_sha256 must match
- **Audit tampering** - Hash chain + Merkle root

## Integration

See [OPENCLAW_INTEGRATION.md](../docs/OPENCLAW_INTEGRATION.md) for agent framework wiring.

## Version

Current release: `v0.1.0-guardrail`

## License

See repository LICENSE.
