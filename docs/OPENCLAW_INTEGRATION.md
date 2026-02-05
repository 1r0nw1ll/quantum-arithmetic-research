# OpenClaw Integration Guide

This document describes how to integrate QA Guardrail into OpenClaw
as a secure `before_tool_call` gate.

## 1. Add QA Guardrail as a Submodule

From OpenClaw root:

```bash
git submodule add https://github.com/1r0nw1ll/quantum-arithmetic-research.git vendor/qa
cd vendor/qa
git checkout v0.1.0-guardrail
```

Directory structure:

```
openclaw/
  vendor/
    qa/
      qa_alphageometry_ptolemy/
        qa_guardrail/
          __init__.py
          qa_guardrail.py
          threat_scanner.py
          ...
```

## 2. Python Integration

### Recommended Wrapper

Create `openclaw/security/qa_guardrail_gate.py`:

```python
import sys
sys.path.insert(0, "vendor/qa")

from qa_alphageometry_ptolemy.qa_guardrail import (
    guard,
    GuardrailContext,
    AuditLogger,
    create_verification_receipt,
)

# Session-level logger
_logger = None

def init_session(session_id: str):
    global _logger
    _logger = AuditLogger(session_id=session_id)

def before_tool_call(planned_move: str, context: dict) -> dict:
    """
    Gate a tool call through QA Guardrail.

    Args:
        planned_move: The generator invocation (e.g., "tool.bash({...})")
        context: Dict with generators, content, policy, capabilities

    Returns:
        {"allow": bool, "result": GuardrailResult, "entry": audit_entry}
    """
    # Build verification receipt if content provided
    receipt = None
    if context.get("content") and context.get("policy", {}).get("require_verification_receipt"):
        receipt = create_verification_receipt(
            context["content"],
            context["policy"],
            generators=set(context.get("generators", [])) if context.get("bind_generators") else None,
        )

    ctx = GuardrailContext(
        active_generators=set(context.get("generators", ["sigma", "mu", "lambda", "nu"])),
        content=context.get("content"),
        verification_receipt=receipt or context.get("verification_receipt"),
        instruction_content_cert=context.get("ic_cert"),
        policy=context.get("policy", {}),
        capabilities=set(context.get("capabilities", [])),
    )

    result = guard(planned_move, ctx)

    entry = None
    if _logger:
        entry = _logger.log(planned_move, ctx, result)

    return {
        "allow": result.ok,
        "result": result,
        "entry": entry,
    }

def get_audit_log():
    """Get JSONL audit log for current session."""
    return _logger.to_jsonl() if _logger else ""

def get_merkle_root():
    """Get Merkle root for current session."""
    return _logger.merkle_root() if _logger else None

def get_summary():
    """Get session summary."""
    return _logger.summary() if _logger else {}
```

## 3. Hook into OpenClaw Pipeline

At the planner/executor boundary:

```python
from security.qa_guardrail_gate import before_tool_call, init_session

# At session start
init_session(f"openclaw-{session_id}")

# Before each tool call
def execute_tool(move, context):
    gate_result = before_tool_call(move, context)

    if not gate_result["allow"]:
        result = gate_result["result"]
        raise ToolDenied(
            move=move,
            fail_type=result.fail_record["fail_type"],
            reason=result.fail_record.get("detail"),
            checks=result.checks,
        )

    # Proceed with tool execution
    return actual_execute(move)
```

## 4. TypeScript Integration (CLI Mode)

For TypeScript, use subprocess invocation:

```typescript
// openclaw/extensions/qa-guardrail/index.ts
import { spawn } from 'child_process';

interface GuardrailRequest {
  planned_move: string;
  context: {
    active_generators: string[];
    content?: string;
    policy?: Record<string, any>;
    capabilities?: string[];
  };
}

async function runGuardrail(request: GuardrailRequest): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn('python', [
      '-m', 'qa_alphageometry_ptolemy.qa_guardrail.qa_guardrail',
      '--guard'
    ], { cwd: 'vendor/qa' });

    let stdout = '';
    proc.stdout.on('data', (data) => stdout += data);
    proc.stdin.write(JSON.stringify(request));
    proc.stdin.end();

    proc.on('close', (code) => {
      try {
        resolve(JSON.parse(stdout));
      } catch (e) {
        reject(e);
      }
    });
  });
}

export const hooks = {
  async before_tool_call(ctx: ToolCallContext): Promise<{allow: boolean, reason?: string}> {
    const result = await runGuardrail({
      planned_move: `tool.${ctx.toolName}(${JSON.stringify(ctx.params)})`,
      context: {
        active_generators: ctx.activeGenerators,
        content: ctx.userInput,
        policy: ctx.securityPolicy,
        capabilities: ctx.grantedCapabilities,
      }
    });

    if (result.ok && result.result === "ALLOW") {
      return { allow: true };
    }
    return {
      allow: false,
      reason: result.fail_record?.detail
    };
  }
};
```

## 5. Context Mapping

| OpenClaw Field | Guardrail Field |
|----------------|-----------------|
| generators | active_generators |
| user_input | content |
| ic_cert | instruction_content_cert |
| security_policy | policy |
| granted_caps | capabilities |
| verification_receipt | verification_receipt |

## 6. Policy Configuration

### Recommended Production Policy

```python
policy = {
    "scan_content": True,
    "deny_on_threats": True,
    "require_verification_receipt": True,
    "receipt_ttl_seconds": 300,  # 5 minute freshness
}
```

### Development Policy (Looser)

```python
policy = {
    "scan_content": True,
    "deny_on_threats": False,  # Warn only
}
```

## 7. Audit Log Handling

After each session:

```python
from security.qa_guardrail_gate import get_audit_log, get_merkle_root, get_summary

# Save audit trail
with open(f"audit/{session_id}.jsonl", "w") as f:
    f.write(get_audit_log())

# Record merkle root (for tamper detection)
summary = get_summary()
summary["merkle_root"] = get_merkle_root()
save_session_metadata(session_id, summary)
```

## 8. Version Pinning

**Always pin to a specific tag:**

```bash
cd vendor/qa
git checkout v0.1.0-guardrail
```

**Upgrade process:**

1. Review changelog
2. Run tests against new version
3. Update pin
4. Commit submodule update

**Never track `main` in production.**

## 9. Testing Integration

Add to CI:

```yaml
# .github/workflows/test.yml
- name: Test QA Guardrail
  run: |
    cd vendor/qa
    python -m qa_alphageometry_ptolemy.qa_guardrail --fixtures
    python -m qa_alphageometry_ptolemy.qa_guardrail.e2e_test
```

## 10. Operational Guidelines

- **Do not** vendor-copy files into OpenClaw
- **Do not** modify guardrail internals
- **Treat** as security boundary (like a firewall)
- **Log** all DENY events to persistent storage
- **Monitor** merkle roots for anomalies
- **Alert** on unexpected fail_types

## Support

Report issues: https://github.com/1r0nw1ll/quantum-arithmetic-research/issues

Canonical source: https://github.com/1r0nw1ll/quantum-arithmetic-research
