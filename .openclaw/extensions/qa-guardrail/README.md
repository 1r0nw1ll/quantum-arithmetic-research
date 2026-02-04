# QA Guardrail OpenClaw Plugin

**Agents propose, validators decide** - at the tool boundary where it matters.

## Overview

This plugin implements the QA Guardrail as an OpenClaw `before_tool_call` gate. Every tool call is converted to a QA "planned_move" and validated against the guardrail before execution is permitted.

## Architecture

```
User Request
     │
     ▼
┌─────────────────────┐
│   OpenClaw Agent    │
│  (Model + Planner)  │
└─────────────────────┘
     │
     │ Tool Call: { toolName, params }
     ▼
┌─────────────────────┐
│  QA Guardrail Gate  │◄── before_tool_call hook
│                     │
│  planned_move =     │
│  tool.{name}({...}) │
└─────────────────────┘
     │
     │ GUARDRAIL_REQUEST.v1 (stdin JSON)
     ▼
┌─────────────────────┐
│  qa_guardrail.py    │
│  --guard            │
└─────────────────────┘
     │
     │ GUARDRAIL_RESULT.v1 (stdout JSON)
     ▼
┌─────────────────────┐
│  ALLOW → Execute    │
│  DENY  → Block      │
└─────────────────────┘
```

## Installation

1. Copy this directory to `.openclaw/extensions/qa-guardrail/`
2. Copy `config.example.json` to `.openclaw/config.json` and customize
3. Ensure Python 3.10+ is available

## Configuration

```json
{
  "extensions": {
    "qa-guardrail": {
      "enabled": true,
      "pythonPath": "python",
      "guardrailPath": "qa_alphageometry_ptolemy/qa_guardrail/qa_guardrail.py",
      "policy": {
        "deny": ["tool.exec", "tool.gmail.*"],
        "require_verified_ic_cert": false
      },
      "capabilities": ["READ", "WRITE"],
      "activeGenerators": ["sigma", "mu", "lambda", "nu"],
      "logDenials": true
    }
  }
}
```

### Policy Options

- `deny`: Array of tool patterns to always deny (supports `*` suffix)
- `allow`: If set, only these tools are allowed (whitelist mode)
- `require_verified_ic_cert`: Require verified instruction/content separation certificate
- `required_capability`: Capability token required for all calls

### Capability Mapping

Tools are automatically mapped to capabilities:
- `bash`, `exec`, `shell` → `EXEC`
- `write`, `edit`, `file_write` → `WRITE`
- `gmail`, `browser` → `NET_CREDENTIAL`
- `web_search`, `web_fetch` → `NET`

## Fail Types

When a tool call is denied, the `fail_record` includes one of:

| Fail Type | Description |
|-----------|-------------|
| `UNAUTHORIZED_GENERATOR` | Tool not in active generators |
| `POLICY_CONSTRAINT_VIOLATION` | Tool in deny list or not in allow list |
| `MISSING_CAPABILITY` | Required capability not granted |
| `PROMOTION_FORBIDDEN` | Content tried to become instruction |
| `INSTRUCTION_CONTENT_BOUNDARY_VIOLATION` | IC cert required but not verified |

## Testing

```bash
# Run Python self-tests
python qa_guardrail.py --validate

# Validate golden fixtures
python qa_guardrail.py --fixtures

# Test guard mode directly
echo '{"planned_move": "tool.bash({\"cmd\": \"ls\"})", "context": {"active_generators": ["sigma"]}}' \
  | python qa_guardrail.py --guard
```

## Security Model

1. **Default Closed**: Unknown tools are denied
2. **Fail Closed**: Guardrail errors result in DENY
3. **Content is Inert**: User input cannot become instructions
4. **Audit Trail**: All denials are logged with structured fail_record

## Integration with Gemini Threat Detection

The `verified` field in `instruction_content_cert` is a seam for plugging in Gemini's pattern-based threat detection:

```json
{
  "instruction_content_cert": {
    "schema_id": "QA_INSTRUCTION_CONTENT_SEPARATION_CERT.v1",
    "verified": true,
    "instruction_domain": ["sigma", "mu", "lambda", "nu"],
    "content_domain": ["user_input"]
  }
}
```

Set `policy.require_verified_ic_cert: true` to enforce verification.
