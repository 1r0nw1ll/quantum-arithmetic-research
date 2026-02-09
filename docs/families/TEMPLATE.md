# [NN] Family Name

## What this is

One paragraph: what claim this family certifies and why it exists in QA.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__XXX.yaml` |
| Semantics cert | `certs/QA_XXX_SEMANTICS_CERT.v1.json` |
| Witness pack | `certs/witness/QA_XXX_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_XXX_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_xxx_validator.py` |
| Schema(s) | `schemas/QA_XXX_*.schema.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate this family
python qa_xxx_validator.py --demo

# Or via meta-validator
python qa_meta_validator.py
```

## Semantics

Bullet list of invariants this family asserts.

- **Invariant 1**: description
- **Invariant 2**: description

### Hash domains (if applicable)

| Domain | Used for |
|--------|----------|
| `QA/XXX/v1` | description |

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `EXAMPLE_FAILURE` | What went wrong | How to fix it |

## Example

**Passing**:
```json
{
  "minimal_passing_example": "..."
}
```

**Failing** (`EXAMPLE_FAILURE`):
```json
{
  "minimal_failing_example": "...",
  "expected_fail_type": "EXAMPLE_FAILURE"
}
```

## Changelog

- **vX.Y.Z** (YYYY-MM-DD): Initial release.
