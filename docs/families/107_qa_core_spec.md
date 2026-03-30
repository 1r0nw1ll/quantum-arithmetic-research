# [107] QA Core Spec Kernel

**Family ID**: 107
**Schema**: `QA_CORE_SPEC.v1`
**Scope**: `kernel` (root executable ontology for all downstream QA cert families)
**Directory**: `qa_alphageometry_ptolemy/qa_core_spec/`

---

## Purpose

A `QA_CORE_SPEC` instance is the base executable specification of a QA system.
It is not documentation — it is a deterministic, failure-complete description of:

- a **state space** (typed variables + algebraic constraints)
- a **generator set** (each with kind, action, domain preconditions, emitted failure modes, and preserved invariants)
- **invariants** (named algebraic rules that must hold across all reachable states)
- **reachability semantics** (bounded BFS or equivalent)
- a **failure algebra** (nonempty classified list of fail types)
- **logging requirements** (required fields: `move`, `fail_type`, `invariant_diff`)
- a **validator gate policy** (must declare gates `[0,1,2,3,4,5]`)
- a **certificate contract** (downstream cert format + required witness fields)

All downstream QA cert families should reference this spec via `spec_scope: "family_extension"`.

---

## Directory Layout

```
qa_core_spec/
  mapping_protocol_ref.json           Gate 0 protocol reference
  schemas/
    qa_core_spec_v1.schema.json       JSON Schema (draft-07)
  fixtures/
    qa_core_spec_minimal_pass.json    Canonical kernel witness (PASS)
    qa_core_spec_fail_missing_failure_algebra.json  FAIL: failure_algebra.types=[]
    qa_core_spec_fail_bad_gate_sequence.json         FAIL: gates=[0,1,2,3]
    qa_core_spec_fail_duplicate_generator_name.json  FAIL: two generators named "sigma"
  qa_core_spec_validate.py            Validator + self-test
```

---

## Validator Checks (V1–V5)

| Check | Description | Fail Type |
|-------|-------------|-----------|
| V1 | Generator names are unique | `DUPLICATE_GENERATOR_NAME` |
| V2 | `failure_algebra.types` is nonempty | `MISSING_FAILURE_ALGEBRA` |
| V3 | `validation.gates` is exactly `[0,1,2,3,4,5]` | `BAD_GATE_SEQUENCE` |
| V4 | `logging.required_fields` ⊇ `{move, fail_type, invariant_diff}` | `LOGGING_INCOMPLETE` |
| V5 | All `preserves_invariants` references resolve to declared invariants | `INVARIANT_REFERENCE_UNRESOLVED` |

---

## Failure Algebra

| Fail Type | Meaning |
|-----------|---------|
| `DUPLICATE_GENERATOR_NAME` | Two or more generators share the same name; makes BFS transition labels ambiguous |
| `MISSING_FAILURE_ALGEBRA` | `failure_algebra.types` is empty; spec cannot support deterministic failure replay |
| `BAD_GATE_SEQUENCE` | `validation.gates` ≠ `[0,1,2,3,4,5]`; missing Gate 4 (invariant_diff) and/or Gate 5 (Merkle hash integrity) breaks tamper-evidence |
| `LOGGING_INCOMPLETE` | `logging.required_fields` missing one or more of `move`, `fail_type`, `invariant_diff` |
| `INVARIANT_REFERENCE_UNRESOLVED` | A generator's `preserves_invariants` entry names an invariant not declared in `invariants[]` |

---

## Gates

The full gate sequence `[0,1,2,3,4,5]` is required for kernel specs:

- **Gate 0**: Mapping protocol intake (Gate 0 constitution)
- **Gate 1**: Schema validation
- **Gate 2**: Generator uniqueness + invariant reference resolution
- **Gate 3**: Failure algebra completeness
- **Gate 4**: Invariant diff check (catches silent invariant violations without explicit fail_type)
- **Gate 5**: Canonical hash / Merkle integrity (closes the replay chain, makes certs tamper-evident)

Truncating to `[0,1,2,3]` removes tamper-evidence. This is what `BAD_GATE_SEQUENCE` catches.

---

## The Kernel Spec

The canonical kernel (`qa_core_spec_minimal_pass.json`) specifies the **QA tuple system**:

**State space**: `(b, e, d, a)` where `b + e = d` and `e + d = a`

**Generators**:
| Name | Kind | Action | Emits |
|------|------|--------|-------|
| `sigma` | bijection | `e → e+1 (mod m)`; propagates to d, a | `OUT_OF_BOUNDS` |
| `mu` | bijection | `swap(b,e)`; recomputes d=b+e, a=b+2e | — |
| `lambda` | partial | scale all coordinates by k (k≠0) | `ZERO_DENOMINATOR`, `OUT_OF_BOUNDS` |
| `nu` | contract | divide all coordinates by 2 when all even | `PARITY` |

**Failure algebra**: `OUT_OF_BOUNDS`, `PARITY`, `INVARIANT_BREAK`, `ZERO_DENOMINATOR`

**Reachability**: bounded BFS, max_depth=24, connected by invariant equivalence classes

---

## FAIL Fixture Rationale

### `MISSING_FAILURE_ALGEBRA`
`failure_algebra.types = []`. This is the most critical failure mode: a spec with no classified failure modes cannot support deterministic failure replay, invariant diff logging, or cert-level failure documentation. QA's primary advantage over DeepMind/Anthropic-style specs is failure completeness — a spec that is silent about failures has forfeited that advantage.

### `BAD_GATE_SEQUENCE`
`validation.gates = [0,1,2,3]` — missing gates 4 and 5. Gate 4 catches silent invariant violations (tuples that drift without emitting a fail_type). Gate 5 closes the replay chain with a Merkle hash, making the cert tamper-evident and externally auditable. A spec without these two gates cannot produce certs that pass external audit.

### `DUPLICATE_GENERATOR_NAME`
Two generators both named `"sigma"`. Duplicate names create ambiguity in failure_modes_emitted lookups, BFS transition graph construction, and preserves_invariants cross-references. The validator detects this with a counter (`names.count(n) > 1`) and emits `DUPLICATE_GENERATOR_NAME`.

---

## Running

```bash
# Self-test (all 4 fixtures)
python qa_core_spec/qa_core_spec_validate.py --self-test

# Single cert
python qa_core_spec/qa_core_spec_validate.py --file qa_core_spec/fixtures/qa_core_spec_minimal_pass.json

# Full meta-validator
python qa_meta_validator.py
```

Expected output (self-test):
```json
{"ok": true, "results": [
  {"fixture": "qa_core_spec_minimal_pass.json", "ok": true, "label": "PASS"},
  {"fixture": "qa_core_spec_fail_missing_failure_algebra.json", "ok": true, "label": "PASS"},
  {"fixture": "qa_core_spec_fail_bad_gate_sequence.json", "ok": true, "label": "PASS"},
  {"fixture": "qa_core_spec_fail_duplicate_generator_name.json", "ok": true, "label": "PASS"}
]}
```

---

## Downstream Inheritance

Families extending this kernel should:
1. Set `spec_scope: "family_extension"` in their cert
2. Reference the kernel cert ID in their `certificate_contract` or `parent_spec`
3. Add any domain-specific generators with their own `failure_modes_emitted`
4. Extend `failure_algebra.types` with domain-specific fail types
5. Preserve the full gate sequence `[0,1,2,3,4,5]`
