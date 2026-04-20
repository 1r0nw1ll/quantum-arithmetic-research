# Prospective A/B — Arm B (mapping-first) raw output

**Date**: 2026-04-19
**Cert concept**: `qa_sequence_cert_v1` — validate a list of (b,e) mod-9 tuples against QA axioms.
**Instruction**: author `mapping_protocol_ref.json` first (declare observer projection + axiom applicability), then derive `validator.py` with each check tracing to a mapping field.
**Subagent**: general-purpose, 32.7k tokens, 62s duration, 0 tool calls (worked from spec only).

## Artifact 1: mapping_protocol_ref.json

```json
{
  "schema_version": "qa_mapping_protocol_ref.v1",
  "cert_family": "qa_sequence_cert_v1",
  "title": "QA Discrete State Sequence Validator",
  "spec_authority": "docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md",
  "modulus": 9,
  "observer_projection": {
    "real_world_referent": "A finite, ordered trajectory of QA discrete states, each presented as a (b, e) integer pair on modulus m=9.",
    "boundary_crossings": {
      "input_to_observer": "Upstream producer emits a Python list[tuple[int, int]] — the sole entry into the QA layer.",
      "qa_to_output": "Validator emits a discrete verdict dict {pass, violations, checks_run} — the sole exit from the QA layer."
    },
    "firewall_stance": "No continuous/float quantity ever re-enters as causal QA input. The sequence is treated as pure discrete symbols; validation is integer-only.",
    "path_time_interpretation": "Index position i in the sequence IS QA time. There is no continuous time variable; k = i is the integer path length."
  },
  "discrete_state_space": {
    "primitive_states": "Pairs (b, e) with b, e integers in {1, 2, ..., 9}. Zero is forbidden (A1).",
    "derived_coords": {
      "d": "d_i = ((b_i + e_i - 1) mod 9) + 1  — derived from (b, e), never independent",
      "a": "a_i = ((b_i + 2*e_i - 1) mod 9) + 1  — derived from (b, e), never independent"
    },
    "step_operator": "qa_step(b, e) = ((b + e - 1) mod 9) + 1",
    "transition_rule": "b_{i+1} = qa_step(b_i, e_i); e_{i+1} free in {1..9}",
    "path_time": "Integer step index i in [0, len(sequence)-1]. No dt, no continuous parametrization."
  },
  "applicable_axioms": {
    "A1_no_zero": {"applies": true, "why": "State space is explicitly {1..9}.", "checks": ["each b_i in {1..9}", "each e_i in {1..9}", "derived d_i, a_i in {1..9} via no-zero formula"]},
    "A2_derived_coords": {"applies": true, "why": "d and a are not carried in primitive state — must be re-derivable from (b, e). The 2-tuple shape enforces this structurally.", "checks": ["each element is a 2-tuple (b, e), never a 3- or 4-tuple supplying d or a independently"]},
    "T2_firewall": {"applies": true, "why": "State must be pure integer symbols; a float in b/e = observer-layer leak (T2-b).", "checks": ["type(b_i) is int (bool excluded)", "type(e_i) is int (bool excluded)"]},
    "S1_no_pow2": {"applies": false, "why": "This cert only inspects sequence data, not source code. S1 is a code-level linter concern for validators computing squares; we compute no squares, so there is nothing to enforce on the input sequence."},
    "S2_no_float_state": {"applies": true, "why": "Equivalent at data level to T2 for this cert: b/e must be int or integer-valued Fraction.", "checks": ["b_i, e_i are int (Fraction with denominator==1 accepted); any float rejected"]},
    "T1_path_time": {"applies": true, "why": "The sequence encodes QA time as integer index.", "checks": ["sequence is list or tuple (ordered, integer-indexed)", "no element carries an extra timestamp/time field (tuple arity strictly 2 — dovetails with A2 check)"]}
  },
  "transition_checks": {
    "consecutive_b_matches_qa_step": "For i in [0, N-2]: b_{i+1} == ((b_i + e_i - 1) mod 9) + 1. Violations listed per-index.",
    "rationale": "Enforces that the sequence is a valid QA trajectory rather than an arbitrary list of states."
  },
  "validator_contract": {
    "entry_point": "validate(sequence) -> dict",
    "return_shape": {"pass": "bool", "violations": "list[dict]", "checks_run": "list[str]"},
    "derivation_note": "Every check_id in checks_run traces to a field in applicable_axioms[*].checks or transition_checks. No check exists in validator.py that is not declared here."
  },
  "qa_compliance_declaration": ["A1", "A2", "T2", "S2", "T1"]
}
```

## Artifact 2: validator.py (derived from mapping)

```python
"""qa_sequence_cert_v1 validator.

Every check in this module is derived from a field in mapping_protocol_ref.json;
nothing is enforced that the mapping did not declare.

Authority: docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md
"""

from fractions import Fraction

QA_COMPLIANCE = {
    "cert_family": "qa_sequence_cert_v1",
    "axioms_enforced": ["A1", "A2", "T2", "S2", "T1"],
    "axioms_not_applicable": ["S1"],
    "modulus": 9,
    "mapping_ref": "mapping_protocol_ref.json",
}

M = 9


def _qa_step(b, e):
    """A1-compliant step: ((b + e - 1) mod m) + 1."""
    return ((b + e - 1) % M) + 1


def _derive_d(b, e):
    """A2: d = b + e, reduced with A1."""
    return ((b + e - 1) % M) + 1


def _derive_a(b, e):
    """A2: a = b + 2e, reduced with A1."""
    return ((b + 2 * e - 1) % M) + 1


def _is_pure_int(x):
    """T2/S2: accept int (but not bool) and Fraction with denominator 1."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return False


def validate(sequence):
    violations = []
    checks_run = []

    # T1: mapping.applicable_axioms.T1_path_time.checks[0]
    checks_run.append("T1:container_is_ordered_sequence")
    if not isinstance(sequence, (list, tuple)):
        violations.append({"axiom": "T1", "index": None,
                           "detail": f"sequence must be list or tuple, got {type(sequence).__name__}"})
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    # A2/T1: mapping.applicable_axioms.A2_derived_coords.checks[0] + T1.checks[1]
    checks_run.append("A2:element_is_2_tuple_no_independent_d_or_a")
    checks_run.append("T1:no_extra_time_field_per_element")
    # A1: mapping.applicable_axioms.A1_no_zero.checks
    checks_run.extend(["A1:b_in_1_to_9", "A1:e_in_1_to_9",
                       "A1:derived_d_in_1_to_9", "A1:derived_a_in_1_to_9"])
    # T2/S2: mapping.applicable_axioms.T2_firewall.checks + S2.checks
    checks_run.extend(["T2:b_is_int_no_float_cast", "T2:e_is_int_no_float_cast",
                       "S2:b_is_int_or_unit_fraction", "S2:e_is_int_or_unit_fraction"])

    for i, elem in enumerate(sequence):
        if not isinstance(elem, tuple) or len(elem) != 2:
            violations.append({"axiom": "A2", "index": i,
                               "detail": f"element must be a 2-tuple (b, e); got {type(elem).__name__}"})
            continue
        b, e = elem
        if not _is_pure_int(b):
            violations.append({"axiom": "T2", "index": i,
                               "detail": f"b must be int (or unit Fraction); got {type(b).__name__}={b!r}"})
            continue
        if not _is_pure_int(e):
            violations.append({"axiom": "T2", "index": i,
                               "detail": f"e must be int (or unit Fraction); got {type(e).__name__}={e!r}"})
            continue
        b = int(b); e = int(e)
        if not (1 <= b <= M):
            violations.append({"axiom": "A1", "index": i, "detail": f"b={b} not in {{1..{M}}}"})
        if not (1 <= e <= M):
            violations.append({"axiom": "A1", "index": i, "detail": f"e={e} not in {{1..{M}}}"})
        d = _derive_d(b, e); a = _derive_a(b, e)
        if not (1 <= d <= M):
            violations.append({"axiom": "A1", "index": i, "detail": f"derived d={d} escaped {{1..{M}}}"})
        if not (1 <= a <= M):
            violations.append({"axiom": "A1", "index": i, "detail": f"derived a={a} escaped {{1..{M}}}"})

    # TRANSITION: mapping.transition_checks.consecutive_b_matches_qa_step
    checks_run.append("TRANSITION:consecutive_b_matches_qa_step")
    for i in range(len(sequence) - 1):
        cur, nxt = sequence[i], sequence[i + 1]
        if not isinstance(cur, tuple) or not isinstance(nxt, tuple) or len(cur) != 2 or len(nxt) != 2:
            continue
        b_cur, e_cur = cur
        b_nxt, _ = nxt
        if not (_is_pure_int(b_cur) and _is_pure_int(e_cur) and _is_pure_int(b_nxt)):
            continue
        b_cur, e_cur, b_nxt = int(b_cur), int(e_cur), int(b_nxt)
        expected = _qa_step(b_cur, e_cur)
        if b_nxt != expected:
            violations.append({"axiom": "TRANSITION", "index": i,
                               "detail": f"b[{i+1}]={b_nxt} != qa_step(b[{i}]={b_cur}, e[{i}]={e_cur})={expected}"})

    return {"pass": len(violations) == 0, "violations": violations, "checks_run": checks_run}
```

**Axioms selected by the mapping (one sentence):** A1, A2, T2, S2, T1 enforced; **S1 declared non-applicable with reason** ("validator inspects sequence data, not source code; computes no squares"); validator checks all trace to mapping fields via `mapping.applicable_axioms.*.checks` references.
