# [44] QA Rational Trig Type System (QA_RATIONAL_TRIG_TYPE_SYSTEM.v1)

Fuses **Rational Trigonometry** (Wildberger) with **Type Theory** ("Types vs Sets")
into a single QA cert family where:

- RT objects are **typed state manifolds** (Point2, Line2, Triangle) with formation rules
- RT theorems are **generator moves** (Cross law, Spread law, etc.) that add constraints
- Failures are **typed obstructions** (degenerate triangle, weak algebra, zero divisor)

## Machine tract

Directory: `qa_rational_trig_type_system_v1/`

Files:
- `qa_rational_trig_type_system_v1/schema.json`
- `qa_rational_trig_type_system_v1/validator.py`
- `qa_rational_trig_type_system_v1/fixtures/` (1 valid, 2 negative)
- `qa_rational_trig_type_system_v1/mapping_protocol_ref.json` (Gate 0 intake)
- `Documents/QA_MAPPING_PROTOCOL__RATIONAL_TRIG_TYPE_SYSTEM.v1.json` (mapping protocol)

### What it validates (gates)

- **Gate 1 -- Schema validity**: JSON Schema Draft-07 conformance
- **Gate 2 -- Determinism contract**: `canonical_json=true`, `no_rng=true`, `stable_sorting=true`, `invariant_diff` present
- **Gate 3 -- Typed formation**: Triangle non-collinearity check (`det([B-A, C-A]) != 0`)
- **Gate 4 -- Base algebra adequacy**: Laws requiring division (Spread law) need `integral_domain`/`field`
- **Gate 5 -- Step determinism**: `deterministic_hash = sha256(canonical_json({uses_law_id, inputs, outputs}))`

### Failure taxonomy (typed obstructions)

| Fail type | Trigger |
|-----------|---------|
| `DEGENERATE_TRIANGLE_COLLINEAR` | Triangle formation rule violated (collinear points) |
| `BASE_ALGEBRA_TOO_WEAK` | Law requires division but base ring lacks no-zero-divisors |
| `ZERO_DIVISOR_OBSTRUCTION` | Division by zero in derivation step |
| `LAW_PRECONDITION_FAILED` | Generator move preconditions not met |
| `LAW_EQUATION_MISMATCH` | Step claims equation satisfaction but LHS != RHS |
| `MISSING_INVARIANT_DIFF` | Certificate missing invariant_diff section |
| `NONDETERMINISM_CONTRACT_VIOLATION` | Step hash mismatch or contract flags not strict |

### Run

```bash
python qa_rational_trig_type_system_v1/validator.py --self-test
python qa_rational_trig_type_system_v1/validator.py qa_rational_trig_type_system_v1/fixtures/valid_minimal.json
```

### RT core laws encoded as generator moves

| Law ID | Name | Equation |
|--------|------|----------|
| RT_LAW_01 | Pythagoras (quadrance form) | Q_k = Q_i + Q_j (when s_k = 1) |
| RT_LAW_02 | Triple quad formula | (Q1+Q2+Q3)^2 = 2(Q1^2+Q2^2+Q3^2) |
| RT_LAW_03 | Spread law | s1/Q1 = s2/Q2 = s3/Q3 |
| RT_LAW_04 | Cross law | (Q1-Q2-Q3)^2 = 4*Q2*Q3*(1-s1) |
| RT_LAW_05 | Triple spread formula | (s1+s2+s3)^2 = 2(s1^2+s2^2+s3^2) + 4*s1*s2*s3 |

## Human tract

### "Types vs Sets" -- the core translation

| Concept | Set-theoretic view | Type-theoretic view | QA manifestation |
|---------|-------------------|--------------------|--------------------|
| Triangle | A set of 3 points {A, B, C} | Dependent type: `Triangle(A,B,C)` with formation rule requiring `det([B-A,C-A]) != 0` | Typed state that **fails to exist** under `DEGENERATE_TRIANGLE_COLLINEAR` |
| RT law | A fact: "for all triangles, equation X holds" | An elimination rule: given a well-formed `Triangle`, derive `constraint_X` | A **generator move** that requires typed preconditions and emits deterministic outputs |
| Division | Implicit (just divide) | Requires the base type to support it (field/integral_domain) | Gate 4 checks `BASE_ALGEBRA_TOO_WEAK` before allowing laws that divide |
| Failure | "Element not in the set" (generic) | **Typed obstruction**: which formation/elimination rule failed, and why | `fail_type` enum with deterministic witness (gate + inputs + obstruction label) |

The key philosophical point: in set theory, a degenerate triangle "isn't a triangle"
because it fails a membership test. In type theory (and in QA), it **fails to form** --
the construction rule `FR_TRIANGLE_NONCOLLINEAR` is not satisfiable, producing a typed
obstruction with a concrete witness (`det = 0`). The failure carries information; it is
not merely "not in the set."

This aligns with Martin-Lof type theory (formation + introduction + elimination +
computation rules) and HoTT (invariance under equivalence). Objects exist only when
their formation rules succeed.

### Why RT is "QA-shaped"

Rational Trigonometry replaces transcendental trig (sin, cos, arctan) with algebraic
invariants: **quadrance** (squared distance) and **spread** (squared sine, defined
algebraically). The five main laws are polynomial/rational relations over the base
algebra, making them natural transition contracts in a QA reachability framework:

- **State manifold**: Triangle(A,B,C) + observables (Q1,Q2,Q3,s1,s2,s3)
- **Generator moves**: RT laws add constraints to the typed state (acyclic derivation)
- **Terminal states**: fully constrained typed objects
- **Failure states**: typed obstructions (collinear, weak algebra, zero divisor)

No transcendental functions appear anywhere. Everything is polynomial/rational over
the declared base algebra, which is exactly the substrate QA operates on.

### RT laws as generator moves

Each RT law application is a move of the form:

```
(typed_state, observables, constraints)  --[RT_LAW_k]-->  (typed_state, observables, constraints + outputs)
```

A move is **legal only if**:
1. The object exists as a type (Triangle formation rule satisfied)
2. The base algebra supports the required operations (no division in a ring without `no_zero_divisors`)
3. The move's preconditions are met by current invariants
4. The move is deterministic under the contract (step hash matches canonical recomputation)
5. The move's equation holds (if numeric evaluation is claimed)

Failure is therefore a **typed obstruction**, not a "wrong element."

### Failure algebra with gate mapping

| Fail type | Meaning | Gate | Obstruction witness |
|-----------|---------|------|---------------------|
| `MISSING_INVARIANT_DIFF` | QA contract discipline violated | Gate 2 | Missing/malformed invariant_diff section |
| `NONDETERMINISM_CONTRACT_VIOLATION` | Step hash or contract flags wrong | Gate 2/5 | Expected vs actual hash |
| `DEGENERATE_TRIANGLE_COLLINEAR` | Typed formation failure | Gate 3 | det([B-A,C-A]) = 0 |
| `BASE_ALGEBRA_TOO_WEAK` | Algebraic substrate insufficient | Gate 4 | Law requires division, ring lacks it |
| `ZERO_DIVISOR_OBSTRUCTION` | Division by zero in derivation | Gate 4 | Denominator is zero divisor |
| `LAW_PRECONDITION_FAILED` | Generator move preconditions not met | Gate 3/4 | Missing observables or side-conditions |
| `LAW_EQUATION_MISMATCH` | Claimed equation satisfaction, but LHS != RHS | Gate 5 | Numeric mismatch |

### External semantics anchoring

This family's external semantics are anchored via:

- **Gate 0 (mapping protocol ref)**: `mapping_protocol_ref.json` sha256-locks
  `Documents/QA_MAPPING_PROTOCOL__RATIONAL_TRIG_TYPE_SYSTEM.v1.json`, which encodes
  the full state manifold, generators, invariants, failure taxonomy, and reachability
  analysis. Gate 0 is enforced by the meta-validator (`require_mapping_protocol()`),
  not by the family validator -- this is the standard pattern for families [35]+.

- **Source semantics**: Each cert instance carries `source_semantics.video_ref` (pinned
  to Wildberger's "Types vs Sets" lecture, timestamp 5:35) and `source_semantics.rt_refs`
  (pinned to Wildberger 2007, 2008 papers). These are structural fields, not yet
  sha256-locked to local copies.

### Sources

- Wildberger, N.J. (2007). "A Rational Approach to Trigonometry."
  https://web.maths.unsw.edu.au/~norman/papers/RationalTrig.pdf
- Wildberger, N.J. (2008). "The ancient Greeks present: Rational Trigonometry."
  https://arxiv.org/pdf/0806.3481
- Wildberger, N.J. (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry.*
- Video: "Towards a New Mathematics and Types versus Sets" (Wildberger)
  https://www.youtube.com/watch?v=x6rb-qOEXtQ (timestamp 5:35)
- Martin-Lof, P. (1984). *Intuitionistic Type Theory.*
- Univalent Foundations Program (2013). *Homotopy Type Theory.*

## Notes

- This family uses `mapping_protocol_ref.json` pointing to
  `Documents/QA_MAPPING_PROTOCOL__RATIONAL_TRIG_TYPE_SYSTEM.v1.json`
- The valid fixture demonstrates the Cross law on a right isosceles triangle (0,0)-(1,0)-(0,1)
- Negative fixtures test: missing invariant_diff (Gate 2 typed obstruction) and collinear points (Gate 3)
- Gate 0 is enforced at the meta-validator layer, not inside `validator.py` -- same as families [35]-[39]
