# QA_ENGINEERING_CORE_CERT.v1 — Proposed Cert Family Spec

**Proposed Family ID**: [121]
**Schema**: `QA_ENGINEERING_CORE_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]
**Status**: SPECIFICATION (not yet built — this is the design document)

---

## Purpose

`QA_ENGINEERING_CORE_CERT.v1` certifies that a classical engineering system (described in standard applied mathematics terms) maps validly to a QA specification. It is the formal bridge between:

- Classical control theory (state-space model, stability, controllability)
- QA engineering (orbit graph, generator set, reachability certificate)

This cert family would enable a practitioner to take any classical system and produce a machine-verifiable proof that:
1. Their system maps to a valid QA state space
2. Classical stability conditions correspond to QA invariant preservation
3. Classical controllability corresponds to QA reachability
4. The optimal path in classical terms corresponds to minimal-k in QA terms

---

## The Three Core Claims

### Claim 1: Valid QA Mapping (EC1–EC5)

A classical system `(A, B, C, x₀, x_target)` maps to QA iff:

- EC1: The system state can be encoded as `(b, e) ∈ Caps(N, N)` for some N
- EC2: System transitions correspond to legal generator applications (σ, μ, λ, or ν or domain-specific extensions)
- EC3: System failure modes can be classified into QA failure taxonomy (OUT_OF_BOUNDS, PARITY, INVARIANT, PHASÉ_VIOLATION, REDUCTION, or domain extensions)
- EC4: The target condition corresponds to membership in a QA orbit family (singularity / satellite / cosmos)
- EC5: The Q(√5) norm `f(b,e) = b²+be-e²` is preserved across system transitions (generator applications preserve orbit family classification)

### Claim 2: Stability ↔ Invariant Preservation (EC6–EC8)

- EC6: Classical Lyapunov stability (V(x) decreasing along trajectories) maps to QA invariant preservation (I = |C-F| > 0 maintained across all generator applications)
- EC7: The orbit contraction factor `ρ(O) = ∏(1-κ_t)²` provides a computable Lyapunov certificate: ρ(O) < 1 iff κ_min > 0 iff the system converges
- EC8: Classical equilibrium (Ax = 0) maps to QA singularity (orbit family = singularity, fixed point under generator F)

### Claim 3: Controllability ↔ Reachability (EC9–EC11)

- EC9: Classical Kalman rank condition (rank[B, AB, A²B, ...] = n) maps to QA BFS reachability (target orbit family reachable within max_depth_k from initial state)
- EC10: Classical optimal control (minimize ∫uᵀRu dt) maps to QA minimal path (minimize path_length_k via BFS with minimality_witness)
- EC11: If `v_p(r) = 1` for an inert prime p, the target state is arithmetically unreachable regardless of classical controllability analysis (obstruction spine supersedes Kalman rank check for modular systems)

---

## Proposed Failure Algebra

| Fail Type | Meaning |
|-----------|---------|
| `STATE_ENCODING_INVALID` | Classical state x cannot be encoded as (b,e) ∈ Caps(N,N) |
| `TRANSITION_NOT_GENERATOR` | A system transition has no corresponding generator in {σ,μ,λ,ν} or declared extensions |
| `FAILURE_TAXONOMY_INCOMPLETE` | A system failure mode cannot be classified into QA failure types |
| `TARGET_NOT_ORBIT_FAMILY` | Target condition does not correspond to a QA orbit family |
| `NORM_NOT_PRESERVED` | Q(√5) norm f(b,e) changes orbit family across a declared-legal transition |
| `LYAPUNOV_QA_MISMATCH` | Classical Lyapunov certificate present but QA orbit contraction factor ρ(O) ≥ 1 |
| `CONTROLLABILITY_QA_MISMATCH` | Kalman rank full but QA BFS reports target unreachable (check: arithmetic obstruction may override) |
| `ARITHMETIC_OBSTRUCTION_IGNORED` | System claims target reachable but v_p(r)=1 for inert prime p |
| `ORBIT_FAMILY_CLASSIFICATION_FAILURE` | Declared orbit family for a state is inconsistent with f(b,e) and its 3-adic valuation |

---

## Proposed Validator Checks (EC1–EC11 + IH1–IH3)

```
IH1: inherits_from = QA_CORE_SPEC.v1                    ← kernel inheritance
IH2: spec_scope = "family_extension"
IH3: gate sequence = [0,1,2,3,4,5]

EC1:  state_encoding.b and state_encoding.e are positive integers ≤ N
EC2:  all system_transitions map to a declared generator name
EC3:  all system_failures map to a QA fail_type
EC4:  target_condition.orbit_family ∈ {singularity, satellite, cosmos}
EC5:  for each transition, f(b',e') orbit_family = f(b,e) orbit_family (norm preserved)
      UNLESS the transition is explicitly an orbit-crossing move (satellite→cosmos etc.)
EC6:  stability_claim.lyapunov_function references I or f(b,e)
EC7:  stability_claim.orbit_contraction_factor is computed as ρ(O) = ∏(1-κ_t)² and < 1
EC8:  equilibrium_state maps to singularity orbit family
EC9:  reachability_witness is a BFS result reaching target_condition from initial_state
EC10: reachability_witness includes minimality_witness if optimization_claim is present
EC11: if v_p(r)=1 for any inert prime p, target is flagged OBSTRUCTED and
      optimization_claim.obstructed = true, nodes_expanded = 0
```

---

## Proposed Canonical Fixtures

### PASS: Classical Linear System → QA

A simple spring-mass system:
```
ẋ = Ax,  A = [[0, 1], [-ω², -2ζω]]
```
Discretized and mapped to:
- State: (b, e) = (displacement_class, velocity_class) ∈ Caps(9, 9)
- σ: increase velocity class (e → e+1)
- μ: swap displacement and velocity roles
- Orbit family: cosmos (periodic oscillation), satellite (damped transient), singularity (equilibrium)
- Stability claim: f(b,e) decreasing along damped trajectory → ρ(O) < 1 ← EC7 PASS
- Reachability: BFS from singularity to cosmos in k=2 ← EC9 PASS

**Result**: PASS. Classical spring-mass system maps validly to QA.

### FAIL: Classical System with Arithmetic Obstruction

A system targeting r = 6 with modulus 9:
```
System claims: target state (b=2, e=3) reachable (r = b·e = 6)
Kalman rank: full (classical controllability satisfied)
```
But: `v₃(6) = 1` and 3 is inert in Z[φ].
```
EC11: ARITHMETIC_OBSTRUCTION_IGNORED
```
The cert fails because the classical controllability check missed the modular arithmetic obstruction. **This is the core value of the QA engineering cert**: it catches what Kalman rank analysis cannot.

**Result**: FAIL. `ARITHMETIC_OBSTRUCTION_IGNORED`.

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  └── [121] QA_ENGINEERING_CORE_CERT.v1 (family_extension)
              Maps any classical (A,B,C) system to QA
              Proves: stability↔invariants, controllability↔reachability
              Catches: arithmetic obstructions invisible to classical theory
```

This family would then be the parent for:
- `[122] QA_SPRING_MASS_CERT.v1` — spring-mass domain instance
- `[123] QA_CIRCUIT_CERT.v1` — RLC circuit domain instance
- `[124] QA_ROBOTICS_CERT.v1` — robotic joint control domain instance
- etc.

---

## Implementation Plan

To build this cert family (next development session):

1. **Schema**: `qa_engineering_core_cert.v1.schema.json`
   - Fields: `classical_system` (A/B/C matrices or description), `qa_mapping` (state encoding, generator mapping, failure mapping), `stability_claim`, `reachability_witness`, `obstruction_check`
   - JSON Schema draft-07

2. **Validator**: `qa_engineering_core_cert_validate.py`
   - IH1–IH3: kernel inheritance checks (import from qa_core_spec validator)
   - EC1–EC5: mapping validity checks
   - EC6–EC8: stability equivalence checks
   - EC9–EC11: controllability/reachability checks
   - Self-test: 3 fixtures (1 PASS, 1 FAIL: arithmetic obstruction, 1 FAIL: invalid mapping)

3. **Mapping protocol**: `mapping_protocol_ref.json` referencing `qa_mapping_protocol_ref/` root

4. **Register in meta-validator**: add to `FAMILY_SWEEPS` in `qa_alphageometry_ptolemy/qa_meta_validator.py`

---

## Why Build This

1. **For SVP practitioners**: makes explicit that QA is a rigorous engineering framework with formal relationships to every standard engineering discipline
2. **For academic readers**: provides the bridge to classical control theory literature (Kalman, Pontryagin, Lyapunov)
3. **For AI-assisted engineering**: the cert structure gives any AI a formal way to verify that a proposed classical-to-QA mapping is correct
4. **For the builder tier**: once this cert exists, every domain a builder wants to map gets a standard template to follow

---

## Source References

- ChatGPT Engineering & Applied Mathematics foundation analysis (2026-03-24)
- Classical control: Kalman (1960), Pontryagin (1962), Lyapunov (1892)
- QA kernel: cert family [107] `QA_CORE_SPEC.v1`
- Obstruction spine: cert families [111]–[116]
- Finite-Orbit Descent: `memory/curvature_theory.md`
- `CLASSICAL_TO_QA_MAP.md`: the full equivalence table this cert certifies
