# QA Maxwell Derivation Program

Status: proof-program scaffold, not a registered cert.

Purpose: define what would count as "QA derives Maxwell" without importing
Maxwell's equations as hidden premises. This document replaces the blanket
claim-classifier answer "reject" with a sharper rule:

- "QA has already derived Maxwell" is rejected.
- "QA cannot derive Maxwell" is not established.
- "QA may derive Maxwell if the proof obligations below are met" is the
  correct open research position.

## Current Position

The current certified Whittaker ladder does not derive Maxwell.

Cert `[507]` proves a narrow exact-algebra statement: for QA-rational
plane-wave packets, Whittaker's 1904 two-scalar-potential operator yields
exact rational coefficients and several exact divergence identities. That is
useful structure, but it imports Whittaker's EM operator from a physics
primary source. It is not a derivation of electromagnetism from QA primitives.

Layer 5, as currently mapped, is an observer-side reconstruction/evaluation
layer. It may evaluate finite Whittaker packets and prove an approximation
bound under explicit smoothness and band-limit assumptions. That still is not
a Maxwell derivation.

## Derivation Standard

A future cert may say "QA derives Maxwell" only if it starts from declared
QA-native primitives and proves the Maxwell system, or an explicitly equivalent
discrete/continuum-limit system, without assuming Maxwell equations as input.

Minimum primitive list:

- QA state space and admissible labels, with A1/A2 respected.
- QA time as path length `k`, not continuous time.
- QA field carriers, defined internally: scalar, 1-cochain, 2-cochain, or
  equivalent exact finite objects over QA cells/orbits.
- QA differential/coboundary operator, defined combinatorially.
- QA metric/Hodge/constitutive object, or a proof that no extra metric object
  is needed.
- QA source/current object, if inhomogeneous equations are claimed.
- Explicit QA -> observer boundary map and crossing count under Theorem NT.

If any item is stipulated from classical EM, the resulting claim is
"conditional recovery under declared observer-side assumptions", not
"derivation".

## Equation Split

Maxwell has two structurally different halves. They should not be certified in
one jump.

### Homogeneous Half

Target, in exterior-calculus form:

```text
dF = 0
```

Equivalent vector form, convention-dependent:

```text
div B = 0
curl E + partial_t B = 0
```

QA route:

1. Build a QA combinatorial cell/cochain complex.
2. Define a QA potential carrier `A`.
3. Define field carrier `F = delta_QA(A)`.
4. Prove `delta_QA(delta_QA(A)) = 0` from boundary-of-boundary nilpotency.
5. Show that, under an observer projection, this becomes the homogeneous
   Maxwell pair.

This is the plausible first derivation target. It is an algebraic identity,
not yet physical dynamics.

### Inhomogeneous Half

Target, in exterior-calculus form:

```text
d star F = J
```

Equivalent vector form, convention-dependent:

```text
div E = rho / epsilon
curl B - mu epsilon partial_t E = mu J
```

QA route:

1. Derive or declare a QA Hodge/metric operator `star_QA`.
2. Derive or declare a QA source/current `J_QA`.
3. Prove the constitutive/source equation
   `delta_QA(star_QA(F_QA)) = J_QA`.
4. Prove source continuity from nilpotency:
   `delta_QA(J_QA) = 0`.
5. Show the observer projection recovers the standard inhomogeneous Maxwell
   equations under declared units and sign conventions.

This is the hard part. A Hodge star is not free: it encodes metric, orientation,
and constitutive physics. If `star_QA` is imported from Euclidean/Minkowski
observer geometry, the cert has not derived full Maxwell from QA; it has
conditionally recovered it.

## Proposed Cert Ladder

### M0: `[508]` QA Discrete Exterior Nilpotency

Slug: `qa_discrete_exterior_nilpotency_cert_v1`

Status: built 2026-07-03.

Claim: on a finite oriented QA cell complex, the QA coboundary applied twice is
identically zero.

Allowed: pure combinatorics, exact integers/Fractions, no physical fields.

Rejected: any mention that this already proves electromagnetism.

Implemented checks: `DN_1` through `DN_7`; 1 PASS + 4 FAIL fixtures.

### M1: [509] QA Field 2-Form Bianchi

Slug: `qa_field_2form_bianchi_cert_v1`

Status: built 2026-07-03.

Claim: if `A_QA` is a QA 1-carrier and `F_QA = delta_QA(A_QA)`, then
`delta_QA(F_QA) = 0`; under a declared observer projection this matches the
homogeneous Maxwell equations.

Allowed: "QA derives the homogeneous Maxwell/Bianchi identities for exact
field carriers."

Rejected: "QA derives all Maxwell equations."

Implemented checks: `BIA_1` through `BIA_8`; 1 PASS + 4 FAIL fixtures.

What is actually proved: an exact finite integer edge-potential 1-cochain `A`
induces a face-field 2-cochain `F = delta(A)`, and the signed sum `delta(F)`
vanishes on every declared volume. This is the discrete exterior Bianchi
identity / homogeneous Maxwell side for exact field carriers.

What is still not proved: inhomogeneous Maxwell, a Hodge star, constitutive
laws, sources, charge/current generation, units, physical electromagnetism, or
observer-side numeric fields.

### M2: [510] QA Hodge Boundary / Constitutive Gate

Slug: `qa_hodge_constitutive_boundary_cert_v1`

Status: built 2026-07-03.

Claim: classify whether `star_QA` is QA-native or observer-side.

Allowed outcomes:

- `QA_NATIVE`: `star_QA` is built from QA invariants with no imported metric.
- `OBSERVER_BOUNDARY`: `star_QA` is supplied by observer geometry/medium.
- `INVALID`: hidden continuous/float/metric import inside QA logic.

This gate decides whether the later inhomogeneous claim can be called a QA
derivation or only a conditional recovery.

Implemented checks: `HCB_1` through `HCB_8`; 2 PASS + 5 FAIL fixtures.

Built result: [510] now has both an `OBSERVER_BOUNDARY` witness and a
`QA_NATIVE` seed witness. The observer-boundary witness uses exact rational
matrix entries for `star_QA`, but metric signature, orientation, units, and
medium parameters are declared observer imports. The QA-native seed has exact
rational matrix entries, native integer-cell pairing, QA-invariant metric
source evidence, exact orientation witness, and no observer imports. This fixes
the Hodge-evidence side of the blocker, but still does not license "QA derives
full Maxwell"; source evidence and M4/M5 recovery/assembly remain required.

### M3: [511] QA Source Continuity

Slug: `qa_source_continuity_cert_v1`

Status: built 2026-07-03.

Claim: if `J_QA = delta_QA(star_QA(F_QA))`, then
`delta_QA(J_QA) = 0`.

Allowed: current conservation / charge continuity as a structural consequence.

Rejected: physical source law unless source generation is also derived.

Implemented checks: `SRC_1` through `SRC_8`; 2 PASS + 5 FAIL fixtures.

Built result: source continuity is certified as a finite cochain nilpotency
consequence after either `[510]`'s `OBSERVER_BOUNDARY` gate or its `QA_NATIVE`
Hodge seed. The native branch requires explicit QA source-carrier evidence:
`J` is an exact cochain and no observer source imports are present. This fixes
the native source-carrier evidence side of the blocker, but it is still not
physical charge/current generation and not inhomogeneous Maxwell yet.

### M4: [512] QA Inhomogeneous Maxwell Recovery

Slug: `qa_inhomogeneous_maxwell_recovery_cert_v1`

Status: built 2026-07-03.

Claim: under a declared `star_QA` and source construction, the observer
projection recovers `d star F = J` with explicit units/sign conventions.

Allowed if M2 is `OBSERVER_BOUNDARY`: "conditional recovery of inhomogeneous
Maxwell under declared constitutive assumptions."

Allowed if M2 is `QA_NATIVE` and source-carrier evidence is QA-native:
"QA-native symbolic recovery of the inhomogeneous Maxwell equation."

Implemented checks: `IMR_1` through `IMR_9`; 2 PASS + 5 FAIL fixtures.

Built result: exact finite recovery/assembly is certified at M4:
`starF = star_QA(F)` and `J = delta(starF)` are recomputed as exact rational
cochains under declared sign/unit/projection conventions. M4 still rejects full
Maxwell, physical electromagnetism, physical fields, and physical source
generation. The remaining work is M5: assemble the already-built homogeneous
and inhomogeneous halves into a bounded full-Maxwell claim.

### M5: Full Maxwell Derivation Claim

Candidate slug: `qa_maxwell_derivation_cert_v1`

Claim allowed only if M1, M2, M3, and M4 pass and M2 is `QA_NATIVE`.

Required exact phrase:

```text
QA derives Maxwell only within the stated carrier, boundary, source, metric,
unit, and observer-projection conventions certified here.
```

This cert must include negative fixtures that reject:

- importing Maxwell equations as assumptions;
- using Whittaker's operator as a premise while claiming QA-native derivation;
- using observer-side floats/trig before the QA exact proof is complete;
- hiding the Hodge star inside notation;
- claiming physical electromagnetism beyond the certified projection;
- claiming scalar-wave-energy, free-energy, Bearden/Pond/SVP, or
  longitudinal-energy physics.

## Relation To Whittaker Layers

The Whittaker ladder is evidence of compatibility, not derivation.

Layer 4 `[507]` can serve as a bridge fixture for "this QA packet algebra is
consistent with a classical scalar-potential representation." It cannot serve
as the root proof of Maxwell because Whittaker's formulas already live inside
classical electromagnetic theory.

Layer 5 can be a reconstruction/evaluation cert consuming `[507]`.

The Maxwell derivation program should run beside the Whittaker ladder, not
inside it, until the QA-native carrier/differential/Hodge/source pieces exist.

## Claim Classifier

| Claim text | Disposition |
| --- | --- |
| "QA has derived Maxwell" | Reject today |
| "QA cannot derive Maxwell" | Not established |
| "QA derives homogeneous Maxwell/Bianchi identities" | Built in `[509]` for exact finite field carriers only; not full Maxwell |
| "QA conditionally recovers inhomogeneous Maxwell under an imported Hodge star" | Built in `[512]` for the observer-boundary branch; still not full derivation |
| "QA has native Hodge/source-carrier evidence" | Seed evidence built in `[510]` and `[511]`; M4 native symbolic recovery built in `[512]` |
| "QA derives all Maxwell equations from QA-native primitives" | Future M5 only if the claim is bounded to the certified carrier/source/metric/unit/projection conventions |
| "Whittaker `[507]` derives Maxwell" | Reject |
| "Whittaker `[507]` supplies exact packet algebra compatible with a scalar-potential EM representation" | Already certified, narrow |

## Open Technical Questions

- What is the canonical QA cell complex: orbit graph, product lattice, or a
  cellulation induced by rational direction packets?
- Is the QA differential the graph coboundary, a higher-dimensional cell
  coboundary, or a T-operator-derived finite difference?
- Can a Hodge star be built from QA invariants such as `C`, `F`, `G`, orbit
  class, and path length without importing observer metric structure?
- What is the QA-native source/current object, if any?
- Does the observer projection recover Lorentz-covariant Maxwell equations,
  or only a discrete anisotropic approximation?
- Which sign/unit convention is canonical for the cert fixtures?

## References

- J. C. Maxwell (1865), "A Dynamical Theory of the Electromagnetic Field,"
  Philosophical Transactions of the Royal Society of London 155:459-512.
- E. T. Whittaker (1904), "On an expression of the electromagnetic field due
  to electrons by means of two scalar potential functions," Proc. London Math.
  Soc. s2-1:367-372. DOI: 10.1112/plms/s2-1.1.367.
- Existing QA context: `[507]`
  `qa_whittaker_two_scalar_potential_bridge_cert_v1`,
  `docs/specs/QA_WHITTAKER_LAYER5_LAYER6_CLAIM_BOUNDARY.md`,
  `docs/theory/QA_QFT_ETCR_CROSSMAP.md`.
