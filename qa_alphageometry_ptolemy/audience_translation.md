# Audience Translation — QARM TLA+ Bundle

> Primary source: (Lamport, 1994) The Temporal Logic of Actions — DOI: 10.1145/177492.177726.

This document translates the project-private vocabulary of the QARM (QA
Recursive Module) TLA+ bundle into ordinary TLA+ terminology so that an
outsider TLA+ reader can ground the spec without first internalizing the QA
project's conventions.

## What is being modeled

The TLA+ files in this directory model the discrete generator algebra of
Quantum Arithmetic (QA, in the project sense — i.e. an integer constraint
system with derived coordinates, not the physics word). The state is a
4-tuple `(b, e, d, a)` of natural numbers with `d = b + e` and `a = d + e`,
plus an integer modular packing `qtag = 24 * Phi9(a) + Phi24(a)` and three
auxiliary state variables (`fail` failure-mode classifier, `lastMove`
provenance trace, observer crossing counters in `QAAxioms.tla`). The
generator actions are:

- `Sigma` — succession step `(b, e) -> (e, b + e)`, also written `σ`.
- `Mu` — coordinate involution `(b, e) -> (e, b)`, also written `μ`.
- `Lambda` — scaling `(b, e) -> (k*b, k*e)` for `k ∈ KSet`, also written `λ_k`.

`Init` initializes the tuple within `0..CAP`. `Next` is the disjunction of
the three generator actions and their explicit failure variants. `Spec`
combines `Init` with `[][Next]_vars` plus weak-fairness guards.

## Why the model is useful

For TLA+ practitioners: the bundle is a worked example of lifting a
text-level lint into TLC-checkable temporal invariants. The QA project
already has a Python linter (`tools/qa_axiom_linter.py`) that enforces six
axioms on source text. `QAAxioms.tla` makes the same axioms checkable at
runtime as `[]Inv_*` properties of the QARM transition system. TLC verifies
each invariant over the bounded reachable state space and reports the first
violating action when one occurs. This gives the project a second
independent witness on what the axioms mean: when the linter and TLC
disagree, that is a bug to investigate.

For outside readers: the bundle exemplifies the
"text-rule -> TLA+ invariant" pattern at a small scale (one module per
generator family, six axiom invariants, paired non-vacuity tests). It is
useful as a template for similarly-structured projects with a text-level
correctness layer.

## How project-private terms translate

- "Theorem NT" / "Observer Projection Firewall" — i.e. the safety invariant
  `Inv_NT` in `QAAxioms.tla` which states that a float-valued observer
  variable, once written by the `Project` action, never feeds back into the
  QA-side state variables. This is ordinary TLA+ refinement-style
  separation between two state subsets, expressed as a temporal invariant.
- "QA legality" — i.e. the conjunction of all six axiom invariants
  (`Inv_A1`, `Inv_A2`, `Inv_T1`, `Inv_T2`, `Inv_S1_NoSquareOperator`,
  `Inv_S2_IntegerState`) plus `Inv_NT`. A reachable state is "legal" iff
  every conjunct holds. The negative-test specs (`*_negative_*.cfg`) each
  drop one conjunct and verify TLC reports the dropped invariant as the
  first to fail, which provides outsider-facing confidence that each
  axiom is doing real work (not vacuously satisfied).
- "Lane 2" / "lane-two" — i.e. the second of three layers in the project's
  process, here mapping to the TLA+ verification layer (the first lane is
  the text-level linter; the third is the Python cert-validator family).
  The three lanes differ by what they enforce, not by what they claim;
  they are distinct redundant checks rather than a hierarchy.

## What this document does NOT claim

- It does not claim TLC has verified the QARM invariants for unbounded
  states. The verified state space is bounded by `CAP=20` and `KSet={2,3}`.
- It does not claim that the bundle is appropriate for upstream submission
  to `tlaplus/Examples`. That is a separate scope-of-fit question covered
  in `repo_fit_review.json`.
- It does not formalize the observer-side projections (continuous-to-discrete
  boundary). Only the QA-side firewall invariant `Inv_NT` is checked.

## References

- (Lamport, 1994) The Temporal Logic of Actions. ACM TOPLAS 16(3). DOI: 10.1145/177492.177726.
- (Lamport, 2002) Specifying Systems. Addison-Wesley. ISBN: 978-0-321-14306-8.
- `docs/specs/QA_TLA_PLUS.md` — three-layer constitutional architecture.
- `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` — Theorem NT spec.
