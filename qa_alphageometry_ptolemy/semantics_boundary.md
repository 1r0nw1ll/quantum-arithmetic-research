# Semantics vs Bounds — QARM TLA+ Bundle

> Primary source: (Lamport, 2002) Specifying Systems — ISBN: 978-0-321-14306-8.

This document separates the **intrinsic semantics** of the QARM specification
from the **bounds and caps** imposed by TLC model checking, so that a reader
does not mistake a TLC PASS at finite scope for a soundness proof at
unbounded scope.

## Intrinsic semantics

The QARM spec has well-defined intrinsic semantics independent of any
particular TLC configuration:

- States are 4-tuples `(b, e, d, a) ∈ Nat^4` plus auxiliary fields
  (`qtag`, `fail`, `lastMove`).
- `TupleClosed(b, e, d, a) == (d = b + e) /\ (a = d + e)` is a closure
  invariant on the QA derived coordinates; it is a total constraint, not
  bounded by `CAP`.
- The generator actions `Sigma`, `Mu`, `Lambda` are total partial-functions
  on `Nat^4` (lambda_k succeeds iff the result stays in the natural-number
  domain after scaling).
- The six axiom invariants (`Inv_A1`/`Inv_A2`/`Inv_T1`/`Inv_T2`/
  `Inv_S1_NoSquareOperator`/`Inv_S2_IntegerState`) and the firewall
  invariant `Inv_NT` are universally quantified over all reachable states
  under the unbounded transition system.

## Bounds imposed by TLC model checking

TLC verifies the invariants only over a **bounded reachable state space**:

| Bound | Value | Effect |
|---|---|---|
| `CAP` | 20 | Caps each of `b`, `e`, `d`, `a` at 20. |
| `KSet` | `{2, 3}` | Caps the lambda_k scaling factors. |
| Search depth | TLC default | Bounded breadth-first search; stutter-allowed. |

Within those bounds, TLC has verified each named invariant with **0 errors**
across all reachable states (see `QARM_PROOF_LEDGER.md` for the run records,
including the dual-platform parity check on Linux and Darwin). Beyond
`CAP=20`, the same invariants are **claimed** by structural induction from
the inductive case in the `Sigma`/`Mu`/`Lambda` actions, but that induction
is not mechanically verified — it is a paper argument, not a proof.

## What this distinction means for readers

A reader should **not** interpret a TLC PASS at `CAP=20` as a proof of
soundness for arbitrarily large `b`, `e`. Specifically:

- The intrinsic-semantics claim ("these axioms hold over all reachable
  states under the unbounded transition system") is **stronger** than the
  TLC-verified claim and is not certified by this bundle.
- The TLC-verified claim ("these axioms hold over all reachable states with
  `CAP=20`, `KSet={2,3}`") is what the bundle actually establishes.
- Cite the **bounded form** when reporting results downstream.

## Negative-test discipline

Each TLC PASS run is paired with one or more negative-test runs whose
configurations intentionally drop or violate one named invariant. The
expected outcome of each negative run is a TLC FAIL on the specific
invariant that was dropped, which provides outsider-facing confidence that
each invariant catches a real failure mode rather than being vacuously
satisfied. The negative-test pairings are recorded in
`QARM_PROOF_LEDGER.md`.

## What this document does NOT claim

- It does not certify that `CAP=20` is large enough to expose all bug
  classes; it only documents the verified state space.
- It does not claim that the structural-induction argument from bounded to
  unbounded scope is rigorous; the argument is informal and outside this
  bundle's scope.
- It does not formalize TLC itself; the model-checking semantics are
  inherited from (Lamport, 2002).

## References

- (Lamport, 2002) Specifying Systems. Addison-Wesley. ISBN: 978-0-321-14306-8.
- TLA+ Toolbox / TLC 2.20 — https://lamport.azurewebsites.net/tla/tla.html
- `QARM_PROOF_LEDGER.md` — the bundle's TLC run ledger.
