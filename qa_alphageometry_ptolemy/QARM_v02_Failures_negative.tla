------------------------- MODULE QARM_v02_Failures_negative -------------------------
EXTENDS Naturals, Integers, TLC

(*
  QARM_v02_Failures_negative.tla

  NEGATIVE TEST SPEC for QARM_v02_Failures.

  Deliberately introduces a single action that violates Inv_TupleClosed
  (the canonical tuple-closure invariant d = b+e, a = d+e). If TLC reports
  "no error" for this spec, the invariant is vacuously satisfied in the
  main QARM_v02_Failures spec and the Run 1 result is meaningless.

  Expected TLC result: Inv_TupleClosed violated after exactly 2 states, with
  a counterexample trace showing BrokenTupleClosure writing d' != b' + e'.

  This spec follows the pattern used in llm_qa_wrapper/spec/cert_gate_negative.tla
  (4 non-vacuity specs, one per checkable invariant). Future work: add
  cert_gate_negative_bounds / cert_gate_negative_qdef / cert_gate_negative_fail
  for the other four QARM invariants (InBounds, QDef, FailDomain, MoveDomain).
*)

VARIABLES
  b, e, d, a,
  qtag,
  fail,
  lastMove

vars == <<b, e, d, a, qtag, fail, lastMove>>

(*** Helpers (copied from QARM_v02_Failures; kept local so this module is
     standalone and doesn't pull the full spec's transition machinery) ***)

DR(n) ==
  IF n = 0 THEN 0 ELSE 1 + ((n - 1) % 9)

Phi24(n) == n % 24
Phi9(n)  == DR(n)
QDef(bv, ev, dv, av) == 24 * Phi9(av) + Phi24(av)

TupleClosed(bv, ev, dv, av) ==
  /\ bv \in Nat /\ ev \in Nat /\ dv \in Nat /\ av \in Nat
  /\ dv = bv + ev
  /\ av = dv + ev

(*** Init: a legal starting state (b=1, e=1 => d=2, a=3), satisfies
     TupleClosed at the initial state. ***)

Init ==
  /\ b = 1
  /\ e = 1
  /\ d = 2
  /\ a = 3
  /\ qtag = QDef(1, 1, 2, 3)
  /\ fail = "OK"
  /\ lastMove = "NONE"

(*** BROKEN ACTION: writes d' = 5 while b' + e' = 4. Violates the tuple-
     closure rule dv = bv + ev. The Inv_TupleClosed invariant from
     QARM_v02_Failures must fire on the resulting state. ***)

BrokenTupleClosure ==
  /\ b' = 2
  /\ e' = 2
  /\ d' = 5        \* BAD: should be b' + e' = 4
  /\ a' = 7        \* BAD: should be d' + e' = 4 (or 7 if d'=5, but d' itself is wrong)
  /\ qtag' = qtag
  /\ fail' = "OK"
  /\ lastMove' = "σ"

Next == BrokenTupleClosure

Spec == Init /\ [][Next]_vars

(*** The invariant under test. Identical declaration to QARM_v02_Failures.
     If this test passes (no violation), the invariant is broken or vacuous.
     If it fails (violation found), the invariant is real and the main
     spec's Run 1 result is valid. ***)

Inv_TupleClosed == TupleClosed(b, e, d, a)

================================================================================
