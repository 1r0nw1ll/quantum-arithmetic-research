--------------------- MODULE QARM_v02_Failures_negative_bounds ---------------------
EXTENDS Naturals, Integers, TLC

(*
  NEGATIVE TEST for Inv_InBounds (QARM_v02_Failures).

  Writes b' = 99 (far outside 0..CAP) directly, violating InBounds which
  requires all of b, e, d, a \in 0..CAP. Inv_InBounds must fire with a
  ≤2-state counterexample.

  Closes Lane 1 non-vacuity gap item 1/4 (per QARM_PROOF_LEDGER.md §"Scope
  limitations" #2). Pattern mirrors QARM_v02_Failures_negative.tla
  (Inv_TupleClosed negative, Lane 1 Run 2).
*)

VARIABLES b, e, d, a, qtag, fail, lastMove
vars == <<b, e, d, a, qtag, fail, lastMove>>

DR(n) == IF n = 0 THEN 0 ELSE 1 + ((n - 1) % 9)
Phi24(n) == n % 24
Phi9(n)  == DR(n)
QDef(bv, ev, dv, av) == 24 * Phi9(av) + Phi24(av)

Init ==
  /\ b = 1
  /\ e = 1
  /\ d = 2
  /\ a = 3
  /\ qtag = QDef(1, 1, 2, 3)
  /\ fail = "OK"
  /\ lastMove = "NONE"

\* BROKEN: b' = 99 — exceeds CAP = 20 from QARM_v02_Failures.cfg.
\* Note: this negative spec has no CONSTANTS (standalone); the violated
\* bound is 0..CAP from the production spec Inv_InBounds, simulated here
\* via the concrete InBounds check with hardwired CAP = 20.
CAP_TEST == 20
InBounds(bv, ev, dv, av) ==
  /\ bv \in 0..CAP_TEST /\ ev \in 0..CAP_TEST
  /\ dv \in 0..CAP_TEST /\ av \in 0..CAP_TEST

BrokenInBounds ==
  /\ b' = 99        \* BAD: 99 \notin 0..20
  /\ e' = 1
  /\ d' = 100       \* derived d = 99 + 1
  /\ a' = 101
  /\ qtag' = qtag
  /\ fail' = "OK"
  /\ lastMove' = "σ"

Next == BrokenInBounds
Spec == Init /\ [][Next]_vars

Inv_InBounds == InBounds(b, e, d, a)

================================================================================
