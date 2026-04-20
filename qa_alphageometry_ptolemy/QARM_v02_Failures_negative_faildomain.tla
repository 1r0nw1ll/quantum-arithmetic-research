--------------------- MODULE QARM_v02_Failures_negative_faildomain ---------------------
EXTENDS Naturals, Integers, TLC

(*
  NEGATIVE TEST for Inv_FailDomain (QARM_v02_Failures).

  Writes fail' to a value OUTSIDE the finite fail alphabet
  {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}. Inv_FailDomain must
  fire with a ≤2-state counterexample.

  Closes Lane 1 non-vacuity gap item 3/4.
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

\* BROKEN: fail' = "PANIC" (not in the declared fail alphabet).
BrokenFailDomain ==
  /\ b' = 1
  /\ e' = 2
  /\ d' = 3
  /\ a' = 5
  /\ qtag' = qtag
  /\ fail' = "PANIC"    \* BAD: unknown fail class
  /\ lastMove' = "σ"

Next == BrokenFailDomain
Spec == Init /\ [][Next]_vars

Inv_FailDomain == fail \in {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}

================================================================================
