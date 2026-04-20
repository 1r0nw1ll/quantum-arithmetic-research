--------------------- MODULE QARM_v02_Failures_negative_movedomain ---------------------
EXTENDS Naturals, Integers, TLC

(*
  NEGATIVE TEST for Inv_MoveDomain (QARM_v02_Failures).

  Writes lastMove' to a value OUTSIDE the finite generator alphabet
  {"NONE","σ","μ","λ"}. Inv_MoveDomain must fire with a ≤2-state
  counterexample.

  Closes Lane 1 non-vacuity gap item 4/4 — completes the ledger's
  5-of-5 invariant non-vacuity coverage.
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

\* BROKEN: lastMove' = "γ" — γ is NOT a declared QARM generator (only σ/μ/λ).
BrokenMoveDomain ==
  /\ b' = 1
  /\ e' = 2
  /\ d' = 3
  /\ a' = 5
  /\ qtag' = qtag
  /\ fail' = "OK"
  /\ lastMove' = "γ"   \* BAD: γ outside {NONE, σ, μ, λ}

Next == BrokenMoveDomain
Spec == Init /\ [][Next]_vars

Inv_MoveDomain == lastMove \in {"NONE","σ","μ","λ"}

================================================================================
