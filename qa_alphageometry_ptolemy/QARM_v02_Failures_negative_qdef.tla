--------------------- MODULE QARM_v02_Failures_negative_qdef ---------------------
EXTENDS Naturals, Integers, TLC

(*
  NEGATIVE TEST for Inv_QDef (QARM_v02_Failures).

  Writes qtag' to a value that does NOT equal QDef(b', e', d', a'),
  violating the duo-modular packing invariant qtag = 24 * Phi9(a) + Phi24(a).
  Inv_QDef must fire with a ≤2-state counterexample.

  Closes Lane 1 non-vacuity gap item 2/4. Pattern mirrors
  QARM_v02_Failures_negative.tla (Lane 1 Run 2).
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

\* BROKEN: qtag' = 9999, unrelated to QDef of the tuple. Tuple itself
\* stays well-formed; only the duo-modular tag is corrupted.
BrokenQDef ==
  /\ b' = 2
  /\ e' = 2
  /\ d' = 4
  /\ a' = 6
  /\ qtag' = 9999    \* BAD: should be QDef(2,2,4,6) = 24*6 + 6 = 150
  /\ fail' = "OK"
  /\ lastMove' = "σ"

Next == BrokenQDef
Spec == Init /\ [][Next]_vars

Inv_QDef == qtag = QDef(b, e, d, a)

================================================================================
