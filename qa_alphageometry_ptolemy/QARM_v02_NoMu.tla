------------------------------ MODULE QARM_v02_NoMu ------------------------------
EXTENDS Naturals, Integers, TLC

(*
  QARM_v02_NoMu.tla

  Variant of QARM_v02_Failures with μ REMOVED from Next.
  Used to test whether failure counts remain invariant under generator set changes.
*)

CONSTANTS
  CAP,
  KSet

VARIABLES
  b, e, d, a,
  qtag,
  fail,
  lastMove

(*** Helpers ***)

InCap(x) == x \in 0..CAP

DR(n) ==
  IF n = 0 THEN 0 ELSE 1 + ((n - 1) % 9)

Phi24(n) == n % 24
Phi9(n)  == DR(n)

QDef(bv, ev, dv, av) == 24 * Phi9(av) + Phi24(av)

TupleClosed(bv, ev, dv, av) ==
  /\ bv \in Nat /\ ev \in Nat /\ dv \in Nat /\ av \in Nat
  /\ dv = bv + ev
  /\ av = dv + ev

InBounds(bv, ev, dv, av) ==
  /\ InCap(bv) /\ InCap(ev) /\ InCap(dv) /\ InCap(av)

StateOK ==
  /\ TupleClosed(b, e, d, a)
  /\ InBounds(b, e, d, a)
  /\ qtag = QDef(b, e, d, a)
  /\ fail \in {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}
  /\ lastMove \in {"NONE","σ","μ","λ"}

(*** Init ***)

Init ==
  /\ b \in 0..CAP
  /\ e \in 0..CAP
  /\ d = b + e
  /\ a = d + e
  /\ TupleClosed(b, e, d, a)
  /\ InBounds(b, e, d, a)
  /\ qtag = QDef(b, e, d, a)
  /\ fail = "OK"
  /\ lastMove = "NONE"

(*** σ ***)

SigmaSucc ==
  LET e2 == e + 1 IN
  LET b2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ InBounds(b2, e2, d2, a2)
  /\ QDef(b2, e2, d2, a2) = qtag
  /\ b' = b2 /\ e' = e2 /\ d' = d2 /\ a' = a2
  /\ qtag' = qtag
  /\ fail' = "OK"
  /\ lastMove' = "σ"

SigmaFail_OUT_OF_BOUNDS ==
  LET e2 == e + 1 IN
  LET b2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ ~InBounds(b2, e2, d2, a2)
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "OUT_OF_BOUNDS"
  /\ lastMove' = "σ"

SigmaFail_FIXED_Q ==
  LET e2 == e + 1 IN
  LET b2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ InBounds(b2, e2, d2, a2)
  /\ QDef(b2, e2, d2, a2) # qtag
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "FIXED_Q_VIOLATION"
  /\ lastMove' = "σ"

Sigma ==
  SigmaSucc \/ SigmaFail_OUT_OF_BOUNDS \/ SigmaFail_FIXED_Q

(*** λ_k ***)

LambdaSucc ==
  \E k \in KSet :
    LET b2 == k * b IN
    LET e2 == k * e IN
    LET d2 == b2 + e2 IN
    LET a2 == d2 + e2 IN
    /\ fail = "OK"
    /\ InBounds(b2, e2, d2, a2)
    /\ QDef(b2, e2, d2, a2) = qtag
    /\ b' = b2 /\ e' = e2 /\ d' = d2 /\ a' = a2
    /\ qtag' = qtag
    /\ fail' = "OK"
    /\ lastMove' = "λ"

LambdaFail_OUT_OF_BOUNDS ==
  \E k \in KSet :
    LET b2 == k * b IN
    LET e2 == k * e IN
    LET d2 == b2 + e2 IN
    LET a2 == d2 + e2 IN
    /\ fail = "OK"
    /\ ~InBounds(b2, e2, d2, a2)
    /\ UNCHANGED <<b,e,d,a,qtag>>
    /\ fail' = "OUT_OF_BOUNDS"
    /\ lastMove' = "λ"

LambdaFail_FIXED_Q ==
  \E k \in KSet :
    LET b2 == k * b IN
    LET e2 == k * e IN
    LET d2 == b2 + e2 IN
    LET a2 == d2 + e2 IN
    /\ fail = "OK"
    /\ InBounds(b2, e2, d2, a2)
    /\ QDef(b2, e2, d2, a2) # qtag
    /\ UNCHANGED <<b,e,d,a,qtag>>
    /\ fail' = "FIXED_Q_VIOLATION"
    /\ lastMove' = "λ"

Lambda ==
  LambdaSucc \/ LambdaFail_OUT_OF_BOUNDS \/ LambdaFail_FIXED_Q

(*** Next WITHOUT μ ***)

Next ==
  \/ Sigma
  \/ Lambda
  \/ /\ fail # "OK" /\ UNCHANGED <<b,e,d,a,qtag,fail,lastMove>>

Spec ==
  Init /\ [][Next]_<<b,e,d,a,qtag,fail,lastMove>>

(*** Invariants ***)

Inv_TupleClosed == TupleClosed(b,e,d,a)
Inv_InBounds    == InBounds(b,e,d,a)
Inv_QDef        == qtag = QDef(b,e,d,a)
Inv_FailDomain  == fail \in {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}
Inv_MoveDomain  == lastMove \in {"NONE","σ","λ"}  \* NOTE: "μ" removed from domain

THEOREM Spec => []Inv_TupleClosed
THEOREM Spec => []Inv_InBounds
THEOREM Spec => []Inv_QDef
THEOREM Spec => []Inv_FailDomain
THEOREM Spec => []Inv_MoveDomain

===============================================================================
