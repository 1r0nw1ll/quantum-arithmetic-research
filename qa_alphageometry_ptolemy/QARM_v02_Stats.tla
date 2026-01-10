------------------------------ MODULE QARM_v02_Stats ------------------------------
EXTENDS Naturals, Integers, TLC, TLCExt

(*
  QARM_v02_Stats.tla

  Variant with state statistics collection.
  Uses TLC!Print to output failure state counts.
*)

CONSTANTS CAP, KSet

VARIABLES b, e, d, a, qtag, fail, lastMove

(*** Helpers ***)

InCap(x) == x \in 0..CAP
DR(n) == IF n = 0 THEN 0 ELSE 1 + ((n - 1) % 9)
Phi24(n) == n % 24
Phi9(n)  == DR(n)
QDef(bv, ev, dv, av) == 24 * Phi9(av) + Phi24(av)

TupleClosed(bv, ev, dv, av) ==
  /\ bv \in Nat /\ ev \in Nat /\ dv \in Nat /\ av \in Nat
  /\ dv = bv + ev
  /\ av = dv + ev

InBounds(bv, ev, dv, av) ==
  /\ InCap(bv) /\ InCap(ev) /\ InCap(dv) /\ InCap(av)

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

(*** Actions ***)

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

SigmaFail_OOB ==
  LET e2 == e + 1 IN
  LET b2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ ~InBounds(b2, e2, d2, a2)
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "OUT_OF_BOUNDS"
  /\ lastMove' = "σ"
  /\ PrintT("FAIL_OOB_SIGMA: " \o ToString(<<b,e,d,a,qtag>>))

SigmaFail_FQ ==
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
  /\ PrintT("FAIL_FQ_SIGMA: " \o ToString(<<b,e,d,a,qtag>>))

Sigma == SigmaSucc \/ SigmaFail_OOB \/ SigmaFail_FQ

MuSucc ==
  LET b2 == e IN
  LET e2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ InBounds(b2, e2, d2, a2)
  /\ QDef(b2, e2, d2, a2) = qtag
  /\ b' = b2 /\ e' = e2 /\ d' = d2 /\ a' = a2
  /\ qtag' = qtag
  /\ fail' = "OK"
  /\ lastMove' = "μ"

MuFail_OOB ==
  LET b2 == e IN
  LET e2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ ~InBounds(b2, e2, d2, a2)
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "OUT_OF_BOUNDS"
  /\ lastMove' = "μ"
  /\ PrintT("FAIL_OOB_MU: " \o ToString(<<b,e,d,a,qtag>>))

MuFail_FQ ==
  LET b2 == e IN
  LET e2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ InBounds(b2, e2, d2, a2)
  /\ QDef(b2, e2, d2, a2) # qtag
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "FIXED_Q_VIOLATION"
  /\ lastMove' = "μ"
  /\ PrintT("FAIL_FQ_MU: " \o ToString(<<b,e,d,a,qtag>>))

Mu == MuSucc \/ MuFail_OOB \/ MuFail_FQ

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

LambdaFail_OOB ==
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
    /\ PrintT("FAIL_OOB_LAMBDA: " \o ToString(<<b,e,d,a,qtag,k>>))

LambdaFail_FQ ==
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
    /\ PrintT("FAIL_FQ_LAMBDA: " \o ToString(<<b,e,d,a,qtag,k>>))

Lambda == LambdaSucc \/ LambdaFail_OOB \/ LambdaFail_FQ

Next ==
  \/ Sigma
  \/ Mu
  \/ Lambda
  \/ /\ fail # "OK" /\ UNCHANGED <<b,e,d,a,qtag,fail,lastMove>>

Spec == Init /\ [][Next]_<<b,e,d,a,qtag,fail,lastMove>>

Inv_TupleClosed == TupleClosed(b,e,d,a)
Inv_InBounds    == InBounds(b,e,d,a)
Inv_QDef        == qtag = QDef(b,e,d,a)

===============================================================================
