------------------------- MODULE QARM_v02_NoMu_Stats -------------------------
EXTENDS Naturals, Integers, Sequences, TLC, TLCExt

(*
  QARM_v02_NoMu_Stats.tla

  Variant of QARM_v02_Stats with μ REMOVED from Next.
  Used to answer the 2025 ChatGPT question verbatim:

    "verify that the number of states with fail = OUT_OF_BOUNDS is
     invariant across different generator sets"

  Identical PrintT instrumentation to QARM_v02_Stats so the
  per-action failure tally is directly comparable.
*)

CONSTANTS CAP, KSet

VARIABLES b, e, d, a, qtag, fail, lastMove

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

(*** Next WITHOUT μ ***)

Next ==
  \/ Sigma
  \/ Lambda
  \/ /\ fail # "OK" /\ UNCHANGED <<b,e,d,a,qtag,fail,lastMove>>

Spec == Init /\ [][Next]_<<b,e,d,a,qtag,fail,lastMove>>

Inv_TupleClosed == TupleClosed(b,e,d,a)
Inv_InBounds    == InBounds(b,e,d,a)
Inv_QDef        == qtag = QDef(b,e,d,a)

===============================================================================
