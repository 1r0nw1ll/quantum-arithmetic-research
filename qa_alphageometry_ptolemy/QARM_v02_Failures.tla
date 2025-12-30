------------------------------ MODULE QARM_v02_Failures ------------------------------
EXTENDS Naturals, Integers, TLC

(*
  QARM_v02_Failures.tla

  Minimal constitutional spec for QA/QARM with:
   - Canonical QA tuple constraints (d=b+e, a=d+e)
   - Duo-modular fixed-q: qtag = 24*phi9(a) + phi24(a) in [0,239]
   - Generator set Σ = {σ, μ, λ_k}
   - Failures are first-class states (do not vanish)

  TLC requirements:
   - CAP finite (e.g. 20..50)
   - KSet finite (e.g. {2,3})
*)

CONSTANTS
  CAP,          \* bound on b,e,d,a
  KSet          \* finite set of scaling factors for λ (e.g. {2,3})

VARIABLES
  b, e, d, a,
  qtag,
  fail,         \* {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}
  lastMove      \* {"NONE","σ","μ","λ"}

(*** Helpers ***)

InCap(x) == x \in 0..CAP

DR(n) ==
  \* Digital root, with DR(0)=0 for convenience
  IF n = 0 THEN 0 ELSE 1 + ((n - 1) % 9)

Phi24(n) == n % 24
Phi9(n)  == DR(n)

\* Duo-modular packing: qtag = 24*phi9 + phi24
\* phi9 in 0..9, phi24 in 0..23 => qtag in 0..239
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

(*** Move attempt semantics ****************************************************
  Each move has:
    - Success action (updates tuple, preserves qtag)
    - Fail_OUT_OF_BOUNDS (if successor would exceed CAP)
    - Fail_FIXED_Q (if successor would change duo-modular qtag)

  Important: Fail actions leave the state unchanged except fail/lastMove.
  This matches "failure as algebraic object" (logging an attempted move).
*******************************************************************************)

(*** σ : growth on e by +1; close tuple canonically ***)

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


(*** μ : swap b <-> e; close tuple canonically ***)

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

MuFail_OUT_OF_BOUNDS ==
  LET b2 == e IN
  LET e2 == b IN
  LET d2 == b2 + e2 IN
  LET a2 == d2 + e2 IN
  /\ fail = "OK"
  /\ ~InBounds(b2, e2, d2, a2)
  /\ UNCHANGED <<b,e,d,a,qtag>>
  /\ fail' = "OUT_OF_BOUNDS"
  /\ lastMove' = "μ"

MuFail_FIXED_Q ==
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

Mu ==
  MuSucc \/ MuFail_OUT_OF_BOUNDS \/ MuFail_FIXED_Q


(*** λ_k : scale (b,e) by k; close tuple canonically ***)

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


(*** Global Next ***************************************************************)

Next ==
  \/ Sigma
  \/ Mu
  \/ Lambda
  \/ /\ fail # "OK" /\ UNCHANGED <<b,e,d,a,qtag,fail,lastMove>>
     \* absorbing stutter once failure recorded (keeps failure states distinct)

Spec ==
  Init /\ [][Next]_<<b,e,d,a,qtag,fail,lastMove>>

(*** Invariants ****************************************************************)

Inv_TupleClosed == TupleClosed(b,e,d,a)
Inv_InBounds    == InBounds(b,e,d,a)
Inv_QDef        == qtag = QDef(b,e,d,a)
Inv_FailDomain  == fail \in {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}
Inv_MoveDomain  == lastMove \in {"NONE","σ","μ","λ"}

THEOREM Spec => []Inv_TupleClosed
THEOREM Spec => []Inv_InBounds
THEOREM Spec => []Inv_QDef
THEOREM Spec => []Inv_FailDomain
THEOREM Spec => []Inv_MoveDomain

===============================================================================
