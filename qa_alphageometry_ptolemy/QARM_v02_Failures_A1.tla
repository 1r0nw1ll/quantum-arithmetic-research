------------------------------ MODULE QARM_v02_Failures_A1 ------------------------------
EXTENDS Naturals, Integers, TLC

(*
  QARM_v02_Failures_A1.tla — A1-corrected variant of QARM_v02_Failures.tla
  ------------------------------------------------------------------------
  Differs from QARM_v02_Failures ONLY in that the Init predicate constrains
  b, e \in 1..CAP (not 0..CAP). This enforces QA Axiom A1 (No-Zero): QA
  states live in {1..N}, never {0..N-1}.

  Authoring note (2026-04-20, session cert-qa-axioms-tla / Lane 2):
  the original QARM_v02_Failures.tla uses 0..CAP in Init as authored
  2025-12-30. That is an A1 GAP per CLAUDE.md "QA Axiom Compliance" and
  QARM_PROOF_LEDGER.md §"Scope limitations" item 3. Rather than silently
  mutating the 2025-12-30 spec, this file is authored as a variant so:
    * Lane 1 baseline (121 inits / 504 states, Run 1 of the ledger) remains
      comparable artifact-for-artifact to its dated counterparts.
    * Lane 2 (QAAxioms.tla) can EXTENDS this A1-compliant variant and
      assert Inv_A1_NoZero as a true-throughout-reachable-states invariant.

  All other logic — generator actions, failure classes, duo-modular qtag
  packing, absorbing-stutter semantics — is byte-identical to
  QARM_v02_Failures.tla.

  Primary references:
    - Lamport, L. (1994). The Temporal Logic of Actions. ACM TOPLAS 16(3).
    - Lamport, L. (2002). Specifying Systems. Addison-Wesley.
    - CLAUDE.md "QA Axiom Compliance" section for A1 text.

  TLC requirements:
   - CAP finite (e.g. 20..50)
   - KSet finite (e.g. {2,3})
*)

CONSTANTS
  CAP,          \* bound on b,e,d,a
  KSet          \* finite set of scaling factors for lambda (e.g. {2,3})

VARIABLES
  b, e, d, a,
  qtag,
  fail,         \* {"OK","OUT_OF_BOUNDS","FIXED_Q_VIOLATION","ILLEGAL"}
  lastMove      \* {"NONE","σ","μ","λ"}

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

(*** Init — A1-corrected: b, e \in 1..CAP (not 0..CAP) ***)

Init ==
  /\ b \in 1..CAP             \* A1: No-Zero. Was 0..CAP in QARM_v02_Failures.
  /\ e \in 1..CAP             \* A1: No-Zero. Was 0..CAP in QARM_v02_Failures.
  /\ d = b + e
  /\ a = d + e
  /\ TupleClosed(b, e, d, a)
  /\ InBounds(b, e, d, a)
  /\ qtag = QDef(b, e, d, a)
  /\ fail = "OK"
  /\ lastMove = "NONE"

(*** Move actions — byte-identical to QARM_v02_Failures from here down ***)

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
