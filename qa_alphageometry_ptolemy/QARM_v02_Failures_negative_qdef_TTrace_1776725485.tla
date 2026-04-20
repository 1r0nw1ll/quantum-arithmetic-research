---- MODULE QARM_v02_Failures_negative_qdef_TTrace_1776725485 ----
EXTENDS Sequences, TLCExt, QARM_v02_Failures_negative_qdef, Toolbox, Naturals, TLC

_expression ==
    LET QARM_v02_Failures_negative_qdef_TEExpression == INSTANCE QARM_v02_Failures_negative_qdef_TEExpression
    IN QARM_v02_Failures_negative_qdef_TEExpression!expression
----

_trace ==
    LET QARM_v02_Failures_negative_qdef_TETrace == INSTANCE QARM_v02_Failures_negative_qdef_TETrace
    IN QARM_v02_Failures_negative_qdef_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        a = (6)
        /\
        fail = ("OK")
        /\
        b = (2)
        /\
        d = (4)
        /\
        e = (2)
        /\
        lastMove = ("σ")
        /\
        qtag = (9999)
    )
----

_init ==
    /\ lastMove = _TETrace[1].lastMove
    /\ a = _TETrace[1].a
    /\ b = _TETrace[1].b
    /\ d = _TETrace[1].d
    /\ e = _TETrace[1].e
    /\ qtag = _TETrace[1].qtag
    /\ fail = _TETrace[1].fail
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ lastMove  = _TETrace[i].lastMove
        /\ lastMove' = _TETrace[j].lastMove
        /\ a  = _TETrace[i].a
        /\ a' = _TETrace[j].a
        /\ b  = _TETrace[i].b
        /\ b' = _TETrace[j].b
        /\ d  = _TETrace[i].d
        /\ d' = _TETrace[j].d
        /\ e  = _TETrace[i].e
        /\ e' = _TETrace[j].e
        /\ qtag  = _TETrace[i].qtag
        /\ qtag' = _TETrace[j].qtag
        /\ fail  = _TETrace[i].fail
        /\ fail' = _TETrace[j].fail

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("QARM_v02_Failures_negative_qdef_TTrace_1776725485.json", _TETrace)

=============================================================================

 Note that you can extract this module `QARM_v02_Failures_negative_qdef_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `QARM_v02_Failures_negative_qdef_TEExpression.tla` file takes precedence 
  over the module `QARM_v02_Failures_negative_qdef_TEExpression` below).

---- MODULE QARM_v02_Failures_negative_qdef_TEExpression ----
EXTENDS Sequences, TLCExt, QARM_v02_Failures_negative_qdef, Toolbox, Naturals, TLC

expression == 
    [
        \* To hide variables of the `QARM_v02_Failures_negative_qdef` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        lastMove |-> lastMove
        ,a |-> a
        ,b |-> b
        ,d |-> d
        ,e |-> e
        ,qtag |-> qtag
        ,fail |-> fail
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_lastMoveUnchanged |-> lastMove = lastMove'
        
        \* Format the `lastMove` variable as Json value.
        \* ,_lastMoveJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(lastMove)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_lastMoveModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].lastMove # _TETrace[s-1].lastMove
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE QARM_v02_Failures_negative_qdef_TETrace ----
\*EXTENDS IOUtils, QARM_v02_Failures_negative_qdef, TLC
\*
\*trace == IODeserialize("QARM_v02_Failures_negative_qdef_TTrace_1776725485.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE QARM_v02_Failures_negative_qdef_TETrace ----
EXTENDS QARM_v02_Failures_negative_qdef, TLC

trace == 
    <<
    ([a |-> 3,fail |-> "OK",b |-> 1,d |-> 2,e |-> 1,lastMove |-> "NONE",qtag |-> 75]),
    ([a |-> 6,fail |-> "OK",b |-> 2,d |-> 4,e |-> 2,lastMove |-> "σ",qtag |-> 9999])
    >>
----


=============================================================================

---- CONFIG QARM_v02_Failures_negative_qdef_TTrace_1776725485 ----

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Mon Apr 20 18:51:26 EDT 2026