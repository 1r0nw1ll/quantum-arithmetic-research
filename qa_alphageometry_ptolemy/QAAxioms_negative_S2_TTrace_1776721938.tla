---- MODULE QAAxioms_negative_S2_TTrace_1776721938 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, QAAxioms_negative_S2

_expression ==
    LET QAAxioms_negative_S2_TEExpression == INSTANCE QAAxioms_negative_S2_TEExpression
    IN QAAxioms_negative_S2_TEExpression!expression
----

_trace ==
    LET QAAxioms_negative_S2_TETrace == INSTANCE QAAxioms_negative_S2_TETrace
    IN QAAxioms_negative_S2_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        a = (3)
        /\
        fail = ("OK")
        /\
        b = ("ghost")
        /\
        d = (2)
        /\
        obs_float = (0)
        /\
        e = (1)
        /\
        lastMove = ("σ")
        /\
        obs_cross_count = (1)
        /\
        qtag = (75)
    )
----

_init ==
    /\ lastMove = _TETrace[1].lastMove
    /\ obs_float = _TETrace[1].obs_float
    /\ a = _TETrace[1].a
    /\ b = _TETrace[1].b
    /\ d = _TETrace[1].d
    /\ e = _TETrace[1].e
    /\ obs_cross_count = _TETrace[1].obs_cross_count
    /\ qtag = _TETrace[1].qtag
    /\ fail = _TETrace[1].fail
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ lastMove  = _TETrace[i].lastMove
        /\ lastMove' = _TETrace[j].lastMove
        /\ obs_float  = _TETrace[i].obs_float
        /\ obs_float' = _TETrace[j].obs_float
        /\ a  = _TETrace[i].a
        /\ a' = _TETrace[j].a
        /\ b  = _TETrace[i].b
        /\ b' = _TETrace[j].b
        /\ d  = _TETrace[i].d
        /\ d' = _TETrace[j].d
        /\ e  = _TETrace[i].e
        /\ e' = _TETrace[j].e
        /\ obs_cross_count  = _TETrace[i].obs_cross_count
        /\ obs_cross_count' = _TETrace[j].obs_cross_count
        /\ qtag  = _TETrace[i].qtag
        /\ qtag' = _TETrace[j].qtag
        /\ fail  = _TETrace[i].fail
        /\ fail' = _TETrace[j].fail

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("QAAxioms_negative_S2_TTrace_1776721938.json", _TETrace)

=============================================================================

 Note that you can extract this module `QAAxioms_negative_S2_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `QAAxioms_negative_S2_TEExpression.tla` file takes precedence 
  over the module `QAAxioms_negative_S2_TEExpression` below).

---- MODULE QAAxioms_negative_S2_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, QAAxioms_negative_S2

expression == 
    [
        \* To hide variables of the `QAAxioms_negative_S2` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        lastMove |-> lastMove
        ,obs_float |-> obs_float
        ,a |-> a
        ,b |-> b
        ,d |-> d
        ,e |-> e
        ,obs_cross_count |-> obs_cross_count
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
\*---- MODULE QAAxioms_negative_S2_TETrace ----
\*EXTENDS IOUtils, TLC, QAAxioms_negative_S2
\*
\*trace == IODeserialize("QAAxioms_negative_S2_TTrace_1776721938.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE QAAxioms_negative_S2_TETrace ----
EXTENDS TLC, QAAxioms_negative_S2

trace == 
    <<
    ([a |-> 3,fail |-> "OK",b |-> 1,d |-> 2,obs_float |-> 0,e |-> 1,lastMove |-> "NONE",obs_cross_count |-> 1,qtag |-> 75]),
    ([a |-> 3,fail |-> "OK",b |-> "ghost",d |-> 2,obs_float |-> 0,e |-> 1,lastMove |-> "σ",obs_cross_count |-> 1,qtag |-> 75])
    >>
----


=============================================================================

---- CONFIG QAAxioms_negative_S2_TTrace_1776721938 ----

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
\* Generated on Mon Apr 20 17:52:20 EDT 2026