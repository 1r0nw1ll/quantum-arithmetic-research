------------------- MODULE cert_gate_negative_composition -------------------
(***************************************************************************)
(* NEGATIVE TEST for Inv_Composition.                                       *)
(*                                                                           *)
(* Creates a ledger entry that has no matching cert in `certs`, OR whose  *)
(* hash is not in `executed`. Violates Inv_Composition.                   *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Agents, Tools, Payloads

VARIABLES pending, certs, executed, ledger, denied
vars == <<pending, certs, executed, ledger, denied>>

Hash(agent, tool, payload, prev, ctr) ==
    [tag |-> "h", a |-> agent, t |-> tool, p |-> payload, pr |-> prev, c |-> ctr]

GENESIS == [tag |-> "g"]

Init ==
    /\ pending  = {}
    /\ certs    = {}
    /\ executed = {}
    /\ ledger   = <<>>
    /\ denied   = {}

(* Broken: append a cert to the ledger without adding it to `executed`.   *)
(* Violates Inv_Composition.                                                *)
BrokenComposition(a, t, p) ==
    /\ ledger = <<>>
    /\ LET rec == [agent |-> a, tool |-> t, payload |-> p,
                   ch |-> Hash(a, t, p, GENESIS, 0),
                   prev |-> GENESIS, ctr |-> 0]
       IN  /\ certs' = certs \cup {rec}
           /\ ledger' = <<rec>>  \* BAD: no corresponding entry in executed
    /\ UNCHANGED <<pending, executed, denied>>

Next == \E a \in Agents, t \in Tools, p \in Payloads : BrokenComposition(a, t, p)

Spec == Init /\ [][Next]_vars

Inv_Composition ==
    \A i \in 1..Len(ledger) :
        /\ \E cert \in certs : cert = ledger[i]
        /\ ledger[i].ch \in executed

================================================================================
