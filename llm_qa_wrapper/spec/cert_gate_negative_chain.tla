---------------------- MODULE cert_gate_negative_chain -----------------------
(***************************************************************************)
(* NEGATIVE TEST for Inv_LedgerChainValid.                                  *)
(*                                                                           *)
(* Creates a ledger with a broken hash chain (the second entry's prev is   *)
(* genesis instead of ledger[1].ch). The Inv_LedgerChainValid invariant    *)
(* must fire on the resulting state.                                        *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Agents, Tools, Payloads, MaxLedgerDepth

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

(* Broken: build a two-entry ledger where entry 2's prev is GENESIS instead *)
(* of ledger[1].ch. Violates Inv_LedgerChainValid.                          *)
BrokenChain(a1, a2, t, p) ==
    /\ ledger = <<>>
    /\ LET rec1 == [agent |-> a1, tool |-> t, payload |-> p,
                    ch |-> Hash(a1, t, p, GENESIS, 0),
                    prev |-> GENESIS, ctr |-> 0]
           rec2 == [agent |-> a2, tool |-> t, payload |-> p,
                    ch |-> Hash(a2, t, p, GENESIS, 1),
                    prev |-> GENESIS, ctr |-> 1]  \* BAD: prev should be rec1.ch
       IN  ledger' = <<rec1, rec2>>
    /\ UNCHANGED <<pending, certs, executed, denied>>

Next == \E a1 \in Agents, a2 \in Agents, t \in Tools, p \in Payloads :
          BrokenChain(a1, a2, t, p)

Spec == Init /\ [][Next]_vars

Inv_LedgerChainValid ==
    \A i \in 1..Len(ledger) :
        ledger[i].prev = IF i = 1 THEN GENESIS ELSE ledger[i-1].ch

================================================================================
