---------------------- MODULE cert_gate_negative_bind ------------------------
(***************************************************************************)
(* NEGATIVE TEST for Inv_CertBindsPayload.                                  *)
(*                                                                           *)
(* Creates a cert whose `ch` field does NOT equal the Hash of its other    *)
(* fields. The Inv_CertBindsPayload invariant must fire.                    *)
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

(* Broken: build a cert whose ch field is GENESIS instead of the computed *)
(* Hash of its other fields. Violates Inv_CertBindsPayload.                *)
BrokenBinding(a, t, p) ==
    /\ certs = {}
    /\ LET badCert == [agent |-> a, tool |-> t, payload |-> p,
                       ch |-> GENESIS,  \* BAD: should be Hash(a,t,p,...)
                       prev |-> GENESIS, ctr |-> 0]
       IN  certs' = {badCert}
    /\ UNCHANGED <<pending, executed, ledger, denied>>

Next == \E a \in Agents, t \in Tools, p \in Payloads : BrokenBinding(a, t, p)

Spec == Init /\ [][Next]_vars

Inv_CertBindsPayload ==
    \A cert \in certs :
        cert.ch = Hash(cert.agent, cert.tool, cert.payload, cert.prev, cert.ctr)

================================================================================
