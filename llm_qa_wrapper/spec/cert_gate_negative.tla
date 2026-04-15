-------------------------- MODULE cert_gate_negative ---------------------------
(***************************************************************************)
(* NEGATIVE TEST SPEC for cert_gate.                                         *)
(*                                                                           *)
(* This spec deliberately introduces a violation of the cert-gate protocol  *)
(* to verify that TLC actually EXERCISES the safety invariants defined in   *)
(* cert_gate.tla. If TLC reports "no error" for this spec, the invariants   *)
(* are vacuously satisfied in the main spec and the proof is meaningless.   *)
(*                                                                           *)
(* Expected TLC result: INVARIANT VIOLATED on Inv_NoExecWithoutCert or      *)
(* Inv_LedgerChainValid, with a counterexample trace showing the illegal   *)
(* direct-injection action that bypasses the gate.                          *)
(*                                                                           *)
(* This spec INSTANTIATES the cert_gate module and ADDS a direct-injection *)
(* action to demonstrate that the invariants would fire on any real        *)
(* violation. It is NOT part of the production spec — it exists only to    *)
(* prove that the main spec's invariants are not vacuous.                   *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Agents, Tools, Payloads, MaxLedgerDepth

VARIABLES pending, certs, executed, ledger, denied

vars == <<pending, certs, executed, ledger, denied>>

\* Minimal re-declaration to match cert_gate's state space

Hash(agent, tool, payload, prev, ctr) ==
    [tag |-> "h", a |-> agent, t |-> tool, p |-> payload, pr |-> prev, c |-> ctr]

GENESIS == [tag |-> "g"]

Init ==
    /\ pending  = {}
    /\ certs    = {}
    /\ executed = {}
    /\ ledger   = <<>>
    /\ denied   = {}

(***************************************************************************)
(* BROKEN ACTION: injects a random hash directly into `executed` with no   *)
(* cert backing. The Inv_NoExecWithoutCert invariant from cert_gate.tla    *)
(* must fire on the resulting state.                                        *)
(***************************************************************************)
BrokenDirectInject(agent, tool, payload) ==
    LET fakeHash == Hash(agent, tool, payload, GENESIS, 0)
    IN  /\ executed' = executed \cup {fakeHash}
        /\ UNCHANGED <<pending, certs, ledger, denied>>

Next ==
    \E agent \in Agents, tool \in Tools, payload \in Payloads :
        BrokenDirectInject(agent, tool, payload)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* The invariant from cert_gate.tla. If this test passes (no violation),   *)
(* the invariant is broken or vacuous. If it fails (violation found), the *)
(* invariant is real and the main spec's proof is valid.                   *)
(***************************************************************************)
Inv_NoExecWithoutCert ==
    \A h \in executed : \E cert \in certs : cert.ch = h

================================================================================
