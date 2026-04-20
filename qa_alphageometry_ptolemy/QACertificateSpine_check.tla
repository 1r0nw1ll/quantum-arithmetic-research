----------------------- MODULE QACertificateSpine_check -----------------------
EXTENDS QACertificateSpine

(*
  QACertificateSpine_check.tla — model-check wrapper for QACertificateSpine.

  QACertificateSpine.tla (authored 2026-01-21, 13 KB, 111 days dormant)
  declares seven certificate record types, their validators, and the
  `FailureFirstClass` theorem:

      THEOREM FailureFirstClass ==
          \A cert \in Certificate:
              NoSilentFailures(cert) /\ ObstructionHasWitness(cert)

  The module has no VARIABLES / Init / Next, so it cannot be TLC-checked
  directly. This wrapper:
    1. Defines a bounded witness alphabet so Seq(STRING) becomes finite.
    2. Instantiates `cert` as a single state variable drawn from a bounded
       subset of Certificate.
    3. Checks the two invariants used in FailureFirstClass as state
       predicates.

  If TLC finds a reachable cert state violating Inv_NoSilentFailures or
  Inv_ObstructionHasWitness, the theorem FailureFirstClass is false
  relative to the Certificate type as defined — which points to a
  design inconsistency between the CertificateStatus domain (which
  INCLUDES "INVALID") and NoSilentFailures (which REJECTS "INVALID").

  Session: cert-qa-axioms-tla-followup (2026-04-20, Lane 2 follow-up
  task (2) per QARM_PROOF_LEDGER.md §"Next steps (post-Lane 2)").
*)

\* Bounded witness domain for TLC tractability. Real witness strings are
\* opaque; TLC only needs enough variety to exercise empty / non-empty cases.
WitnessStrings == {"ok", "w1", "w2"}
BoundedWitnessSeqs == {<<>>, <<"ok">>, <<"w1">>, <<"w1", "w2">>}

BoundedCertificate == [
    status: CertificateStatus,
    witness: BoundedWitnessSeqs,
    verifiable: BOOLEAN
]

VARIABLES cert

Init == cert \in BoundedCertificate

\* No state transitions — this is a point-set model-check. cert takes a
\* value at Init and never changes; TLC will explore every element of
\* BoundedCertificate as an initial state.
Next == UNCHANGED cert

Spec == Init /\ [][Next]_cert

\* Invariants asserted by the FailureFirstClass theorem, each named
\* individually so TLC can report which fires first on a counterexample.

Inv_NoSilentFailures       == NoSilentFailures(cert)
Inv_ObstructionHasWitness  == ObstructionHasWitness(cert)

===============================================================================
