# TwoPhase: A Two-Phase Commit Protocol

## What is modeled
This artifact models a two-phase commit protocol with a coordinator and
multiple resource managers. The protocol uses a prepare-phase followed by a
commit-phase, and guarantees atomicity across all participants.

## Variable/action justification
- `round`: the protocol round number; advances after each phase.
- `Advance`: advances the protocol to the next phase.

## Intrinsic semantics
- `round` tracks the two-phase protocol state across participants.
- `Advance` represents a coordinator-driven phase transition.

## TLC bounds
- No bounds are required; the model is finite by termination.

## Source grounding
- Derived from the Gray–Lamport consensus-protocol description.
- Grounded in the visible task description of a distributed commit.

## Repository fit
This aims at tlaplus/examples. Comparable examples:
- similar to Paxos.tla for consensus structure.
