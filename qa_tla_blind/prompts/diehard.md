# DieHard — Blind Reproduction Prompt

## Source
Spec: `tlaplus/Examples/specifications/DieHard/DieHard.tla` (hidden from reproducing session)

## Problem statement (English only)
From the movie Die Hard 3: the heroes must obtain exactly 4 gallons of water using a 5-gallon jug, a 3-gallon jug, and a water faucet. The jugs are uncalibrated (no volume markings between empty and full). Starting from both jugs empty, the heroes can fill a jug from the faucet, empty a jug onto the ground, or pour water from one jug into the other. The goal is to reach a state in which exactly 4 gallons of water sit in one of the jugs.

## State variables (names only)
- `big` — the current amount of water (in gallons) held in the 5-gallon jug.
- `small` — the current amount of water (in gallons) held in the 3-gallon jug.

## Named properties to reproduce
The reproducing session must write a QA-native form for each of these:

- `TypeOK` — a type/range invariant asserting that each jug's contents stays within its physical capacity (non-negative, not exceeding the jug's size).
- `NotSolved` — a predicate asserting that the puzzle has not yet been solved; a state that violates this invariant is a witness trajectory to the goal (the puzzle's actual solution). The model checker is run expecting this invariant to fail, and the counterexample IS the solution.

## What is explicitly withheld
- Init predicate body
- Next-state action body (the six allowed moves and their algebraic effects on `big`, `small`)
- Invariant predicate bodies (exact TypeOK range expressions; exact NotSolved target expression)
- Any proof or model-checker counterexample (the solving trajectory)
- Any bounds/constants beyond what the English problem statement implies (jug capacities 3 and 5, target 4 are given; anything else is withheld)

## Notes for the reproducing session
- The spec is finite-state: the reachable state space is small (bounded by jug capacities), so exhaustive search is feasible.
- There are six discrete action types available at each step (fill / empty / pour, across the two jugs). You will need to enumerate them from the physical problem description — do not assume the TLA+ spec's exact decomposition.
- Pouring between uncalibrated jugs is "all-or-nothing at a boundary": a pour stops either when the source is empty or the destination is full. Partial intermediate volumes cannot be achieved by a single pour.
- The puzzle has a known short solution; do not attempt to guess its length or trajectory — derive it by QA-native search.
- Time in this spec is discrete action steps (path length k), which aligns with QA axiom T1.
- Both state variables are non-negative integers at every reachable state; this should align naturally with QA's No-Zero handling (note: physical zero-fill IS a valid puzzle state, so your encoding must be honest about how it represents "empty jug" — this is a mapping-protocol decision, not a free choice).
