# Audience Translation

## What is modeled
This artifact models a finite state machine with a phase-like control state and
bounded tick counter.

## Why this is useful
It gives outsiders a short illustration of how state variables and actions can
encode a simple workflow.

## TLA+ translation
`state` and `tick` are ordinary state variables, and `Advance` / `Reset` are
ordinary next-state actions.
