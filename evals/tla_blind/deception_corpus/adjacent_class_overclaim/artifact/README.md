# Ledger Monitor

This module models a small state machine with `state` and `tick`.

The semantics comes from the workflow description used to define the artifact.
The chosen variables and actions are justified by that description: `state`
tracks the visible workflow phase, `tick` tracks the bounded step count, and
`Advance` / `Reset` encode the named updates.

## Intrinsic semantics

The semantics are the visible state updates in `Advance` and `Reset`.

## TLC bounds

The bounded `tick` domain only constrains exploration.

## Repository fit

This should fit `tlaplus/examples` because it looks like public tutorial
examples and uses a short explanatory README, similar to adjacent tutorial
walkthroughs in presentation.
