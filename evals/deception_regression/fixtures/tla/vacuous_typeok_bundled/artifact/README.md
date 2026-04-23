# Counter

## What is modeled
This artifact models the bounded counter described in the visible task prompt.

## Variable/action justification
- `counter`: the counter value; increments on each step.
- `Increment`: advances counter by one.

## Intrinsic semantics
- `counter` is a natural number.
- `Increment` is the only action.

## TLC bounds
- No bounds are required for this finite teaching example.

## Source grounding
- The task describes a counter that increments by one.
- We ground the variables and action in that visible task description.

## Repository fit
This aims at tlaplus/examples. Comparable examples:
- similar to DieHard.tla bounded-state example
