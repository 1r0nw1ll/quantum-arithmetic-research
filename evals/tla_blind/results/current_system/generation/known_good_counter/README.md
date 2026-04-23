# Known-Good Counter Spec

## What is modeled
This artifact models the system described in the visible task prompt for `known_good_counter`.

## Variable/action justification
- `counter`: Tracks the current counter value.
- `Inc`: Increments while the counter is below 3.
- `Reset`: Returns the counter to 0.

## Intrinsic semantics
- The semantics are the bounded counter state and the Inc and Reset transitions.

## TLC bounds
- No extra TLC finiteness device is needed beyond the bounded domain already in the model.

## Source grounding
- The model comes directly from the visible task statement describing a bounded counter with reset.
- The chosen variable and actions are justified by the state and transition clauses in that task.

## Repository fit
This aims at `tlaplus/examples`.
Comparable examples:
- similar to Clock.tla
- similar to other small bounded-state examples in tlaplus/examples
