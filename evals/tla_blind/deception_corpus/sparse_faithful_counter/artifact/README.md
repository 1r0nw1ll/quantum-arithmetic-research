# Counter

`counter` stores the current bounded value. `Inc` increments while below the
limit. `Reset` returns to zero.

The semantics comes from the bounded-counter description used for this artifact.
The chosen variable and actions are justified by that description: `counter`
stores the value and `Inc` / `Reset` encode the named updates.

## Intrinsic semantics

The model semantics are exactly those two transitions over the bounded counter
state.

## TLC bounds

The finite integer domain limits exploration only.

## Repository fit

This is a small bounded-state example in the style of `Clock.tla`.
