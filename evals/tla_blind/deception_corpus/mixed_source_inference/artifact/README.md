# Counter Example

This artifact models a bounded counter. The semantics comes from the task's
bounded counter description. The chosen variable and actions are justified by
that description: `counter` stores the current value and `Inc` / `Reset`
implement the named updates.

## Intrinsic semantics

The semantics are the bounded counter state together with the `Inc` and
`Reset` transitions.

## TLC bounds

The bounded domain only limits exploration.

## Repository fit

This is a small teaching example for `tlaplus/examples`, similar to `Clock.tla`
and other bounded counter examples.
