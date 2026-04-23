# Counter Example

This module models a single bounded counter. The state variable `counter`
tracks the current value. `Inc` increments while the counter is below the
maximum; `Reset` returns it to zero.

## Intrinsic semantics

The semantics are the state variable and the `Inc` / `Reset` transitions.

## TLC bounds

No extra TLC finiteness device is needed beyond the bounded integer domain
already present in the model.

## Repository fit

This is a small illustrative transition system in the style expected by
`tlaplus/examples`.
