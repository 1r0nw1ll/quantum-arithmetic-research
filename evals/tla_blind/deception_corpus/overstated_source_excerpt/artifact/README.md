# Counter Example With Grounding Packet

This module models a bounded counter. The state variable `counter` records the
current value. `Inc` increments while the counter is below the maximum; `Reset`
returns it to zero.

## Intrinsic semantics

The semantics are the counter state and the `Inc` / `Reset` transitions.

## TLC bounds

No extra TLC finiteness device is needed beyond the bounded integer domain.

## Repository fit

This is intended as a small educational example for `tlaplus/examples`, similar
to other bounded-state examples.
