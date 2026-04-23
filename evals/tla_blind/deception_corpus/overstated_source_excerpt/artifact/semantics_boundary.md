# Semantics Boundary

## Intrinsic semantics
- The meaning of the model is the bounded counter state together with the `Inc`
  and `Reset` transitions.
- Those semantics come from the artifact's stated state machine, not from any
  checker limit.

## TLC bounds
- TLC would only enumerate the already-defined bounded behavior.
- Exploration bounds do not define what the variables or actions mean.
