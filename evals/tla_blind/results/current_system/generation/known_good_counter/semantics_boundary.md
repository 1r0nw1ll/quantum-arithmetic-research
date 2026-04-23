# Semantics Boundary For Known-Good Counter Spec

## Intrinsic semantics
- The semantics are the bounded counter state and the Inc and Reset transitions.
- The intrinsic semantics come from the visible task description of the state space and allowed transitions, not from any search bound or checker setting.

## TLC bounds
- No extra TLC finiteness device is needed beyond the bounded domain already in the model.
- TLC configuration only limits exploration of the already-defined transition system; those limits do not change what the variables or actions mean.
