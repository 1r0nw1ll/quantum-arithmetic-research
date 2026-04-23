# Semantics Boundary For QA-to-TLA Translation Task

## Intrinsic semantics
- The semantics come from the visible task's finite observation states and state-update rules.
- The intrinsic semantics come from the visible task description of the state space and allowed transitions, not from any search bound or checker setting.

## TLC bounds
- Any TLC cap or depth limit would only bound exploration and would not define the meaning of obs or streak.
- TLC configuration only limits exploration of the already-defined transition system; those limits do not change what the variables or actions mean.
