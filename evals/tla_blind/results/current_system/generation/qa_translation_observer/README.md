# QA-to-TLA Translation Task

## What is modeled
This artifact models the system described in the visible task prompt for `qa_translation_observer`.

## Variable/action justification
- `obs`: Tracks the current observation from the visible finite observation set.
- `streak`: Tracks how many consecutive non-idle observations have occurred.
- `ObserveReady`: Sets obs to ready and increments streak.
- `ObserveFault`: Sets obs to fault while preserving the current non-idle streak.
- `ObserveIdle`: Sets obs to idle and resets streak to 0.

## Intrinsic semantics
- The semantics come from the visible task's finite observation states and state-update rules.

## TLC bounds
- Any TLC cap or depth limit would only bound exploration and would not define the meaning of obs or streak.

## Source grounding
- The semantics come from the visible task statement, not from QA-private theorem language.
- The chosen variables map directly to the named observation and streak concepts in the task.

## Repository fit
This aims at `tlaplus/examples`.
Comparable examples:
- similar to small protocol/state-machine examples
- similar to finite-state monitoring examples in tlaplus/examples
