# Calibration Monitor

This module models a small calibration monitor with state variable `phase` and
numeric deviation tracking. `Observe` moves the monitor through its visible
states, while `Reset` returns to idle.

The semantics comes from the stated monitor behavior. The chosen variables and
actions are justified by that monitor description: `phase` stores the visible
state, `deviation` stores the bounded numeric status, and `Observe` / `Reset`
encode the named updates.

## Intrinsic semantics

The model captures the visible calibration states and their updates.

## TLC bounds

The bounded deviation range limits the model's state space but does not define
its semantics.

## Repository fit

The bundle claims that this research calibration ledger belongs in
`tlaplus/examples` because it uses the same tidy README and small-module style
as public examples, similar to `Clock.tla` and `Timer.tla` in surface layout.
