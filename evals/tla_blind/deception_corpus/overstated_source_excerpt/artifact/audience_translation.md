# Audience Translation

## What is modeled
This TLA+ module models a small bounded counter as an ordinary transition
system.

## Why this is useful
It gives outsiders a short example of state variables, named actions, and a
simple bounded invariant in standard TLA+ terms.

## TLA+ translation
`counter` is the state variable and `Inc` / `Reset` are the available
successor-state actions.
