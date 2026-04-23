# Audience Translation For QA-to-TLA Translation Task

## What is modeled
This TLA+ artifact models the outsider-visible transition system described in the task `qa_translation_observer`. The state is represented by obs, streak, and the enabled behaviors are expressed through the actions ObserveReady, ObserveFault, ObserveIdle.

## Why this is useful
This formalization is useful because it translates the task's plain-language state updates into ordinary TLA+ state variables, initial conditions, and next-state actions that an experienced maintainer can inspect without project-private context.

## TLA+ translation
The translation uses standard TLA+ vocabulary: state variables record the current state, actions define valid successor states, and invariants summarize substantive properties that hold across all reachable states in the intended finite model.
