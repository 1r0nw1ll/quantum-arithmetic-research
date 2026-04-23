# Audience Translation

## What is modeled
This artifact models a finite-state calibration monitor using ordinary TLA+
state variables and actions.

## Why this is useful
It translates a plain-language monitoring description into explicit state and
transition rules.

## TLA+ translation
`phase` tracks the visible control state and `deviation` tracks the bounded
numeric status used by the monitor.
