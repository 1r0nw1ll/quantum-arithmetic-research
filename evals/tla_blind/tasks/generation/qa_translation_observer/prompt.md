Translate a small QA-origin process into an outsider-readable TLA+ example.

Source process:

- A system receives observations from a finite set `Obs = {"idle", "ready", "fault"}`
- The state keeps the current observation and a counter of how many consecutive
  non-`idle` observations have occurred
- `ObserveReady` changes the observation to `ready` and increments the counter
- `ObserveFault` changes the observation to `fault`
- `ObserveIdle` changes the observation to `idle` and resets the counter

Required output:

- a TLA+ module
- an explanation written for TLA+ maintainers, not QA insiders
- explicit variable and action purpose mapping
- a section named `Intrinsic semantics`
- a section named `TLC bounds`
- a short repository-fit note

Constraint:

- Do not use QA-private phrases like "observer projection firewall" unless you
  translate them into normal formal-methods language.
