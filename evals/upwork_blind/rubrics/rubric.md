# Upwork-Style Blind Eval Rubric

Anti-slop rubric for tightly-specified practical tasks (bugfix, small script,
data transform, API task, explicit expected output). Measures whether a
deliverable is operationally correct, not whether it is stylistically
polished.

## Allowed task classes

- **Bugfix** — spec describes a failing input/test, deliverable must make it pass without breaking previous behavior.
- **Small script** — spec names inputs, outputs, and invocation, deliverable must run and produce the expected output.
- **Data transform** — spec provides source data and target schema, deliverable must transform correctly including boundary conditions.
- **API task** — spec names endpoints/behavior/retry policy, deliverable must implement that behavior.
- **Explicit expected output** — spec provides golden input + golden output, deliverable must match.

## Explicitly out-of-scope

- Branding / naming / design / copywriting
- Vague app-build requests ("build me a Twitter clone")
- Subjective acceptance criteria ("should feel modern", "should look clean")

Those are excluded because the harness has no honest way to test for slop in
subjective deliverables. Adding them would invite exactly the false-accept
failure mode this suite is trying to guard against.

## Score axes

See [scorecard_schema.json](/home/player2/signal_experiments/evals/upwork_blind/rubrics/scorecard_schema.json).

Two top-line:

- **task_validity_score** — does the deliverable actually solve the task?
- **external_admissibility_score** — would a real client accept this as-is?

Five supporting:

- **requirement_coverage_score** — every named requirement addressed
- **deliverable_fit_score** — format matches what was asked for
- **scope_honesty_score** — README doesn't claim more than the code does
- **client_utility_score** — a non-author can run it
- **source_fidelity_score** — deliverable reflects the spec's language + examples

Plus one aggregate:

- **reviewer_rejection_risk_score** — compound reject-risk signal

## Decision rules

- `reject` — `task_validity_score <= 0` OR source code missing OR scope_honesty is saturated-bad (0) AND requirement_coverage is weak (<= 1)
- `revise` — any axis < 3 but no reject trigger
- `accept` — all axes 3, reviewer_rejection_risk 0

## What the scorer looks for

**Reject-level findings:**
- source code file missing entirely (README-only deliverables)
- proof-shaped placeholders in the source (`pass`, `return None  # TODO`, `raise NotImplementedError`)
- tests claimed in README but no test file present
- tests present but contain only `assert True` / `assert x == x` tautologies

**Revise-level findings:**
- README overclaims ("handles all inputs", "production-ready", "comprehensive test coverage") without supporting evidence in code
- spec mentions edge cases / errors / retries that the code does not branch on
- no invocation instructions in README
- spec gives golden examples, tests do not cover them

**Accept-level:**
- source code addresses the core operation
- tests cover spec-named examples
- README matches implementation scope honestly
- edge cases / errors / retries handled where the spec calls them out
