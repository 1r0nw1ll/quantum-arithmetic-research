# Upwork-Style Blind Eval Suite

Third domain for the anti-slop legitimacy harness, following
[tla_blind](/home/player2/signal_experiments/evals/tla_blind) and
[lean4_blind](/home/player2/signal_experiments/evals/lean4_blind).

**Goal:** test whether the harness generalizes from formal domains (where
failure shape is tautological invariants / deceptive sorries) to practical
task domains (where failure shape is polished but operationally wrong output,
happy-path-only coverage, overclaimed scope, fake test assertions).

**Scope:** benchmark only. Not an income-track recommendation. Task classes
are restricted to tightly specified jobs with hard acceptance criteria —
bugfix, small script, data transform, API task, explicit expected output.
Subjective / vague / branding / design tasks are excluded because the harness
has no honest way to test for slop in them.

## Layout

```text
evals/upwork_blind/
├── README.md                    (this file)
├── runner.py
├── execute_current_system.py
├── test_execute_current_system.py
├── rubrics/
│   ├── rubric.md
│   └── scorecard_schema.json
├── tasks/
│   └── generation/              (blind generation task specs + hidden refs)
├── review_corpus/               (mixed-quality deliverables to review)
├── repair_cases/                (flawed deliverables to diagnose/repair)
├── deception_corpus/            (polished-but-wrong fixtures)
└── results/
```

## Layers

### 1. Blind generation

System sees a task spec but not the hidden reference solution. Produces:
- source code that addresses the task
- README describing what was delivered
- tests that cover the spec's named examples

### 2. Blind review

System reviews mixed bundles without knowing hidden label. Returns
`accept | revise | reject` + a scorecard using the shared schema.

### 3. Blind repair

Flawed deliverable, system must diagnose and decide whether repair is
sufficient or whether the bundle should remain rejected.

### 4. Deception (polished-but-wrong)

Deliverables that look professional but fail hidden acceptance criteria:
- happy-path-only code with overclaimed README
- test files with `assert True` tautologies + claimed coverage
- fake error handling (try/except that logs-and-swallows)
- requirement drop-outs (spec lists 5 things, deliverable does 3)

## Runner

```bash
python3 evals/upwork_blind/runner.py list
python3 evals/upwork_blind/runner.py show review good_csv_domain_count
python3 evals/upwork_blind/execute_current_system.py
```

## CI fit

This suite is run as a peer of the TLA+ and Lean 4 blind suites under
`evals/blind_benchmark/` (internal accuracy) and
`evals/deception_regression/` (precision regression).
