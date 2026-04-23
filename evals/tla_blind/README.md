# TLA Blind Eval Harness

This directory contains Pass 3 of the TLA+/formal-methods remediation plan.
Its purpose is different from Pass 1 and Pass 2:

- Pass 1 blocks unsafe public formal-methods submission paths.
- Pass 2 tries to mechanically bypass those blocks.
- Pass 3 measures whether the post-remediation harness can distinguish
  legitimate formal work from polished nonsense under uncertainty.

The harness is intentionally lightweight. It provides:

- blind generation tasks with hidden reference material kept out of model prompts
- blind review cases with mixed-quality artifact bundles
- blind repair cases where the model must diagnose, repair, and decide whether
  the artifact should still be rejected
- a shared scoring schema with a mandatory split between
  `formal_validity_score` and `external_admissibility_score`
- a small runner that lists cases, prints model-visible prompts, and
  initializes blank scorecards

## Layout

```text
evals/tla_blind/
├── README.md
├── runner.py
├── rubrics/
│   ├── rubric.md
│   └── scorecard_schema.json
├── tasks/
│   └── generation/
│       ├── known_good_counter/
│       └── qa_translation_observer/
├── review_corpus/
│   ├── good_counter_example/
│   └── polished_bad_observer_firewall/
├── repair_cases/
│   ├── reject_projection_slop/
│   └── revise_bounds_mixup/
└── results/
    └── README.md
```

## Blindness Rules

- Model-visible prompts live in `prompt.md` or `case.md`.
- Hidden labels and references live only under `hidden_reference/` or
  `hidden_label/`.
- The runner prints only model-visible material by default.
- Scoring metadata is separate from the task prompt and should not be shown
  to the system under test.

## Layers

### 1. Blind generation

The system sees a problem statement but not the reference TLA+ solution.
It must produce:

- TLA+ module(s)
- README or explanation
- variable/action justification
- semantics-vs-bounds explanation
- repository-fit rationale

### 2. Blind review

The system reviews mixed artifact bundles without knowing authorship or gold
label. It must return:

- `accept`, `revise`, or `reject`
- specific reasons
- a full scorecard using the shared schema

### 3. Blind repair

The system receives a flawed artifact bundle and must:

- diagnose the problems
- propose or apply repairs
- decide whether the bundle should still be rejected after repair

## Required Score Axes

Every scorecard must include:

- `formal_validity_score`
- `external_admissibility_score`
- `semantic_adequacy_score`
- `outsider_comprehensibility_score`
- `invariant_non_vacuity_score`
- `semantics_vs_bounds_clarity_score`
- `repository_fit_plausibility_score`
- `reviewer_rejection_risk_score`

See [scorecard_schema.json](/home/player2/signal_experiments/evals/tla_blind/rubrics/scorecard_schema.json) and [rubric.md](/home/player2/signal_experiments/evals/tla_blind/rubrics/rubric.md).

## Starter Fixtures

This starter set includes:

- a known-good style counter example for contrast
- a polished but bad/jargon-heavy case resembling the recent failure mode
- a QA-to-TLA translation task requiring outsider language
- a repair case where the correct outcome remains reject

## Runner

Examples:

```bash
python3 evals/tla_blind/runner.py list
python3 evals/tla_blind/runner.py show generation known_good_counter
python3 evals/tla_blind/runner.py init-scorecard review polished_bad_observer_firewall
python3 evals/tla_blind/runner.py --self-test
python3 evals/tla_blind/execute_current_system.py
```

## CI / Workflow Fit

This harness does not replace the formal-publication gate. It sits after Pass 1
and Pass 2:

- Pass 1 asks: "can this ship path be blocked?"
- Pass 2 asks: "can the block be bypassed?"
- Pass 3 asks: "even with the block in place, can the system still tell good
  formal work from polished garbage?"

That separation is deliberate. Passing Pass 1 and Pass 2 is not evidence that
the system's judgment is trustworthy. Pass 3 is the trust probe.
