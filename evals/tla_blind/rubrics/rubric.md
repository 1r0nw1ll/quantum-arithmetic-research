# Blind TLA+ Rubric

Use this rubric when scoring outputs from blind generation, blind review, or
blind repair tasks. Scores are ordinal:

- `0` = absent, deeply flawed, or misleading
- `1` = weak, partial, or heavily caveated
- `2` = adequate but with real concerns
- `3` = strong for the task scope

## Mandatory Top-Line Scores

### `formal_validity_score`

Score whether the artifact has a defensible formal structure:

- state variables are coherent
- actions are meaningful
- invariants are non-vacuous
- semantics are internally consistent

Do not give a high score merely because the syntax looks TLA+-like.

### `external_admissibility_score`

Score whether an experienced external maintainer would consider the artifact
legitimate, understandable, and appropriate for the target repository.

This is the primary score that guards against the recent failure mode.

## Secondary Axes

### `semantic_adequacy_score`

Does the artifact explain what is actually modeled and why the chosen state and
actions are necessary?

### `source_grounding_score`

Does the artifact explain:

- what is being modeled
- where the semantics come from
- why the chosen variables and actions are justified

Project-private theory without target-community translation should score poorly
here.

### `outsider_comprehensibility_score`

Could a TLA+ maintainer or technically literate outsider understand the model
without project-private context?

### `invariant_non_vacuity_score`

Do the invariants say something substantive, rather than restating syntax or
tautologies?

### `semantics_vs_bounds_clarity_score`

Does the artifact clearly separate intrinsic semantics from TLC or bounded
model-checking choices?

### `repository_fit_plausibility_score`

Would the claimed target repository plausibly want this artifact?

### `repo_comparables_evidence_score`

Does the artifact support repository-fit claims with comparable accepted
artifact classes or example names, rather than bare assertions?

### `reviewer_rejection_risk_score`

Higher means higher rejection risk:

- `0` = low risk
- `1` = some revision risk
- `2` = likely major revision
- `3` = likely reject

## Decision Guidance

- `accept`: strong formal validity, strong external admissibility, low rejection risk
- `revise`: mixed but salvageable
- `reject`: polished nonsense, unclear semantics, repo mismatch, vacuity, or incoherence

## Blindness Rule

Do not use hidden labels when grading a live model output. Hidden labels are for
post-hoc comparison only.
