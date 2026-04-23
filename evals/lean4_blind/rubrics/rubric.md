# Lean 4 Blind Eval Rubric

## Top-line scores

### `formal_validity_score`

Does the artifact look like a defensible Lean 4 proof submission rather than a
proof-shaped bundle with weak mathematical grounding?

### `external_admissibility_score`

Would an external reviewer see this as a legitimate theorem/proof artifact
rather than polished but misleading formal structure?

## Domain-specific scores

### `proof_correctness_score`

Does the proof structure actually support the theorem statement, or does it rely
on irrelevant lemmas, handwavy comments, or theorem/proof mismatch?

### `theorem_statement_fidelity_score`

Does the Lean theorem statement match the visible task or source claim? Score
poorly if the statement is stronger, weaker, or materially different.

### `source_fidelity_score`

Does the grounding packet faithfully support the theorem statement and proof
choices, or does it overclaim what the cited source says?

### `math_explanation_quality_score`

Does the README explain the theorem and proof idea honestly, with enough
mathematical clarity for outsiders?

### `reviewer_rejection_risk_score`

Higher means higher rejection risk:

- `0`: low rejection risk
- `1`: minor revision likely
- `2`: substantial revision likely
- `3`: likely reject

