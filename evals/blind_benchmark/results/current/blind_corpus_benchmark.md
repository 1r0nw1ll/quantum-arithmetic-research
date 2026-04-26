# Blind Corpus Benchmark Sweep

## Cross-Domain Summary
- Total labeled fixtures: 22
- Overall accuracy: 100.00%
- False accept count: 0
- False reject count: 0

### Recommended Next Fixes
- No corpus-level misclassifications on the current labeled suites. The smallest next step is to enlarge the deception and borderline corpora rather than retuning thresholds.

## TLA+
- Labeled fixtures: 9
- Overall decision accuracy: 100.00%
- False accept rate: 0.00%
- False reject rate: 0.00%
- Status call: balanced

### Label Counts
- Expected: {'accept': 2, 'revise': 2, 'reject': 5}
- Predicted: {'accept': 2, 'revise': 2, 'reject': 5}

### Confusion Matrix
| expected \ predicted | accept | revise | reject |
|---|---:|---:|---:|
| accept | 2 | 0 | 0 |
| revise | 0 | 2 | 0 |
| reject | 0 | 0 | 5 |

### False Accepts
- none

### False Rejects
- none

### Wrong Transitions
- none

### Score Distributions
- `external_admissibility_score`: {'0': 6, '1': 1, '3': 2}
- `formal_validity_score`: {'1': 2, '3': 7}
- `invariant_non_vacuity_score`: {'0': 2, '3': 7}
- `outsider_comprehensibility_score`: {'0': 2, '3': 7}
- `repo_comparable_support_score`: {'1': 2, '2': 2, '3': 5}
- `repo_comparables_evidence_score`: {'1': 2, '2': 2, '3': 5}
- `repository_fit_plausibility_score`: {'0': 4, '1': 2, '3': 3}
- `reviewer_rejection_risk_score`: {'0': 2, '2': 1, '3': 6}
- `semantic_adequacy_score`: {'0': 2, '2': 2, '3': 5}
- `semantics_vs_bounds_clarity_score`: {'1': 3, '3': 6}
- `source_fidelity_score`: {'0': 1, '1': 1, '2': 3, '3': 4}
- `source_grounding_score`: {'0': 3, '1': 1, '2': 1, '3': 4}

## Lean 4
- Labeled fixtures: 5
- Overall decision accuracy: 100.00%
- False accept rate: 0.00%
- False reject rate: 0.00%
- Status call: balanced

### Label Counts
- Expected: {'accept': 2, 'revise': 1, 'reject': 2}
- Predicted: {'accept': 2, 'revise': 1, 'reject': 2}

### Confusion Matrix
| expected \ predicted | accept | revise | reject |
|---|---:|---:|---:|
| accept | 2 | 0 | 0 |
| revise | 0 | 1 | 0 |
| reject | 0 | 0 | 2 |

### False Accepts
- none

### False Rejects
- none

### Wrong Transitions
- none

### Score Distributions
- `external_admissibility_score`: {'1': 2, '2': 1, '3': 2}
- `formal_validity_score`: {'1': 2, '3': 3}
- `math_explanation_quality_score`: {'1': 1, '3': 4}
- `proof_correctness_score`: {'0': 2, '3': 3}
- `reviewer_rejection_risk_score`: {'0': 2, '1': 1, '3': 2}
- `source_fidelity_score`: {'3': 5}
- `theorem_statement_fidelity_score`: {'2': 2, '3': 3}

## Upwork-style
- Labeled fixtures: 8
- Overall decision accuracy: 100.00%
- False accept rate: 0.00%
- False reject rate: 0.00%
- Status call: balanced

### Label Counts
- Expected: {'accept': 2, 'revise': 1, 'reject': 5}
- Predicted: {'accept': 2, 'revise': 1, 'reject': 5}

### Confusion Matrix
| expected \ predicted | accept | revise | reject |
|---|---:|---:|---:|
| accept | 2 | 0 | 0 |
| revise | 0 | 1 | 0 |
| reject | 0 | 0 | 5 |

### False Accepts
- none

### False Rejects
- none

### Wrong Transitions
- none

### Score Distributions
- `client_utility_score`: {'3': 8}
- `deliverable_fit_score`: {'3': 8}
- `external_admissibility_score`: {'0': 5, '2': 1, '3': 2}
- `requirement_coverage_score`: {'0': 1, '1': 1, '3': 6}
- `reviewer_rejection_risk_score`: {'0': 2, '1': 1, '3': 5}
- `scope_honesty_score`: {'0': 4, '1': 2, '3': 2}
- `source_fidelity_score`: {'1': 1, '3': 7}
- `task_validity_score`: {'0': 1, '1': 1, '2': 1, '3': 5}
