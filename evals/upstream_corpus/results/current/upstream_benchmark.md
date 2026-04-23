# Upstream-Approved-Corpus Benchmark (Pass a, as-is)

## Provenance
- **tla**: `tlaplus/Examples` @ `d9ce4db7c770cc82e662870bce168ff8f59aff24`
- **lean4**: `leanprover-community/mathematics_in_lean` @ `2bf0e10dd0c02438b65110f85cd9b68a9dbe6e39`

## TLA
- Total cases: 77
- Decisions: {'accept': 0, 'revise': 0, 'reject': 77}
- **Acceptance rate on upstream approved corpus: 0.00%**
- **False reject rate: 100.00%**
- Non-accept rate (revise+reject): 100.00%

### Finding-bucket totals (count = total occurrences)
- `missing_required_artifact`: 462 findings across 77 cases
- `missing_explicit_evidence`: 215 findings across 77 cases
- `weak_outsider_translation`: 94 findings across 71 cases
- `weak_repo_fit_signal`: 78 findings across 77 cases

### False reject decomposition
- Total false rejects: 77
- Driven purely by missing-required-artifact: 0
- Driven by substantive findings only (no missing-artifact): 0
- Mixed / other: 77

### Top finding strings (verbatim, top 20)
- [missing_required_artifact] `missing required artifact audience_translation.md` — 77
- [missing_required_artifact] `missing required artifact semantics_boundary.md` — 77
- [missing_required_artifact] `missing required artifact repo_fit_review.json` — 77
- [missing_required_artifact] `missing required artifact skeptical_review.json` — 77
- [missing_required_artifact] `missing required artifact source_grounding.json` — 77
- [missing_required_artifact] `missing required artifact repo_comparables.json` — 77
- [missing_explicit_evidence] `Adversarial check: claimed grounding does not robustly support the artifact` — 77
- [weak_repo_fit_signal] `Adversarial check: repository-fit claim is overstated relative to the comparable set` — 77
- [missing_explicit_evidence] `Explanatory text does not justify the chosen variables/actions` — 46
- [missing_explicit_evidence] `Explanatory text does not say where the semantics come from` — 35
- [weak_outsider_translation] `README explanation missing or unreadable` — 29
- [missing_explicit_evidence] `No explanatory text available for source grounding` — 29
- [missing_explicit_evidence] `Explanatory text does not clearly state what is being modeled` — 28
- [weak_outsider_translation] `README does not map action names into outsider-facing prose` — 27
- [weak_outsider_translation] `README does not explain all state variables: flag, max, num, nxt, pc, previous, unchecked` — 1
- [weak_outsider_translation] `README does not explain all state variables: chameneoses, meetingPlace, numMeetings` — 1
- [weak_outsider_translation] `README does not explain all state variables: dealer` — 1
- [weak_outsider_translation] `README does not explain all state variables: ringbuffer` — 1
- [weak_outsider_translation] `README does not explain all state variables: cLogs, executed` — 1
- [weak_outsider_translation] `README does not explain all state variables: grid` — 1

## LEAN4
- Total cases: 43
- Decisions: {'accept': 0, 'revise': 0, 'reject': 43}
- **Acceptance rate on upstream approved corpus: 0.00%**
- **False reject rate: 100.00%**
- Non-accept rate (revise+reject): 100.00%

### Finding-bucket totals (count = total occurrences)
- `weak_outsider_translation`: 86 findings across 43 cases
- `missing_explicit_evidence`: 43 findings across 43 cases
- `weak_repo_fit_signal`: 43 findings across 43 cases
- `substantive_issue`: 22 findings across 19 cases

### False reject decomposition
- Total false rejects: 43
- Driven purely by missing-required-artifact: 0
- Driven by substantive findings only (no missing-artifact): 19
- Mixed / other: 24

### Top finding strings (verbatim, top 20)
- [weak_outsider_translation] `README does not explicitly state theorem-statement fidelity` — 43
- [missing_explicit_evidence] `Source grounding is missing explicit excerpts` — 43
- [weak_outsider_translation] `README does not explain the proof idea` — 43
- [weak_repo_fit_signal] `Bundle does not justify why the proof is worth external review` — 43
- [substantive_issue] `Theorem statement appears to import group-level claims for a natural-number proof task` — 19
- [substantive_issue] `Lean proof contains sorry/admit placeholder` — 3
