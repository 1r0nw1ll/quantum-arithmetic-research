# Upstream-Approved-Corpus Benchmark (Pass 7 — intrinsic vs bundle-completeness)

## Provenance
- **tla**: `tlaplus/Examples` @ `d9ce4db7c770cc82e662870bce168ff8f59aff24`
- **lean4**: `leanprover-community/mathematics_in_lean` @ `2bf0e10dd0c02438b65110f85cd9b68a9dbe6e39`

## Pass-7 Headlines
- **TLA**: pass-a combined accept rate **0.0%**, pass-7 intrinsic-only accept rate **45.5%**. 6/77 cases flipped reject → accept after separating bundle-completeness.
- **LEAN4**: pass-a combined accept rate **0.0%**, pass-7 intrinsic-only accept rate **100.0%**. 0/43 cases flipped reject → accept after separating bundle-completeness.

## TLA
- Total cases: 77

### Axis 1: intrinsic legitimacy (artifact-only, no bundle requirements)
- Decisions: {'accept': 35, 'revise': 42, 'reject': 0}
- **Acceptance rate: 45.45%**
- **False reject rate: 0.00%**

#### Finding-bucket totals (intrinsic axis)
- `weak_outsider_translation`: 65 findings across 42 cases

#### Top intrinsic finding strings
- [weak_outsider_translation] `README does not map action names into outsider-facing prose` — 27
- [weak_outsider_translation] `README does not explain all state variables: flag, max, num, nxt, pc, previous, unchecked` — 1
- [weak_outsider_translation] `README does not explain all state variables: chameneoses, meetingPlace, numMeetings` — 1
- [weak_outsider_translation] `README does not explain all state variables: dealer` — 1
- [weak_outsider_translation] `README does not explain all state variables: ringbuffer` — 1
- [weak_outsider_translation] `README does not explain all state variables: cLogs, executed` — 1
- [weak_outsider_translation] `README does not explain all state variables: grid` — 1
- [weak_outsider_translation] `README does not explain all state variables: Keys, \*, data, keys, mapping, missed, ops, pc, read_keys, snapshotStore, tx, values., write_keys` — 1
- [weak_outsider_translation] `README does not explain all state variables: The, \*, are, here, probability, state, we` — 1
- [weak_outsider_translation] `README does not explain all state variables: j, pc` — 1
- [weak_outsider_translation] `README does not explain all state variables: high, low, pc, result, seq0, val` — 1
- [weak_outsider_translation] `README does not explain all state variables: cnt` — 1
- [weak_outsider_translation] `README does not explain all state variables: msgs, observed, pc, pending` — 1
- [weak_outsider_translation] `README does not explain all state variables: chosen, maxBal, maxVBal, maxVal, msgs, votes` — 1
- [weak_outsider_translation] `README does not explain all state variables: Decided, Maximum, Set, Value, \*, accepted., acceptor, acceptor., all, ballot, cValue, chosen, coordinator., decision, has, maxBallot, maxVBallot, maxValue, seen., sent., value` — 1

### Axis 2: submission-bundle completeness (Codex bundle format)
- Decisions: {'accept': 0, 'revise': 30, 'reject': 47}
- Acceptance rate: 0.00%
- False reject rate: 61.04%
- (Expected to reject all upstream by design — these files lack our local bundle shape.)

### Pass-a vs Pass-7 delta
- 6/77 upstream-approved cases flip from reject (pass-a combined) to accept under intrinsic-only scoring. The remaining 0 rejections are not bundle-dependence — they are deeper intrinsic heuristic overfit.

### Combined (submission-gate simulation)
- Decisions: {'accept': 0, 'revise': 30, 'reject': 47}
- Acceptance rate: 0.00%

## LEAN4
- Total cases: 43

### Axis 1: intrinsic legitimacy (artifact-only, no bundle requirements)
- Decisions: {'accept': 43, 'revise': 0, 'reject': 0}
- **Acceptance rate: 100.00%**
- **False reject rate: 0.00%**

#### Finding-bucket totals (intrinsic axis)
- `substantive_issue`: 1 findings across 1 cases

#### Top intrinsic finding strings
- [substantive_issue] `Lean artifact contains 5 pedagogical sorry (structure-instance field stub); acceptable for textbook exercise corpora` — 1

### Axis 2: submission-bundle completeness (Codex bundle format)
- Decisions: {'accept': 0, 'revise': 43, 'reject': 0}
- Acceptance rate: 0.00%
- False reject rate: 0.00%
- (Expected to reject all upstream by design — these files lack our local bundle shape.)

### Pass-a vs Pass-7 delta
- 0/43 upstream-approved cases flip from reject (pass-a combined) to accept under intrinsic-only scoring. The remaining 0 rejections are not bundle-dependence — they are deeper intrinsic heuristic overfit.

### Combined (submission-gate simulation)
- Decisions: {'accept': 0, 'revise': 43, 'reject': 0}
- Acceptance rate: 0.00%
