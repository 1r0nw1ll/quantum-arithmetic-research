# qa_tla_blind/candidates/ — candidate claims surfaced by the benchmark

## Purpose

When the benchmark's blind reproducer produces a result that goes **beyond**
what the ground truth contains, that result enters this directory as a
**candidate**, not a theorem. Each candidate gets an independent verification
pipeline before any promotion.

## The integrity rule

> The agent that generated the candidate must not be the agent that decides
> whether it is new.

Three lanes, three artifacts, no shared context:

- **Lane A — claimant.** The original blind reproducer. Its output is frozen
  as `CLAIM.md` + `EVIDENCE.md`. It cannot amend itself after freeze.
- **Lane B — source auditor.** A fresh agent reads only the paper/spec
  authority layer + `CLAIM.md`, decides novelty. Output: `NOVELTY_AUDIT.md`
  + updated `NOVELTY_STATUS.json`.
- **Lane C — formal verifier.** A fresh agent writes executable verification
  (enumeration / model check) + proof sketch, decides truth. Output:
  `verify_extension.py` + `fixtures/` + `PROOF_SKETCH.md` + updated
  `FORMAL_STATUS.json`.

No lane sees the others' outputs until all three are filed.

## Layout per candidate

```
candidates/<slug>/
├── CLAIM.md              # Lane A frozen: precise statement, no prose
├── EVIDENCE.md           # Lane A frozen: provenance, empirical checks so far
├── NOVELTY_STATUS.json   # Lane B status
├── NOVELTY_AUDIT.md      # Lane B: audit with exact citations
├── FORMAL_STATUS.json    # Lane C status
├── PROOF_SKETCH.md       # Lane C: structural proof sketch
├── verify_extension.py   # Lane C: executable verifier
└── fixtures/             # Lane C: per-N enumerations
```

## Promotion statuses

- `unchecked` — just filed, no lane has run
- `candidate_observation` — some lane has run but gate conditions unmet
- `formally_supported_candidate` — Lane C clears; Lane B inconclusive or
  pending
- `novel_result_pending_writeup` — Lane B returns `not_found` AND Lane C
  clears with proof sketch
- `canonical_theorem` — reserved for after paper/cert promotion. Do not
  assign from this directory.

## Promotion gate

A candidate moves from `unchecked` → `novel_result_pending_writeup` only if
**both**:

1. `NOVELTY_STATUS.status ∈ {"not_found", "unclear_but_likely_absent"}`
2. `FORMAL_STATUS.status ∈ {"proof_sketch_provided", "machine_verified"}`
   (strictly stronger than `"empirically_supported"`)

If either fails, the candidate stays at its current status. No commit message
or doc may describe it as "new" or "a theorem" below that gate.

## Do not

- Do not let the claimant read the source paper to decide novelty.
- Do not let enumeration alone count as theorem status.
- Do not let "nobody on the team noticed this before" count as novelty.
- Do not bundle a candidate promotion into a benchmark-infrastructure
  commit. Separate commits.

## Current candidates

- `control_extension_oddN_fullSigma/` — status: `unchecked`
