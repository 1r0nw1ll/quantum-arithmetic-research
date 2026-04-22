# DECISION PENDING — promotion gate result for `control_extension_oddN_fullSigma`

**Written:** 2026-04-22.
**Scope:** promotion-gate decision. Based on Lane B + Lane C results; no new
derivation. Supersedes the initial "unchecked" state but does not promote.

## Gate result (per `candidates/README.md`)

The promotion gate requires **both** conditions:

1. `NOVELTY_STATUS.status ∈ {"not_found", "unclear_but_likely_absent"}`
2. `FORMAL_STATUS.status ∈ {"proof_sketch_provided", "machine_verified"}`

Current readings:

- **`NOVELTY_STATUS.status = "partially_present"`** (Lane B, high confidence).
  *Not* in the gate-qualifying set.
- **`FORMAL_STATUS.status = "proof_sketch_provided"`** (Lane C, degraded
  independence — main-session fallback during subagent quota window).
  In the gate-qualifying set.

**Gate status: NOT CLEARED.** Only Lane C cleared its side.

## Ladder position

Per the ladder in `candidates/README.md`:

- `unchecked` — just filed, no lane has run
- `candidate_observation` — some lane has run but gate conditions unmet
- **`formally_supported_candidate`** — Lane C clears; Lane B inconclusive or pending ← **this candidate sits here**
- `novel_result_pending_writeup` — Lane B returns `not_found` AND Lane C clears
- `canonical_theorem` — reserved for post-paper promotion

See `NOVELTY_STATUS.json` and `FORMAL_STATUS.json` — both now carry a
`ladder_status` field set to `formally_supported_candidate`.

## Why "partially_present" — the precise split

The CLAIM.md bundles three substantive propositions. Lane B's per-part verdicts:

| Part | Claim | Lane B verdict | Notes |
|---|---|---|---|
| A | Even N ≥ 2 ⇒ `#SCC = 1, max|SCC| = N²` (universal) | `partially_present_as_instances` | Paper tabulates only N=30 and N=50 under Table 1. Not stated as a theorem over all even N. The universal form is not in the sources. |
| B | Odd N ≥ 3 ⇒ `#SCC = N+1, max|SCC| = (N-1)²` | `not_found` | No odd-N data point, example, or theorem exists in any consulted file. |
| C | Odd-N structural decomposition (inner `Caps(N-1, N-1)` + `N-1` border 2-cycles + singleton `(N, N)`) | `not_found` | Not stated and not derivable in one step from Lemma 3.1 + Theorems 3.2, 3.3. |

Aggregate: `partially_present`. No single verdict covers the mixed outcome
cleanly under the per-part rubric.

## Why NOT to promote now (strict reading)

The bundled CLAIM.md cannot cleanly clear the gate because its Part (A) has
empirical precedent in the paper (N ∈ {30, 50}) even though no universal
theorem is stated. Promoting the bundled claim would implicitly elevate those
two empirical instances to "novel theorem status," which is false. A clean
audit trail requires either:

- Tighter novelty language on Part (A) (e.g., "universal form of an
  empirically-observed pattern for two specific even N"), or
- Splitting the CLAIM into separate sub-candidates.

## Next step (deferred — author decision)

**Split the CLAIM.md into three sub-candidate packets, each with its own
lane results:**

- `control_extension_oddN_fullSigma/A_even_universal/` — carries Part (A) as a
  candidate "universal form of a partially-tabulated pattern." Expected
  `partially_present` for the even-N general theorem; not-found for the
  universal statement; may still be novelty-cleared with careful wording.
- `control_extension_oddN_fullSigma/B_odd_counts/` — carries Part (B). Strong
  candidate for `not_found`, pending a re-audit under the split CLAIM.
- `control_extension_oddN_fullSigma/C_odd_decomposition/` — carries Part (C).
  Strong candidate for `not_found`, pending a re-audit under the split CLAIM.

Each sub-candidate then runs its own Lane B + Lane C (or re-uses the current
Lane C, citing the shared enumeration). After split, the per-candidate gate
can clear cleanly.

## What IS ready and committed

- `CLAIM.md` — frozen statement of the bundled claim (Lane A).
- `EVIDENCE.md` — Lane A's provenance + initial empirical checks.
- `NOVELTY_AUDIT.md` — Lane B's full audit, per-part verdicts with citations.
- `NOVELTY_STATUS.json` — `partially_present`, high confidence.
- `PROOF_SKETCH.md` — Lane C's structural argument (odd-N conditional on
  even-N; flagged completeness gap).
- `verify_extension.py` + `fixtures/` — Lane C's enumerative verifier, 17/17
  N values pass.
- `FORMAL_STATUS.json` — `proof_sketch_provided`, with degraded-independence
  note for Lane C.

No file in this candidate packet promotes, claims novelty, or attributes
authorship of a new theorem. All such language is explicitly absent by
decision of the author and deferred to a later split-and-promote step.

## Audit summary

- Lane A (claimant): blind reproducer, fresh subagent, 2026-04-22.
- Lane B (novelty auditor): fresh subagent, full corpus read, forbidden-reads
  pact held.
- Lane C (formal verifier): main-session fallback due to subagent quota
  exhaustion; script is deterministic and re-runnable.
- Promotion gate: NOT cleared. Candidate sits at `formally_supported_candidate`.
- No commit prior to this one has elevated the candidate beyond "observation."
