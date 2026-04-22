# EVIDENCE — provenance of the candidate claim

**NOT** a verification. Record of how the claim surfaced and what empirical
corroboration has been applied so far. Formal verification lives in Lane C.

## Provenance

- **Benchmark:** `qa_tla_blind/` (QA-vs-TLA+ blind reproduction), positive
  control case.
- **Lane A claimant:** fresh general-purpose subagent invoked on
  `2026-04-22`, with only `qa_tla_blind/prompts/qa_control_theorems.md` +
  `qa_tla_blind/README.md` + `CLAIM.md` axiom references as allowed input.
- **Claimant output:** `qa_tla_blind/attempts/qa_control_theorems.md` §5.
- **Claimant could not have seen prior art:** the forbidden-reads list
  included `QA_CONTROL_THEOREMS.md`, `paper1_qa_control.tex` (both paths),
  `QARM_v02_Stats.tla`, `QARM_v02_Stats.cfg`, `ALL_INVARIANTS_VALIDATED.md`.
  Scorer confirmed no leakage from those files in the attempt.

## Empirical checks performed so far (Lane A's scorer pass)

Performed by the scorer subagent, `2026-04-22`, using Tarjan SCC over a
hand-rolled adjacency list:

- `N = 1, 2, 3, 4, 5, 6` — every SCC count and every component size matches
  the claim.
- `N = 30` — single SCC of size 900 (matches even-N claim for N=30).
- `N = 5` (odd) — components of size `(16, 2, 2, 2, 2, 1)` match the claim:
  one inner 16 = 4² = (N-1)², four border 2-cycles for k ∈ {1, 2, 3, 4},
  one singleton {(5, 5)}.
- `N = 3` (odd) — components of size `(4, 2, 2, 1)` match the claim:
  one inner 4 = 2² = (N-1)², two border 2-cycles for k ∈ {1, 2}, one
  singleton {(3, 3)}.

## What is NOT yet checked

- Novelty against `paper1_qa_control.tex` — the paper may already state this
  result or an equivalent. Lane B is responsible for this check.
- Formal proof — Lane C is responsible for producing a proof sketch that
  derives (A) and (B) from the generator definitions, independent of
  enumeration.
- Extended N ranges — verification above covered small N + N=30. Lane C
  should extend to at least N=20 continuous and a larger parity sample.
- Edge cases — N=0 is not in scope; the claim starts at N=1.

## Supersedes / relates to

- **Subsumes** Theorem 2 in `QA_CONTROL_THEOREMS.md:17-34` ("SCC Count under
  μ-Pairing") when Σ is restricted to `{σ, μ, λ₂}` (the no-reverse-path
  condition). The candidate claim is what happens when you add `ν` back in
  (which breaks the no-reverse-path condition for λ₂).
- **Compatible with** SCC Monotonicity Lemma in `QA_CONTROL_THEOREMS.md:6-15`:
  adding `ν` to `{σ, μ, λ₂}` can only decrease `#SCC`. The candidate claim
  says the decrease is from `(N² + N)/2` down to `1` (even N) or `N + 1`
  (odd N).

## Why it matters (if real)

- Would give a closed form for the full-Σ case that the existing three
  theorems explicitly condition away from.
- Would establish a parity-based phase transition in `#SCC(N)` as a function
  of `N`.
- Would identify QA's Cosmos/Satellite/Singularity orbit structure as the
  SCC decomposition on odd `N`, rather than as a separate axiomatic
  taxonomy.

None of these claims apply until Lane B and Lane C clear.
