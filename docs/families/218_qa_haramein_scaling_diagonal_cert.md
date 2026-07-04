# [218] QA Haramein Scaling Diagonal Cert

**Schema:** `QA_HARAMEIN_SCALING_DIAGONAL_CERT.v1`
**Status:** draft, 2026-04-13
**Originator:** Will Dale
**Validator:** `qa_alphageometry_ptolemy/qa_haramein_scaling_diagonal_cert_v1/qa_haramein_scaling_diagonal_cert_validate.py`
**Theory note:** [`docs/theory/QA_HARAMEIN_SCALING_DIAGONAL.md`](../theory/QA_HARAMEIN_SCALING_DIAGONAL.md)
**Primary source:** Haramein, Rauscher, Hyson (2008), *Scale Unification — A Universal Scaling Law for Organized Matter*, Proc. Unified Theories Conf., Budapest. PDF: `Documents/haramein_rsf/scale_unification_2008.pdf`.

## Claim

The six canonical scale-points of Haramein 2008 Table 1 — Big Bang/Planck, Atomic, Stellar Solar, Galactic G1, Galactic G2, Universe — expressed as integer `(log₁₀ R(cm), log₁₀ ν(Hz))` tuples `(b, e)`, satisfy:

1. **Fixed-d hyperbola:** `b + e ∈ {9, 10, 11}` across all 6 rows (the Schwarzschild line `R·ν = c` after decade-rounding).
2. **φ²-structured ratios:** four segment-pair quadratic-form quotients approximate `{φ², 1/φ²}` within 7%:
   - `(25² + 25²) / (16² + 15²) = 1250 / 481 ≈ φ²` (0.7% off)
   - `(6² + 7²)  / (4² + 4²)   = 85 / 32 ≈ φ²` (1.4% off)
   - `(2² + 3²)  / (4² + 4²)   = 13 / 32 ≈ 1/φ²` (6.3% off)
   - `(16² + 16²) / (25² + 25²) = 512 / 1250 ≈ 1/φ²` (7.3% off)
3. **Null significance:** p < 5×10⁻⁶ against 200,000 random 6-point slope-≈−1 line placements, same structural pair positions.

## QA diagonal class

**Fixed-d hyperbola.** Distinct from:
- `[217]` Fuller VE Diagonal Decomposition — b=e D_1 class
- `[219]` Fibonacci Resonance — order-1 recurrence on mod-9

Same algebraic family as `[219]`: `Q(√5) = ℤ[φ]`, here realized on integer `(Δb, Δe)` quadratic forms rather than additive Fibonacci recurrences.

## Fixtures

- `fixtures/hsd_pass_table1.json` — verified Table 1 with all 4 φ-ratios and null block
- `fixtures/hsd_fail_wrong_quadratic.json` — negative control (corrupted integer quadratic form)

## Checks

| ID              | Scope |
|-----------------|-------|
| HSD_1           | schema_version matches |
| HSD_TABLE       | 6 rows, integer (b,e), `R·ν` within factor 10 of c |
| HSD_REFERENCE_MATCH | canonical six-scale fixtures match a hardcoded, independently-verified reference table exactly |
| HSD_FIXED_D     | all 6 rows satisfy `b + e = const` within ±1 decade |
| HSD_SEGMENTS    | declared segments reproduce `|Δb|² + |Δe|²` from table rows (exact int) |
| HSD_PHI_RATIOS  | all 4 ratios within 7% of `{φ², 1/φ²}` in log-sense |
| HSD_QUADRATIC   | declared ratios match the exact integer quadratic quotient |
| HSD_NULL        | null-test block well-formed and `observed_stat < null_5pct` |
| HSD_SRC         | source attribution references Haramein 2008 + Will Dale |
| HSD_WITNESS     | ≥ 3 witnesses including `fixed_d`, `phi_ratio`, `null` |
| HSD_F           | fail_ledger well-formed |

## What this cert does NOT claim

- No endorsement of Haramein's physical derivations (Schwarzschild-proton, holographic mass, structured vacuum). Those live in the SVP/RSF semantic layer and are separately certifiable.
- No claim that integer-decade separations are **natural** rather than a rounding choice by the paper's authors. A follow-up cert could test unrounded physical values.
- No claim that R·ν = c is QA-derived. It follows from classical GR/Schwarzschild; QA merely diagnoses the integer-quadratic-form structure of Haramein's 6 chosen scale points on that line.

## Cross-references

- Foundation methodology: [`docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md`](../theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md)
- Syntax/semantics split: [`docs/theory/QA_SYNTAX_SVP_SEMANTICS.md`](../theory/QA_SYNTAX_SVP_SEMANTICS.md)
- Companion cert on Q(√5): `[219]` QA Fibonacci Resonance
- Companion cert on diagonal decomposition: `[217]` QA Fuller VE Diagonal Decomposition
- Primary PDF: `Documents/haramein_rsf/scale_unification_2008.pdf`
- MEMORY reference: `feedback_map_best_to_qa.md`, `feedback_primary_sources_vs_consensus.md`

## Correction history

- **v0 (2026-04-13, superseded):** initial null test interpreted "distance" as log-spacing along the line; got p ≈ 0.35 and provisionally rejected the φ claim. That was an unfounded assumption about Fig 2b's geometry.
- **v1 (2026-04-13, current):** decoded Figure 2b directly; segments are 2D Euclidean lengths on the `(log R, log ν)` plot; four structural ratios verified against Table 1 integer exponents; null p < 5×10⁻⁶.

## Verification Note (2026-07-04)

Independently read the actual primary-source PDF (page 5, "Table 1") to
check the fixture's transcription — the validator previously only checked
the fixture's *own* internal arithmetic consistency, never against an
independent ground truth. Found:

- **All 6 `(b, e)` pairs match the scanned table exactly** (Big Bang
  (-33,43), Atomic (-8,18), Stellar (6,5), G1 (8,2), G2 (12,-2), Universe
  (28,-17)) — and all 6 mass exponents match exactly too.
- **One inert-field transcription slip**: Big Bang's `velocity_exp_log10_cms`
  was listed as `10`, but the real table shows `11`. This field is never
  read by any check (only `b`, `e`, `name`, and — now — `mass_exp_log10_g`
  are used), so it did not affect any pass/fail outcome. Fixed anyway for
  transcription accuracy.
- **The φ-ratio claim is a faithful echo of the paper's own text**: page 6
  explicitly states the segment distances "yield a very close approximation
  to the familiar Φ(phi) ratio... and its inverse," and all four
  `paper_value` numbers cited in the fixture (1.612, 1.629, 0.639, 0.637)
  appear directly in the paper's own Figure 2b. This is the paper making
  the golden-ratio observation itself, not a QA-invented projection onto
  unrelated data — consistent with this cert's own "what this does NOT
  claim" scoping (no endorsement of Haramein's broader physical claims,
  only the integer-quadratic-form structure of the 6 chosen points).

Added `HSD_REFERENCE_MATCH`: a hardcoded, independently-verified reference
table, checked exactly whenever a fixture's row names match the canonical
six. Confirmed it both passes the real data and catches a deliberately
corrupted entry.
