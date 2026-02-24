# Family [77]: QA Neighborhood Sufficiency Cert

**Schema version:** QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1
**Directory:** `qa_neighborhood_sufficiency_cert_v1/`
**Added:** 2026-02-23

## Purpose

Encodes the empirical **Neighborhood Sufficiency Principle**: a compact patch-statistic
representation computed over a small spatial window (5×5–7×7) dominates per-pixel
spectral transforms for hyperspectral and multimodal land-cover classification.

Confirmed across three benchmarks: Houston multimodal (+3.0pp OA), Indian Pines
(+10.3pp OA, +16.1pp AA), and PaviaU (+3.9pp OA). Performance saturates at a
scene-dependent minimal radius r*; larger windows introduce boundary contamination.

## QA Concepts

- **Generator dominance:** G_patch(r*) ⊳ G_spec
- **Bounded reachability:** ∃ r* such that ∀ r ≥ r*, Acc(G_patch(r)) ≈ Acc(G_patch(r*))
- **Minimal sufficient radius:** r* tracks scene spatial homogeneity scale

## Gates

| Gate | Description |
|------|-------------|
| gate_1_schema_validity | JSON schema conformance |
| gate_2_canonical_hash | SHA-256 digest integrity |
| gate_3_dominance | patch[r*] OA > spec OA |
| gate_4_monotonicity | OA increases monotonically up to r* |
| gate_5_plateau | OA within tolerance for r > r* |

## Failure Modes

- `NOT_DOMINANT` — patch does not beat spectral baseline
- `NO_PLATEAU` — performance degrades significantly beyond r*
- `BOUNDARY_CONTAMINATION` — stated failure mode when window exceeds class coherence scale
- `INSUFFICIENT_CONTEXT` — window too small to stabilize within-class variance

## Fixtures

| File | Expected | Description |
|------|----------|-------------|
| `valid_houston.json` | PASS | Houston multimodal, r*=5, 14D patch-only |
| `valid_indian_pines.json` | PASS | Indian Pines, r*=7, 10D patch-only |
| `invalid_not_dominant.json` | FAIL gate_3 | Spec OA exceeds patch OA |
| `invalid_no_plateau.json` | FAIL gate_5 | Large drop beyond r* |
| `invalid_digest_mismatch.json` | FAIL gate_2 | Wrong canonical SHA-256 |

## Empirical Results Summary

| Dataset | r* | Patch OA | Spec OA | ΔOA | ΔAA |
|---------|----|---------|---------|-----|-----|
| Houston multimodal | 5×5 | 99.88% | 96.34% | +3.5pp | — |
| Indian Pines | 7×7 | 81.00% | 70.66% | +10.3pp | +16.1pp |
| PaviaU | 7×7 | 91.98% | 88.07% | +3.9pp | +6.4pp |
