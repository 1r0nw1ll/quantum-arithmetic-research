# Family [77]: QA Neighborhood Sufficiency Cert

**Schema version:** QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1.1
**Directory:** `qa_neighborhood_sufficiency_cert_v1/`
**Added:** 2026-02-23 (v1); **Updated:** 2026-02-24 (v1.1 — branching Gate 3, Salinas + KSC fixtures)

## Purpose

Encodes the empirical **Neighborhood Sufficiency Principle**: a compact patch-statistic
representation computed over a small spatial window dominates per-pixel spectral transforms
for hyperspectral and multimodal land-cover classification on *spatially homogeneous* scenes.

v1.1 extends Gate 3 with branching logic to certify both **dominance** and **documented failure**
as first-class outcomes. The KSC wetland dataset is a certified counterexample: boundary
contamination from thin class regions (marshes, scrub) narrower than r* explains the failure.

Confirmed dominant on four benchmarks: Houston (+3.0pp OA), Indian Pines (+10.3pp OA,
+16.1pp AA), PaviaU (+3.9pp OA), Salinas (+4.3pp OA).
KSC is a certified failure case (−1.0pp OA, boundary-contamination explained).

## QA Concepts

- **Generator dominance:** G_patch(r*) ⊳ G_spec
- **Certified failure:** G_spec ⊳ G_patch when fragmentation_scale < r*
- **Bounded reachability:** ∃ r* such that ∀ r ≥ r*, Acc(G_patch(r)) ≈ Acc(G_patch(r*))
- **Minimal sufficient radius:** r* tracks scene spatial homogeneity scale

## Gates (v1.1)

| Gate | Description |
|------|-------------|
| gate_1_schema_validity | JSON schema conformance (v1 or v1.1) |
| gate_2_canonical_hash | SHA-256 digest integrity |
| gate_3_dominance | Branches on `dominance_result`: DOMINANT requires patch[r*] > spec; FAILS_BOUNDARY_CONTAMINATION requires patch[r*] < spec + boundary_metrics present; INCONCLUSIVE passes without delta check |
| gate_4_monotonicity | OA increases monotonically up to r* (DOMINANT only) |
| gate_5_plateau | OA within tolerance for r > r* (DOMINANT only) |

## Failure Modes

- `NOT_DOMINANT` — claims DOMINANT but patch[r*] ≤ spec
- `NO_PLATEAU` — performance degrades significantly beyond r*
- `MISSING_BOUNDARY_METRICS` — FAILS_BOUNDARY_CONTAMINATION declared without boundary_metrics
- `NOT_A_FAILURE_CASE` — FAILS_BOUNDARY_CONTAMINATION declared but patch[r*] ≥ spec
- `BOUNDARY_CONTAMINATION` / `BOUNDARY_CONTAMINATION_DOMINATES` — failure mode annotations
- `INSUFFICIENT_CONTEXT` — window too small to stabilize within-class variance

## Fixtures

| File | Expected | Description |
|------|----------|-------------|
| `valid_houston.json` | PASS | Houston multimodal, r*=5, 14D patch-only (DOMINANT) |
| `valid_indian_pines.json` | PASS | Indian Pines, r*=7, 10D patch-only (DOMINANT) |
| `valid_salinas.json` | PASS | Salinas, r*=7, 10D patch-only, +4.3pp OA (DOMINANT) |
| `valid_ksc_failure.json` | PASS | KSC wetland, r*=7, certified failure −1.0pp (FAILS_BOUNDARY_CONTAMINATION) |
| `invalid_not_dominant.json` | FAIL gate_3 | Claims DOMINANT but patch OA ≤ spec OA |
| `invalid_no_plateau.json` | FAIL gate_5 | Large drop beyond r* (DOMINANT) |
| `invalid_digest_mismatch.json` | FAIL gate_2 | Wrong canonical SHA-256 |
| `invalid_claims_dominant_but_negative_delta.json` | FAIL gate_3 | Claims DOMINANT but uses KSC values (patch < spec) |

## Empirical Results Summary

| Dataset | r* | Patch OA | Spec OA | ΔOA | Notes |
|---------|----|---------|---------|-----|-------|
| Houston multimodal | 5×5 | 99.88% | 96.34% | +3.5pp | Dominant |
| Indian Pines | 7×7 | 81.00% | 70.66% | +10.3pp | Dominant |
| PaviaU | 7×7 | 91.98% | 88.07% | +3.9pp | Dominant |
| Salinas | 7×7 | 97.01% | 92.72% | +4.3pp | Dominant |
| KSC | 7×7 | 88.81% | 89.79% | −1.0pp | Certified failure — fragmentation_scale < r* |
