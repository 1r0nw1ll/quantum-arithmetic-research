# Family [78]: QA Locality Boundary Cert

**Schema version:** QA_LOCALITY_BOUNDARY_CERT.v1
**Directory:** `qa_locality_boundary_cert_v1/`
**Added:** 2026-02-24

## Purpose

Certifies the **Boundary Condition** for locality-based generators: when patch-only generators
fail to dominate the spectral baseline, the failure is explained by spatial fragmentation —
the scene's characteristic fragmentation scale is smaller than the minimal radius r*.

This is the complement cert to Family [77] (Neighborhood Sufficiency). Together they provide
full coverage: [77] certifies *when* locality dominates; [78] certifies *why* locality fails.

The primary empirical case is the Kennedy Space Center (KSC) wetland dataset, where thin
elongated class regions (marshes, cypress swamp, hardwood swamp, scrub) have widths
narrower than 7 pixels, causing all patch windows to straddle class boundaries.

## QA Concepts

- **Boundary contamination:** patch windows exceeding fragmentation scale sample multiple classes
- **Negative delta:** ΔOA = patch[r] − spec < 0 for all tested radii r
- **Structural explanation:** fragmentation_scale < r* is the causal mechanism

## Gates

| Gate | Description |
|------|-------------|
| gate_1_schema_validity | JSON schema conformance |
| gate_2_canonical_hash | SHA-256 digest integrity |
| gate_3_failure_curve | All patch[r] deltas ≤ 0 — patch never beats spec at any tested radius |
| gate_4_delta_flag | `all_deltas_nonpositive` flag matches computed reality |
| gate_5_fragmentation_explanation | `fragmentation_scale_lt_r_star = true` required (causal explanation present) |

## Failure Modes

- `NOT_A_BOUNDARY_CASE` — at least one patch[r] OA exceeds spec OA (this is a dominant dataset, not a failure case)
- `DELTA_FLAG_MISMATCH` — declared `all_deltas_nonpositive` contradicts computed deltas
- `MISSING_FRAGMENTATION_EXPLANATION` — `fragmentation_scale_lt_r_star = false` (no causal explanation)

## Fixtures

| File | Expected | Description |
|------|----------|-------------|
| `valid_ksc_boundary.json` | PASS | KSC wetland, all radii negative (−7.4pp to −0.98pp), fragmentation_scale < r*=7 |
| `invalid_not_a_boundary_case.json` | FAIL gate_3 | Salinas dominant dataset — patch[r] > spec at all radii |
| `invalid_digest_mismatch.json` | FAIL gate_2 | Wrong canonical SHA-256 (deliberate corruption) |

## Empirical Results: KSC

| Radius | Patch OA | Spec OA | ΔOA |
|--------|---------|---------|-----|
| r=3 | 82.37% | 89.79% | −7.42pp |
| r=5 | 86.33% | 89.79% | −3.46pp |
| r=7 | 88.81% | 89.79% | −0.98pp |

**Boundary geometry proxies (KSC):**
- `fragmentation_proxy` = 0.72 (high — many thin class regions)
- `thin_region_proxy` = 0.61 (high — majority of class pixels within 7px of class boundary)

## Relationship to Family [77]

Family [77] (`QA_NEIGHBORHOOD_SUFFICIENCY_CERT.v1.1`) and Family [78]
(`QA_LOCALITY_BOUNDARY_CERT.v1`) are complementary:

- [77] `DOMINANT` outcome ↔ patch beats spec, Gates 4+5 check monotonicity/plateau
- [77] `FAILS_BOUNDARY_CONTAMINATION` outcome ↔ patch loses to spec (structural failure logged)
- [78] independently certifies the failure mechanism with full failure-curve + geometry documentation

A dataset may hold a [77] `FAILS_BOUNDARY_CONTAMINATION` cert and a [78] cert simultaneously —
the two certs certify different aspects of the same phenomenon.
