# Family [78]: QA Locality Boundary Cert

**Schema version:** QA_LOCALITY_BOUNDARY_CERT.v1.1 (v1 backward-compat)
**Directory:** `qa_locality_boundary_cert_v1/`
**Added:** 2026-02-24 (v1); **Updated:** 2026-02-24 (v1.1 — Gate 6 adjacency witness)

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

## Gates (v1.1)

| Gate | Description |
|------|-------------|
| gate_1_schema_validity | JSON schema conformance (v1 or v1.1) |
| gate_2_canonical_hash | SHA-256 digest integrity |
| gate_3_failure_curve | All patch[r] deltas ≤ 0 — patch never beats spec at any tested radius |
| gate_4_delta_flag | `all_deltas_nonpositive` flag matches computed reality |
| gate_5_fragmentation_explanation | `fragmentation_scale_lt_r_star = true` required (causal explanation present) |
| gate_6_adjacency_witness | (v1.1 only) Recomputes `adj_rate_4` from embedded `gt_label_grid`, verifies `gt_label_sha256`, checks declared value matches computed within 1e-6. **Skipped** if `adjacency_witness` absent (v1 backward-compat). |

## Failure Modes

- `NOT_A_BOUNDARY_CASE` — at least one patch[r] OA exceeds spec OA (this is a dominant dataset, not a failure case)
- `DELTA_FLAG_MISMATCH` — declared `all_deltas_nonpositive` contradicts computed deltas
- `MISSING_FRAGMENTATION_EXPLANATION` — `fragmentation_scale_lt_r_star = false` (no causal explanation)
- `ADJ_WITNESS_HASH_MISMATCH` — `gt_label_sha256` does not match computed SHA-256 of `gt_label_grid`
- `ADJ_RATE_MISMATCH` — declared `adj_rate_4` differs from value computed from the embedded grid
- `ADJ_GRID_INVALID` — `gt_label_grid` is empty or non-rectangular

## Fixtures

| File | Expected | Description |
|------|----------|-------------|
| `valid_ksc_boundary.json` | PASS | KSC wetland v1 (no adjacency_witness, Gate 6 skipped) |
| `valid_ksc_boundary_v1_1.json` | PASS | KSC wetland v1.1 with adjacency_witness — 10×10 synthetic stripe grid, adj_rate_4=0.5 verified |
| `invalid_not_a_boundary_case.json` | FAIL gate_3 | Salinas dominant dataset — patch[r] > spec at all radii |
| `invalid_digest_mismatch.json` | FAIL gate_2 | Wrong canonical SHA-256 (deliberate corruption) |
| `invalid_adj_rate_wrong.json` | FAIL gate_6 | Declared adj_rate_4=0.99 but grid computes 0.5 → ADJ_RATE_MISMATCH |

## Adjacency Witness Design (v1.1)

The `adjacency_witness` block replaces narrative proxy values with a deterministic, recomputable measurement:

```json
"adjacency_witness": {
  "adj_rate_4": 0.5,
  "gt_label_sha256": "...",
  "gt_label_grid": [[0,0,...],[1,1,...], ...]
}
```

- `adj_rate_4` = (# 4-neighbor edges where labels differ) / (total 4-neighbor labeled edges)
- `gt_label_grid` is an embedded 2D integer array; the validator recomputes adj_rate_4 from it
- `gt_label_sha256` = SHA-256 of `json.dumps(gt_label_grid, sort_keys=True, separators=(',',':'))` — anchors the witness to a specific grid
- For production use: replace the synthetic 10×10 grid with the real scene GT label mask
- v1.2 upgrade path: accept `gt_mask_path` as an alternative to embedded grid for large scenes

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
