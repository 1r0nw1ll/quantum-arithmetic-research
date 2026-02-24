# Family [79]: QA Locality Regime Separator Cert

**Schema:** `QA_LOCALITY_REGIME_SEP_CERT.v1`
**Root:** `qa_locality_regime_sep_cert_v1/`
**Status:** Active (v1.0)

## Purpose

Theorem-level certificate classifying a hyperspectral scene as **DOMINANT** (variance reduction wins) or **BOUNDARY** (boundary bias dominates), bridging the empirical cert families [77]/[78] to the variance-bias decomposition:

```
err(r) ≈ V(r) + B(r)
V(r) ∝ σ²/(2r+1)²      (decreasing — variance reduction)
B(r) ∝ adj_4·r·‖Δμ‖²   (increasing — boundary contamination)
```

**DOMINANT**: ∃ r s.t. ΔOA(r) > 0 (variance wins for at least one radius)
**BOUNDARY**: ∀ r, ΔOA(r) ≤ 0 (boundary bias dominates all tested radii)

## Validation Gates

| Gate | Name | Description |
|------|------|-------------|
| 1 | Schema validity | JSON validates against `schema.json` |
| 2 | Canonical hash | `canonical_sha256` in digests must match recomputed SHA-256 of cert with digest zeroed |
| 3 | Delta evidence recompute | Recomputes `all_deltas_nonpositive`, `any_delta_positive`, `max_delta_pp`, `min_delta_pp` from `delta_oa_by_radius`; must match declared values in `regime_evidence` |
| 4 | Regime declaration | DOMINANT requires `any_delta_positive=True`; BOUNDARY requires `all_deltas_nonpositive=True`; mismatch → REGIME_INCONSISTENT |
| 5 | Regime consistent flag | `regime_evidence.regime_consistent` must be `True` |

## Failure Modes

- `REGIME_INCONSISTENT` — declared regime contradicts computed delta sign structure (Gate 4)
- `EVIDENCE_MISMATCH` — declared evidence fields don't match recomputed values (Gate 3)
- `SCHEMA_INVALID` — missing required fields or wrong types (Gate 1)
- `HASH_MISMATCH` — canonical_sha256 doesn't match recomputed value (Gate 2)

## Fixtures

| File | Regime | Expected | Notes |
|------|--------|----------|-------|
| `valid_salinas_dominant.json` | DOMINANT | PASS | Salinas: all ΔOA > 0 (+0.56 to +4.29pp) |
| `valid_ksc_boundary.json` | BOUNDARY | PASS | KSC: all ΔOA ≤ 0 (−0.98 to −7.42pp) |
| `invalid_regime_inconsistent.json` | BOUNDARY | FAIL gate_4 | Salinas deltas (all positive) but declares BOUNDARY |
| `invalid_digest_mismatch.json` | BOUNDARY | FAIL gate_2 | KSC values with `deadbeef...` hash |

## Empirical Data (Locality Dominance Study)

| Dataset | adj_rate_4 | r* | Regime | max ΔOA | min ΔOA |
|---------|-----------|-----|--------|---------|---------|
| Salinas | 0.12 | 7 | DOMINANT | +4.29pp | +0.56pp |
| KSC | 0.246 | 7 | BOUNDARY | −0.98pp | −7.42pp |

KSC's high adjacency rate (24.6% vs Salinas 12%) means boundary bias dominates variance reduction across all tested radii — captured by the BOUNDARY regime classification.

## Relationship to Other Families

- **[77] Neighborhood Sufficiency Cert**: certifies dominance/failure for a single dataset+radius (empirical, patch-level)
- **[78] Locality Boundary Cert**: certifies the structural explanation for failure (fragmentation scale, adj rate)
- **[79] Locality Regime Separator** (this family): theorem-level classifier that synthesizes [77]+[78] evidence into a single DOMINANT/BOUNDARY verdict backed by variance-bias theory

## Links

- Schema: `qa_locality_regime_sep_cert_v1/schema.json`
- Validator: `qa_locality_regime_sep_cert_v1/validator.py`
- Mapping protocol: `qa_locality_regime_sep_cert_v1/mapping_protocol_ref.json`
- Theory note: `docs/theory/QA_THEOREM_LOCALITY_BOUNDARY_VARIANCE_BIAS.md`
- Related paper: `papers/in-progress/locality-dominance/`
