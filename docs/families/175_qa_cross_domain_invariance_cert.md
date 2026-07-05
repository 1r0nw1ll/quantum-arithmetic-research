# Family [175] QA_CROSS_DOMAIN_INVARIANCE_CERT.v1

## One-line summary

3 structural invariants (surrogate survival, independent information, domain-general architecture) across 7 Tier 3 domains.

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_cross_domain_invariance_cert_v1/qa_cross_domain_invariance_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-05)

Found a stale domain-list inconsistency (see [173]'s verification note
for the full cross-cert comparison): this cert's three `domains_confirmed`
lists included EMG but were missing **ERA5** (cert [172], confirmed via
the corrected v2 surrogate methodology this same audit cycle), while
sibling cert [173]'s list had the opposite gap. Added ERA5 to all three
invariant lists. Specifically re-verified `invariant_domain_general` for
ERA5 rather than assuming it: read `49_forecast_coherence_surrogates.py`
directly and confirmed it uses the identical topographic (k-means) → QA
orbit (`qa_mod`) → T-operator → QCI architecture as the other domains,
with no domain-specific tuning beyond the standard per-domain parameters
(MODULUS, N_CLUSTERS) every domain already varies. Also fixed the stale
doubled "cert_cert" path above.
