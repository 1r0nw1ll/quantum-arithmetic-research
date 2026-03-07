# Family [100] — QA E8 Alignment Audit Cert

**Cert root:** `qa_e8_alignment_audit_cert_v1/`  
**Validator:** `qa_e8_alignment_audit_cert_v1/validator.py --self-test`  
**Schema:** `QA_E8_ALIGNMENT_AUDIT_CERT.v1.schema.json`

## What it certifies

A full-population audit of QA orbit state alignment with the E8 root
system under a fixed canonical embedding. The decision rule is
**pre-registered** — the threshold criteria are frozen before any results
are inspected, making the outcome publishable either way.

## Canonical embedding

```
v8 = (b, e, d, a, b, e, d, a) / ||·||
```

This is fixed in the cert and cannot vary between runs.

## Pre-registered decision rule

**STRUCTURAL** (all four must hold):
- `median_max_cosine > 0.85`
- `mean_within_orbit_std_12 < 0.05`
- `mean_orbit_persistence_12 > 0.50`
- `gap_vs_random > 0.10` ← *decisive criterion*

**INCIDENTAL** otherwise.

## Mod-9 full population result (81 states, 9 orbits)

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| median_max_cosine | 0.9113 | > 0.85 | YES |
| mean_within_orbit_std_12 | 0.0426 | < 0.05 | YES |
| mean_orbit_persistence_12 | 0.833 | > 0.50 | YES |
| gap_vs_random | **−0.019** | > 0.10 | **NO** |

**Verdict: INCIDENTAL**

Random baseline (n=2000, seed=42): median = 0.9304  
QA states: median = 0.9113  
**Random vectors score higher than QA states.**

## Interpretation

The high absolute cosines (~0.91) are a **projection artifact** of the
`(v,v)/norm` symmetric embedding: any 4-tuple from [1,9] produces
comparable or higher E8 alignment. The orbit persistence (0.83) is real —
successive orbit states maintain similar nearest-root identity — but this
reflects orbit geometric coherence within the embedding, not structural
alignment with E8 specifically.

**The E8 hypothesis is definitively ruled out for this embedding.** If a
different embedding (e.g. using full generator algebra, or E8 Gosset
coordinates) is proposed, a new audit cert should be created with its own
pre-registered rule.

## Three gates

| Gate | Check |
|------|-------|
| A | Enumerate full orbit population from modulus; verify `total_states`, `orbit_count` |
| B | Compute embeddings + cosine to all 240 E8 roots; verify median, mean, std, 12-cycle stats, random baseline |
| C | Apply pre-registered rule; verify `claimed.verdict` matches |
