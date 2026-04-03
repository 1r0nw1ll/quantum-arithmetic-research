---
layout: default
title: Certificate Families
---

# Certificate Families

Every empirical claim in the QA project is backed by a **machine-verifiable certificate**: JSON schema, Python validator, witness data, and human-readable documentation.

**Current count: 186 families** | All pass CI: `python qa_meta_validator.py`

---

## How Certificates Work

Each family directory contains:

```
qa_<name>_cert_v1/
  qa_<name>_cert.v1.json       # Schema + witnesses
  qa_<name>_cert_validate.py   # Validator (deterministic, no network)
  mapping_protocol.json        # QA mapping protocol (Gate 0)
```

The meta-validator checks every family on every commit. CI blocks if any family fails.

**Claim tiers** (from [QA_CLAIM_TIERS](specs/QA_CLAIM_TIERS)):
- **Tier 1**: Structural (algebraic identity, verified by construction)
- **Tier 2**: Statistical (p < 0.05, pre-registered where possible)
- **Tier 3**: Cross-validated (partial correlations beyond best conventional predictor)

---

## Families by Domain

### Mathematical Foundations
- Pythagorean triples and chromogeometry
- 16 QA identities and Koenig series
- E8 alignment and orbit classification
- Rational trigonometry type system
- Conic discriminant and spread-period correspondence

### Signal Processing & Time Series
- EEG seizure detection (CHB-MIT, 10 patients)
- EMG pathology classification
- Audio orbit transition rates
- Cardiac arrhythmia detection

### Geophysics & Climate
- Climate teleconnection (ENSO orbit classification)
- ERA5 atmospheric reanalysis
- WGS-84 ellipsoid QA encoding
- Seismic control analysis

### Archaeogeometry & Geodesy
- Megalithic Yard confirmation (Thom 1962 data)
- Fibonacci resonance in orbital mechanics
- Celestial navigation
- Dead reckoning and inertial navigation

### Graph Theory
- H-null modularity model
- QA feature maps (qa21/qa27/qa83)
- Community detection benchmarks

### Infrastructure & Tooling
- Mapping protocol (Gate 0)
- Surrogate methodology
- Cross-domain invariance
- Observer projection compliance

---

## Browse All Families

Documentation for individual families: [`docs/families/`](families/)

Validator source: [`qa_alphageometry_ptolemy/`](https://github.com/1r0nw1ll/quantum-arithmetic-research/tree/main/qa_alphageometry_ptolemy)

Run the validator:
```bash
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
# 186/186 PASS
```

---

[Back to home](/)
