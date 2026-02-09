# Level 3 Recompute Validation: Generalization Certificates

## What this is

Level 3 recompute validation proves that a third party can reproduce every
witness value in a generalization certificate from raw data and weight
matrices alone. It is the strongest validation level — beyond schema
checking (Level 1) and internal consistency (Level 2).

The script trains a real MLP on real MNIST data using pure numpy (no
PyTorch), extracts actual spectral norms via SVD, computes actual pairwise
distances from pixel data, emits a certificate, then independently
recomputes every witness and checks they match.

## Why it matters

Levels 1-2 check internal coherence: "does the certificate agree with
itself?" Level 3 checks external validity: "does the certificate agree
with the raw data?" This is the difference between a self-consistent
fiction and a verifiable claim.

The key property: given only the training set and weight matrices, an
independent party can reproduce every number in the certificate and the
failure classification. Nothing is taken on faith.

## How to run

```bash
# Full mode (~22s on i7 M620, 2000 train samples, 50 epochs)
python3 qa_alphageometry_ptolemy/level3_recompute_validation.py

# CI mode (~3s, 512 train samples, 15 epochs)
python3 qa_alphageometry_ptolemy/level3_recompute_validation.py --ci
```

## CI knobs (environment variables)

| Variable | Default (full) | Default (--ci) | Description |
|----------|---------------|----------------|-------------|
| `QA_L3_NTRAIN` | 2000 | 512 | Training samples |
| `QA_L3_NTEST` | 500 | 128 | Test samples |
| `QA_L3_EPOCHS` | 50 | 15 | Training epochs |

## Expected output

**CI mode:** Single line indicating pass/fail.

```
[PASS] Level 3 recompute (n=512, epochs=15) L2=6/6 L3=3/3
```

**Full mode:** Detailed report with measurements, followed by per-hook
recompute results and artifacts written to
`qa_alphageometry_ptolemy/external_validation_certs/`.

## Hooks verified

| Hook | What it recomputes | Data source |
|------|--------------------|-------------|
| MetricGeometryHook | Pairwise distances, data hash | Raw MNIST pixels |
| OperatorNormHook | Spectral norms via SVD | Trained weight matrices |
| GeneralizationBoundHook | Bound formula from witnesses | Certificate fields |

## Determinism notes

- Random seed fixed: `np.random.seed(42)`
- Same dataset used for certificate emission and recompute (no subsample
  mismatch)
- SVD singular values are invariant to transpose, so weight matrix
  orientation doesn't affect spectral norm measurement
- All scalar values stored as exact `Fraction` strings in the certificate

## Known result

Norm-based generalization bounds (Bartlett et al. 2017, Neyshabur et al.
2018) are **vacuous** for most trained networks — including a simple
2-layer MLP on MNIST. The spectral norm product across layers typically
exceeds what 1/sqrt(n) can compensate. The framework correctly classifies
this as `bound_vacuous`, which matches the published literature.

This is not a bug — it validates that the framework says the right thing
about real data, including correctly reporting when theoretical bounds
offer no guarantees.

## Hardware requirements

- ~32MB for pairwise distance matrix (2000 samples)
- ~2MB for model weights (784x128 + 128x10)
- No GPU, no PyTorch — pure numpy + scipy
- Tested on: Intel i7 M620 @ 2.67GHz, 7.6GB RAM
