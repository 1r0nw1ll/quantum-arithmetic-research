# QA Constructive Classifier — Indian Pines AVIRIS

Real dataset: Indian Pines AVIRIS + spatial means/vars + band diffs  |  10249 labeled pixels  |  3997 features (200 spectral + 3797 spatial)  |  16 classes
Split: 7692 train / 2557 test (25% stratified)

## Core Claim

QA is constructive: every error is a structural diagnosis, not irreducible noise.
For each class pair the tree issues one of two certificates:

| Certificate | Meaning |
|---|---|
| **SEPARABLE** | Integer threshold exists; tree adds a branch; zero errors guaranteed |
| **INDISTINGUISHABLE** | No threshold at any of the 200 bands; errors require additional features |

## Separability Certificates

| | Count | Fraction |
|---|---:|---:|
| Total class pairs | 120 | 100% |
| **Certifiably separable** | **120** | **100%** |
| Certifiably indistinguishable (spectral limit) | 0 | 0% |

The indistinguishable pairs are spectrally overlapping at every one of the
200 AVIRIS bands.  No single-band integer threshold can separate them.
This is not a classifier weakness — it is a sensor limitation.

## Accuracy Summary

| Method | Train | Test |
|---|---:|---:|
| Single tree | 1.000 | 0.953 |
| **Ensemble (201 trees, bagged, 30% subspace)** | **1.000** | **0.996** |

## Error Diagnosis

| Method | Errors | Tree errors | Spectral-limit errors |
|---|---:|---:|---:|
| Single tree | 120 (4.7%) | 120 | 0 |
| **Ensemble** | **11 (0.4%)** | **11** | **0** |

**Structural interpretation:**
  - Variance errors (single tree minus ensemble spectral-limit) = 109 — these were separable but overfitting caused wrong-branch decisions.
  - 0 residual errors reflect the sensor limit — no spectral/spatial feature separates these classes.

### Top Confused Pairs (ensemble, spectral-limit only)

| True class | Predicted as | Errors | Best gap |
|---|---|---:|---:|

## Per-Class Accuracy (Ensemble)

| Class | N test | Tree acc | **Ens acc** |
|---|---:|---:|---:|
| Alfalfa                |   11 | 0.636 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Corn-notill            |  357 | 0.938 | **0.986** ▓▓▓▓▓▓▓▓▓░ |
| Corn-mintill           |  207 | 0.918 | **0.986** ▓▓▓▓▓▓▓▓▓░ |
| Corn                   |   59 | 0.915 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Grass-Pasture          |  120 | 0.967 | **0.992** ▓▓▓▓▓▓▓▓▓░ |
| Grass-Trees            |  182 | 0.995 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Grass-mowed            |    7 | 0.857 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Hay-windrowed          |  119 | 0.966 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Oats                   |    5 | 0.800 | **0.800** ▓▓▓▓▓▓▓▓░░ |
| Soy-notill             |  243 | 0.905 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Soy-mintill            |  613 | 0.976 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Soy-clean              |  148 | 0.926 | **0.993** ▓▓▓▓▓▓▓▓▓░ |
| Wheat                  |   51 | 0.980 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Woods                  |  316 | 0.981 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Bldg-Grass-Trees       |   96 | 0.948 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Steel-Towers           |   23 | 1.000 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |

## Interpretation

The constructive tree achieves high accuracy on classes with distinct
spectral signatures (Woods, Wheat, Hay-windrowed, Steel-Towers) and
low accuracy on spectrally similar classes (Corn variants, Soybean variants).

The standard ML framing would report a single accuracy number and
attribute errors to 'Bayes error' or 'irreducible noise.'

The QA framing gives a certificate for every confused pair:
  - If the pair is certifiably separable → tree error, fixable by growing deeper
  - If the pair is certifiably indistinguishable → sensor limit, fixable by
    adding LiDAR, multi-temporal NDVI, or spatial texture features

The error count is not a failure metric — it is a roadmap for which
additional features are needed to achieve exact classification.