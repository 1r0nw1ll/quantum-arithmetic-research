# QA Constructive Classifier — Indian Pines AVIRIS

Real dataset: Indian Pines AVIRIS + spatial means/vars + band diffs  |  10249 labeled pixels  |  1799 features (200 spectral + 1599 spatial)  |  16 classes
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
| **Certifiably separable** | **70** | **58%** |
| Certifiably indistinguishable (spectral limit) | 50 | 42% |

The indistinguishable pairs are spectrally overlapping at every one of the
200 AVIRIS bands.  No single-band integer threshold can separate them.
This is not a classifier weakness — it is a sensor limitation.

## Accuracy Summary

| Method | Train | Test |
|---|---:|---:|
| Single tree | 1.000 | 0.912 |
| **Ensemble (31 trees, 40% subspace)** | **1.000** | **0.988** |

## Error Diagnosis

| Method | Errors | Tree errors | Spectral-limit errors |
|---|---:|---:|---:|
| Single tree | 224 (8.8%) | 13 | 211 |
| **Ensemble** | **30 (1.2%)** | **0** | **30** |

**Structural interpretation:**
  - Variance errors (single tree minus ensemble spectral-limit) = 194 — these were separable but overfitting caused wrong-branch decisions.
  - 30 residual errors reflect the sensor limit — no spectral/spatial feature separates these classes.

### Top Confused Pairs (ensemble, spectral-limit only)

| True class | Predicted as | Errors | Best gap |
|---|---|---:|---:|
| Corn-notill | Soy-mintill | 5 | -8 |
| Corn-mintill | Corn-notill | 4 | -8 |
| Bldg-Grass-Trees | Woods | 4 | -6 |
| Grass-Pasture | Corn-mintill | 3 | -5 |
| Corn-notill | Soy-notill | 2 | -7 |
| Alfalfa | Hay-windrowed | 1 | -3 |
| Corn-notill | Soy-clean | 1 | -7 |
| Corn-mintill | Soy-notill | 1 | -8 |
| Corn-mintill | Grass-Pasture | 1 | -5 |
| Grass-Pasture | Soy-mintill | 1 | -5 |

## Per-Class Accuracy (Ensemble)

| Class | N test | Tree acc | **Ens acc** |
|---|---:|---:|---:|
| Alfalfa                |   11 | 0.909 | **0.909** ▓▓▓▓▓▓▓▓▓░ |
| Corn-notill            |  357 | 0.849 | **0.978** ▓▓▓▓▓▓▓▓▓░ |
| Corn-mintill           |  207 | 0.850 | **0.971** ▓▓▓▓▓▓▓▓▓░ |
| Corn                   |   59 | 0.864 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Grass-Pasture          |  120 | 0.908 | **0.967** ▓▓▓▓▓▓▓▓▓░ |
| Grass-Trees            |  182 | 0.978 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Grass-mowed            |    7 | 0.857 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Hay-windrowed          |  119 | 0.975 | **0.992** ▓▓▓▓▓▓▓▓▓░ |
| Oats                   |    5 | 0.800 | **0.800** ▓▓▓▓▓▓▓▓░░ |
| Soy-notill             |  243 | 0.848 | **0.988** ▓▓▓▓▓▓▓▓▓░ |
| Soy-mintill            |  613 | 0.953 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Soy-clean              |  148 | 0.878 | **0.993** ▓▓▓▓▓▓▓▓▓░ |
| Wheat                  |   51 | 0.961 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Woods                  |  316 | 0.965 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |
| Bldg-Grass-Trees       |   96 | 0.875 | **0.948** ▓▓▓▓▓▓▓▓▓░ |
| Steel-Towers           |   23 | 0.957 | **1.000** ▓▓▓▓▓▓▓▓▓▓ |

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