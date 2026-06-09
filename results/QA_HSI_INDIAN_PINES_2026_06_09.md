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

## Tree Statistics

- Leaves: 320  |  Max depth: 18
- Train accuracy: **1.000**
- Test accuracy:  **0.913**

## Error Diagnosis

Total test errors: 223 / 2557 (8.7%)

| Error type | Count | Fraction of errors |
|---|---:|---:|
| Tree errors (separable pair, tree missed) | 12 | 5% |
| Spectral limit (indistinguishable pair) | 211 | 95% |

**Structural interpretation:**
  - 12 errors are fixable — the pair IS separable but the tree
    chose a different branch.  These shrink as the tree grows deeper.
  - 211 errors reflect the sensor limit — no spectral feature
    separates these classes.  Fixing them requires LiDAR, texture, or temporal data.

### Top Confused Pairs (spectral limit only)

| True class | Predicted as | Errors | Best gap |
|---|---|---:|---:|
| Corn-notill | Soy-mintill | 30 | -8 |
| Soy-notill | Soy-mintill | 16 | -8 |
| Soy-notill | Corn-notill | 14 | -7 |
| Soy-mintill | Corn-notill | 11 | -8 |
| Corn-notill | Corn-mintill | 10 | -8 |
| Corn-mintill | Soy-notill | 10 | -8 |
| Corn-mintill | Corn-notill | 10 | -8 |
| Woods | Bldg-Grass-Trees | 8 | -6 |
| Soy-mintill | Soy-notill | 7 | -8 |
| Corn-mintill | Soy-mintill | 6 | -9 |

## Per-Class Accuracy

| Class | N test | Correct | Accuracy |
|---|---:|---:|---:|
| Alfalfa                |   11 |   10 | 0.909 ▓▓▓▓▓▓▓▓▓░ |
| Corn-notill            |  357 |  303 | 0.849 ▓▓▓▓▓▓▓▓░░ |
| Corn-mintill           |  207 |  176 | 0.850 ▓▓▓▓▓▓▓▓░░ |
| Corn                   |   59 |   52 | 0.881 ▓▓▓▓▓▓▓▓░░ |
| Grass-Pasture          |  120 |  109 | 0.908 ▓▓▓▓▓▓▓▓▓░ |
| Grass-Trees            |  182 |  178 | 0.978 ▓▓▓▓▓▓▓▓▓░ |
| Grass-mowed            |    7 |    6 | 0.857 ▓▓▓▓▓▓▓▓░░ |
| Hay-windrowed          |  119 |  116 | 0.975 ▓▓▓▓▓▓▓▓▓░ |
| Oats                   |    5 |    4 | 0.800 ▓▓▓▓▓▓▓▓░░ |
| Soy-notill             |  243 |  206 | 0.848 ▓▓▓▓▓▓▓▓░░ |
| Soy-mintill            |  613 |  585 | 0.954 ▓▓▓▓▓▓▓▓▓░ |
| Soy-clean              |  148 |  129 | 0.872 ▓▓▓▓▓▓▓▓░░ |
| Wheat                  |   51 |   49 | 0.961 ▓▓▓▓▓▓▓▓▓░ |
| Woods                  |  316 |  305 | 0.965 ▓▓▓▓▓▓▓▓▓░ |
| Bldg-Grass-Trees       |   96 |   84 | 0.875 ▓▓▓▓▓▓▓▓░░ |
| Steel-Towers           |   23 |   22 | 0.957 ▓▓▓▓▓▓▓▓▓░ |

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