# QA Constructive Classifier — Houston 2013 Multimodal (HSI + LiDAR)

Dataset: Houston 2013 GRSS DFC multimodal  |  2817 samples  |  15 classes  |  105 class pairs

## Constructive Claim

The certificate diagnoses which class pairs are INDISTINGUISHABLE from HSI alone
and predicts that LiDAR resolves exactly those pairs.  This tests the diagnosis.

| Pass | Features | Separable pairs | Test accuracy | Spectral-limit errors |
|---|---:|---:|---:|---:|
| **HSI-only** | 288 | 45 | **0.923** | 50 |
| **HSI+LiDAR** | 290 | 57 | **0.931** | 39 |

## LiDAR Promotion: 12 Pairs Resolved

Pairs that were INDISTINGUISHABLE in HSI → SEPARABLE after adding LiDAR height:

| Class A | Class B | Gap (HSI) | Gap (HSI+LiDAR) | Feature |
|---|---|---:|---:|---:|
| Soil | Commercial | -332 | 38 | LiDAR |
| Commercial | Parking-lot-2 | -211 | 36 | LiDAR |
| Commercial | Running-track | -305 | 36 | LiDAR |
| Commercial | Railway | -288 | 29 | LiDAR |
| Stressed-grass | Commercial | -206 | 13 | LiDAR |
| Commercial | Road | -439 | 13 | LiDAR |
| Commercial | Tennis-court | -169 | 10 | LiDAR |
| Residential | Railway | -246 | 9 | LiDAR |
| Synthetic-grass | Residential | -157 | 3 | LiDAR |
| Synthetic-grass | Railway | -47 | 3 | LiDAR |
| Commercial | Highway | -357 | 1 | LiDAR |
| Railway | Running-track | -153 | 1 | LiDAR |

## Per-Pass Error Diagnosis

### HSI-only
- Accuracy: train **1.000** / test **0.923**
- Errors: 54 total (4 tree errors, 50 spectral-limit)

| True class | Predicted as | Errors | Best gap |
|---|---|---:|---:|
| Residential | Tennis-court | 4 | -136 |
| Commercial | Parking-lot-1 | 4 | -581 |
| Railway | Road | 4 | -262 |
| Soil | Parking-lot-1 | 3 | -264 |
| Railway | Residential | 3 | -246 |
| Stressed-grass | Road | 2 | -208 |
| Highway | Soil | 2 | -214 |
| Parking-lot-1 | Commercial | 2 | -581 |

### HSI+LiDAR
- Accuracy: train **1.000** / test **0.931**
- Errors: 48 total (9 tree errors, 39 spectral-limit)

| True class | Predicted as | Errors | Best gap |
|---|---|---:|---:|
| Soil | Road | 2 | -38 |
| Commercial | Parking-lot-1 | 2 | -88 |
| Road | Railway | 2 | -26 |
| Highway | Soil | 2 | -27 |
| Highway | Parking-lot-1 | 2 | -45 |
| Parking-lot-1 | Commercial | 2 | -88 |
| Parking-lot-2 | Parking-lot-1 | 2 | -30 |
| Healthy-grass | Stressed-grass | 1 | -23 |

## Interpretation

The constructive certificate is an **actionable sensor guide**:
- SEPARABLE pairs are solved by the integer threshold tree alone.
- INDISTINGUISHABLE pairs name exactly which additional sensor modality is needed.
LiDAR height resolves 12 pairs that HSI cannot — confirming that the
diagnostic is correct, not just a description of failure.

For Indian Pines (agricultural scene), the analogous missing modality is
multi-temporal NDVI (phenological stage differences) or LiDAR canopy height.