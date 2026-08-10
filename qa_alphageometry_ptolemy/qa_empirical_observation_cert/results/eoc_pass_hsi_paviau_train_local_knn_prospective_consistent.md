# PaviaU Train-Local KNN Prospective Correction Cert

Certificate: `qa.cert.empirical.hsi_paviau_train_local_knn_prospective_consistent.v1`

Verdict: `CONSISTENT`  
Result: `PASS`

## Claim

On the fixed PaviaU split and cached ensemble boundary, a prospective train-label-only local KNN-vote QA correction lane selected 36 coordinate-bounded exact integer rules without held-out residual labels. Independent replay reduced held-out ensemble errors from 55 to 0 with 0 harmed correct rows.

## Key Numbers

| Measure | Value |
|---|---:|
| Train rows | 32086 |
| Test rows | 10690 |
| Baseline ensemble errors | 55 |
| Corrected errors | 0 |
| Fixed ensemble errors | 55 |
| Harmed correct rows | 0 |
| Rows touched | 55 |
| Added train-local KNN rules | 36 |
| Total merged rules | 861 |

## Gate

The deployment gate passed with:

```bash
python3 tools/qa_hsi_deployment_gate.py --dataset paviau --rules results/qa_hsi_paviau_accepted_corrected_model_spatial_spectral2_plus_train_local_knn_compact_2026_06_09.json --cache results/qa_hsi_paviau_prediction_cache_cache_oob_spatial_knn_final3_2026_06_09.npz --min-fixed 55 --require-no-heldout-selection
```

The retrospective local-KNN artifact correctly fails the same prospective gate because it contains `uses_heldout_labels_for_selection=true`.

## Limits

This is empirical for one fixed PaviaU split and one cached prediction boundary. It does not prove universal 100% HSI classification. Cross-seed and Salinas transfer remain follow-up tests.
