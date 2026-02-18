# qa_kona_ebm_qa_native_orbit_reg_v1

QA orbit-coherence regularizer added to QA-native RBM (family [64]).

## What this adds over family [63]

Family [63] (`qa_kona_ebm_qa_native_v1`) uses 81 hidden units indexed by
QA states (b,e) mod 9 but applies no structural bias during training.
This family injects an orbit-coherence regularizer that explicitly pushes
orbit-mate weight vectors toward their shared mean.

## Regularizer

    R(W) = sum_O sum_{i in O} ||W_i - mu_O||^2

where O in {COSMOS (72 units), SATELLITE (8 units)}, and
mu_O = W[idxs].mean(axis=0) is the centroid weight vector for orbit O.

Gradient applied per batch (SINGULARITY skipped -- only 1 unit):

    dW[idxs] -= lr * lambda_orbit * 2.0 * (W[idxs] - mu_orbit)

## Non-tautological evaluation

Regularizer directly constrains WEIGHTS to be orbit-coherent. The cert
measures ACTIVATION coherence (mean pairwise Pearson correlation of hidden
unit activations on held-out data) via the permutation gap test inherited
from [63]. These are distinct: weight coherence does not guarantee
activation coherence under arbitrary MNIST inputs.

## Hypothesis

Orbit inductive bias in weights -> higher activation coherence.
z-score (c_real vs permutation null) should be consistently positive
across seeds, especially for COSMOS (72 units, large orbit).

## Failure taxonomy

| status                          | meaning                                      |
|---------------------------------|----------------------------------------------|
| CONVERGED                       | energy decreased, training normal            |
| STALLED                         | energy did not decrease                      |
| GRADIENT_EXPLOSION              | CD gradient exceeded threshold 1000          |
| REGULARIZER_NUMERIC_INSTABILITY | NaN/Inf in W, b_vis, or c_hid after update  |

## CLI

Generate valid cert:
    python generate_cert.py --n-samples 1000 --n-epochs 5 --lr 0.01 \
        --lambda-orbit 1e-3 --seed 42 --cert-id kona_ebm_orbit_reg_stable_001

Validate cert:
    python validator.py fixtures/valid_orbit_reg_stable_run.json
    python validator.py fixtures/valid_orbit_reg_stable_run.json --json

Run self-test (all 3 fixtures):
    python validator.py --self-test

Multi-seed report:
    python generate_multi_seed_report.py --n-epochs 20 --lambda-orbit 1e-3

## Fixtures

  valid_orbit_reg_stable_run.json      -- PASS  (lambda=1e-3, seed=42, 5 epochs)
  invalid_regularizer_instability.json -- FAIL: REGULARIZER_NUMERIC_INSTABILITY
  invalid_trace_mismatch.json          -- FAIL: TRACE_HASH_MISMATCH (Gate 4)

## Related families

  [63] qa_kona_ebm_qa_native_v1: negative control -- no regularizer
