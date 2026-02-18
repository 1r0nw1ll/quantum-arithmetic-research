# qa_kona_ebm_mnist_v1

Certifies a restricted Boltzmann machine (RBM) training run on MNIST under the
QA invariant contract. Real ML, deterministic trace, typed failure obstructions.

## What this certifies

A binary RBM trained with Contrastive Divergence CD-1 on binarised MNIST digits.
The certificate records per-epoch energy, reconstruction error, and gradient norms,
then verifies the training trace is exactly reproducible from the stated seed.

## CLI commands

```bash
# Validate a certificate (human-readable output)
python validator.py fixtures/valid_stable_run.json

# Validate with JSON output
python validator.py fixtures/valid_stable_run.json --json

# Run self-test (all three fixtures)
python validator.py --self-test

# Generate a new certificate
python generate_cert.py --n-hidden 64 --n-samples 1000 --n-epochs 5 \
    --lr 0.01 --seed 42 --cert-id my_run
```

## Invariants

| Status | Meaning |
|---|---|
| `CONVERGED` | Final free energy < initial free energy |
| `STALLED` | Energy did not decrease over training |
| `GRADIENT_EXPLOSION` | Update norm (lr * grad_norm) exceeded 1000.0 |

## Failure taxonomy

**GRADIENT_EXPLOSION** — the weight update magnitude exceeded the stability
threshold 1000.0 (= lr * Frobenius norm of dW). Training halted. The
`result.invariant_diff` object is populated with `fail_type`, `target_path`,
and `reason`. For CONVERGED/STALLED runs, `invariant_diff` must be null.

**TRACE_HASH_MISMATCH** — deterministic replay of the training run produced a
different `trace_hash` than the one recorded in the certificate. This indicates
the cert was generated with a different implementation, seed, or data ordering.

## Determinism discipline

- RNG: `numpy.random.default_rng(seed)` — PCG64, no global state
- Data: first `n_samples` MNIST training images, binarised at threshold 0.5
- Shuffle: one permutation at start, using the seeded RNG
- Batch size: fixed at 100
- No BLAS parallelism: pure numpy matmul only
- Energy computation: `log1p(exp(...))` for numerical stability

## Gates

1. **Gate 1 — Schema**: required fields, types, algorithm == rbm_cd1_numpy
2. **Gate 2 — Config sanity**: n_visible==784, n_hidden in [1,1024], etc.
3. **Gate 3 — Deterministic replay**: re-runs training and checks trace_hash
4. **Gate 4 — Invariant_diff contract**: null iff CONVERGED/STALLED

## Mapping protocol

References `Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`
via `mapping_protocol_ref.json` (QA_MAPPING_PROTOCOL_REF.v1).
