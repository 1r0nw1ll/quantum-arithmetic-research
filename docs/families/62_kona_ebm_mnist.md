# Family [62]: QA Kona EBM MNIST

## Purpose

Certifies a **Restricted Boltzmann Machine (RBM)** training run on MNIST under the QA
invariant contract: deterministic replay, energy-curve integrity, and typed structural
obstructions for training failure modes.

This is the first QA cert family to govern a real ML training loop (CD-1 contrastive
divergence on binarized MNIST digits). The cert does not evaluate model accuracy — it
certifies that the training trace is reproducible and that any instability is classified
as a named structural obstruction rather than left as an opaque numerical failure.

## Domain

Energy-Based Model (EBM) reasoning, mapped to the QA framework via
`QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`.

## Location

`qa_kona_ebm_mnist_v1/`

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | JSON Schema for `QA_KONA_EBM_MNIST_CERT.v1` |
| `validator.py` | Four-gate validator (schema, config, replay, invariant_diff) |
| `rbm_train.py` | Deterministic numpy-only RBM trainer (CD-1) |
| `generate_cert.py` | CLI to emit cert JSON from a training run |
| `fixtures/valid_stable_run.json` | PASS: 5-epoch stable run, energy -65 → -82 |
| `fixtures/invalid_gradient_explosion.json` | FAIL: lr=50.0, GRADIENT_EXPLOSION obstruction |
| `fixtures/invalid_nondeterministic_trace.json` | FAIL: tampered trace_hash, TRACE_HASH_MISMATCH |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |

## Invariants

| Status | Meaning |
|--------|---------|
| `CONVERGED` | Final energy < initial energy; training stable |
| `STALLED` | Energy non-decreasing; solver stuck at local minimum |
| `GRADIENT_EXPLOSION` | Weight update norm exceeded threshold; `invariant_diff` required |

## Failure Taxonomy

| fail_type | Trigger |
|-----------|---------|
| `GRADIENT_EXPLOSION` | `lr * grad_norm > 1000.0` during training |
| `TRACE_HASH_MISMATCH` | Deterministic replay produces different trace hash |

## Gates

1. **Gate 1 — Schema**: required fields, types, `algorithm == "rbm_cd1_numpy"`
2. **Gate 2 — Config sanity**: `n_visible == 784`, ranges, `lr > 0`
3. **Gate 3 — Deterministic replay**: re-run training, verify `trace_hash` exact match
4. **Gate 4 — invariant_diff contract**: `GRADIENT_EXPLOSION` requires typed obstruction; `CONVERGED`/`STALLED` require `invariant_diff == null`

## CLI

```bash
# Validate a fixture
python qa_kona_ebm_mnist_v1/validator.py qa_kona_ebm_mnist_v1/fixtures/valid_stable_run.json

# Self-test (all fixtures)
python qa_kona_ebm_mnist_v1/validator.py --self-test

# Generate a new cert
python qa_kona_ebm_mnist_v1/generate_cert.py --n-hidden 64 --n-samples 1000 --n-epochs 5 --lr 0.01 --seed 42

# Narrated demo
python demos/qa_family_demo.py --family kona_ebm

# CI mode
python demos/qa_family_demo.py --family kona_ebm --ci
```

## Determinism Discipline

- RNG: `numpy.random.default_rng(seed)` exclusively
- Data shuffle: once at start, then fixed order each epoch
- Batch size: fixed at 100
- No BLAS parallelism in critical path (pure numpy operations)
- Trace hash over energy-per-epoch list (6dp rounded floats)

## Notes

This family is the entry point for QA-governed ML training. The next step (option A)
is to use the QA state manifold (b,e) pairs and generator orbits as the RBM's latent
space — asking whether the three QA orbits (24-cycle Cosmos, 8-cycle Satellite,
1-cycle Singularity) align with learned data geometry.
