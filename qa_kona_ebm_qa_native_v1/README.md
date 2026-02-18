# qa_kona_ebm_qa_native_v1

QA state manifold as RBM latent space — orbit structure meets MNIST geometry.

## What this IS

This is NOT a cert layer bolted on top of a standard RBM. The 81 hidden units
are explicitly indexed by the 81 QA state-manifold states (b,e) for mod 9,
in canonical sorted order. The RBM latent space IS the QA state space.

## Mathematical foundation

QA state space (mod 9): all pairs (b,e) with b,e in {1,...,9} — 81 states.

QA step: (b,e) -> (e, d) where d = (b+e) % 9; if d==0 then d=9.

Orbit structure:
  COSMOS:      3 distinct 24-cycles = 72 states (hidden units 0-71 approx)
  SATELLITE:   1 8-cycle            =  8 states: multiples-of-3 pairs
  SINGULARITY: 1 fixed point        =  1 state: (9,9)

## Research question

Do QA orbits align with MNIST digit geometry?

- orbit_class_alignment: for each orbit type, the mean hidden activation
  probability per digit class (0-9). High values on a particular digit
  suggest that orbit type activates preferentially for that digit.

- orbit_coherence_score: mean pairwise Pearson correlation of hidden unit
  activations within each orbit type. A high score means units in the same
  orbit co-activate — the orbit is a coherent functional group.

- orbit_dominant_class: which digit class triggers the highest mean
  activation for each orbit type.

## Determinism contract

Same as family [62] (qa_kona_ebm_mnist_v1):
  numpy.random.default_rng(seed), single shuffle, batch_size=100,
  CD-1, explosion threshold 1000.0.
n_hidden=81 is fixed by the QA state space — not a user parameter.

## CLI

Generate a cert:
    python generate_cert.py --n-samples 1000 --n-epochs 5 --lr 0.01 --seed 42 \
        --cert-id kona_ebm_qa_native_001

Validate a cert:
    python validator.py fixtures/valid_qa_rbm_run.json
    python validator.py fixtures/valid_qa_rbm_run.json --json

Run self-test (all 3 fixtures):
    python validator.py --self-test

Inspect orbit map:
    python qa_orbit_map.py

## Fixtures

  valid_qa_rbm_run.json            -- PASS (n_samples=1000, n_epochs=5, seed=42)
  invalid_orbit_map_violation.json -- FAIL: orbit_map_hash tampered (Gate 3)
  invalid_trace_mismatch.json      -- FAIL: trace_hash tampered (Gate 4)

## Related families

  [62] qa_kona_ebm_mnist_v1: standard RBM with n_hidden as user parameter
  (this family uses n_hidden=81 fixed by QA state space)
