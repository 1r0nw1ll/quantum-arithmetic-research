# Family [63]: QA Kona EBM QA-Native

## Purpose

The first QA cert family where the **QA state manifold is the latent space** itself —
not a cert layer on top of an external model, but an RBM whose 81 hidden units are
explicitly indexed by the 81 QA states (b,e) for mod 9.

The cert reports whether the QA orbit structure (Cosmos / Satellite / Singularity)
aligns with MNIST digit geometry after training.

## Research Question

Do units grouped by QA orbit type (24-cycle Cosmos, 8-cycle Satellite, fixed-point
Singularity) develop correlated receptive fields that partition onto different digit
classes?

## Location

`qa_kona_ebm_qa_native_v1/`

## QA State Manifold (mod 9)

| Orbit | States | Period | Count |
|-------|--------|--------|-------|
| COSMOS | 3 distinct sub-orbits | 24 | 72 |
| SATELLITE | (3,3),(3,6),(3,9),(6,3),(6,6),(6,9),(9,3),(9,6) | 8 | 8 |
| SINGULARITY | (9,9) | 1 | 1 |

Hidden unit index i ↔ canonical sorted state i in `sorted([(b,e) for b in 1..9 for e in 1..9])`.

## Artifacts

| File | Purpose |
|------|---------|
| `qa_orbit_map.py` | Enumerates 81 QA states, classifies orbits, computes orbit_map_hash |
| `rbm_qa_native_train.py` | CD-1 RBM with n_hidden=81 fixed, orbit alignment analysis |
| `schema.json` | Schema for `QA_KONA_EBM_QA_NATIVE_CERT.v1` |
| `validator.py` | Five-gate validator including orbit map integrity (Gate 3) |
| `generate_cert.py` | CLI cert generator |
| `fixtures/valid_qa_rbm_run.json` | PASS: 5-epoch run, orbit alignment reported |
| `fixtures/invalid_orbit_map_violation.json` | FAIL: ORBIT_MAP_VIOLATION (Gate 3) |
| `fixtures/invalid_trace_mismatch.json` | FAIL: TRACE_HASH_MISMATCH (Gate 4) |

## Gates

1. **Gate 1 — Schema**: fields, types, `algorithm == "rbm_qa_native_cd1_numpy"`
2. **Gate 2 — Config sanity**: n_visible==784, ranges, lr > 0
3. **Gate 3 — Orbit map integrity**: recompute orbit_map_hash, verify exact match
4. **Gate 4 — Deterministic replay**: re-run training, verify trace_hash exact match
5. **Gate 5 — invariant_diff contract**: GRADIENT_EXPLOSION requires typed obstruction

## Failure Taxonomy

| fail_type | Gate | Trigger |
|-----------|------|---------|
| `ORBIT_MAP_VIOLATION` | 3 | orbit_map_hash does not match canonical QA enumeration |
| `TRACE_HASH_MISMATCH` | 4 | deterministic replay produces different trace hash |
| `GRADIENT_EXPLOSION` | 5 | weight update norm > 1000.0 |

## CLI

```bash
# Self-test
python qa_kona_ebm_qa_native_v1/validator.py --self-test

# Validate
python qa_kona_ebm_qa_native_v1/validator.py qa_kona_ebm_qa_native_v1/fixtures/valid_qa_rbm_run.json

# Generate a cert (prints orbit_class_alignment)
python qa_kona_ebm_qa_native_v1/generate_cert.py --n-samples 1000 --n-epochs 5 --lr 0.01 --seed 42

# Print orbit map
python qa_kona_ebm_qa_native_v1/qa_orbit_map.py
```

## Baseline Result (seed=42, n_epochs=5, lr=0.01, n_samples=1000)

Orbit coherence scores: COSMOS=0.000972, SATELLITE=0.000974, SINGULARITY=1.0 (trivial).

**Interpretation**: At 5 epochs with standard CD-1, QA orbit membership does not impose
functional grouping on hidden units. Units in the same orbit learn independent receptive
fields. This is the null result — expected without an orbit-coherence regularization term.

The orbit_class_alignment values are digit-dependent (digit 8 and 9 activate all orbits
most strongly, consistent with high pixel density) but orbit-type-independent (COSMOS,
SATELLITE, SINGULARITY respond identically to each digit class).

## Next Steps (Option A Research Track)

1. Add orbit-coherence regularization: `λ * Σ_orbit ||W_orbit - mean(W_orbit)||_F²`
2. Test whether enforced orbit coherence improves or hurts reconstruction
3. Longer training (20-50 epochs) to check for late-stage orbit alignment emergence
4. Ablation: compare n_epochs=5/20/50 orbit coherence scores

## Relation to Family [62]

Family [62] (`qa_kona_ebm_mnist_v1`) is the Option (b) baseline: QA cert layer on top
of a standard RBM. This family [63] is Option (a): the QA manifold IS the latent space.
Both families are under the same cert contract and meta-validator gate.
