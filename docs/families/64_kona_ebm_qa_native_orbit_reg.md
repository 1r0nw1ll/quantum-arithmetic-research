# Family [64]: QA Kona EBM QA-Native Orbit Reg

## Purpose

Family [64] extends the QA-native EBM (family [63]) by adding an explicit
**orbit-coherence regularizer** to the RBM training objective. The regularizer
penalises weight divergence within each QA orbit type, encouraging hidden units
that share the same orbit (Cosmos / Satellite / Singularity) to learn similar
receptive fields.

## Research Question

Does enforcing orbit coherence via a regularization term `λ · R(W)` cause units
grouped by QA orbit membership to develop correlated receptive fields, and does
this effect strengthen as `λ` grows?

## Location

`qa_kona_ebm_qa_native_orbit_reg_v1/`

## Regularizer Definition

```
R(W) = Σ_{O ∈ {COSMOS, SATELLITE, SINGULARITY}}
         Σ_{i ∈ O}
           ||W_i - μ_O||²

where μ_O = (1/|O|) Σ_{i∈O} W_i   (mean weight vector for orbit O)
```

Gradient applied per batch after the CD-1 update:

```
W[i] -= lr * lambda_orbit * 2.0 * (W[i] - μ_O)   for each i ∈ O
```

This is numerically identical to an L2 penalty pulling each unit's weights
toward the orbit mean.

## QA State Manifold (mod 9)

Same as family [63]:

| Orbit | Period | Count |
|-------|--------|-------|
| COSMOS | 24 | 72 |
| SATELLITE | 8 | 8 |
| SINGULARITY | 1 | 1 |

## Artifacts

| File | Purpose |
|------|---------|
| `rbm_qa_orbit_reg_train.py` | CD-1 RBM + orbit-coherence regularizer, `train_qa_orbit_reg_rbm()` |
| `schema.json` | Schema for `QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1` |
| `validator.py` | Five-gate validator (Gates 1–5) |
| `generate_cert.py` | CLI cert generator |
| `generate_multi_seed_report.py` | Multi-seed orbit coherence report with `--lambda-orbit` |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_orbit_reg_stable_run.json` | PASS: λ=1e-3, seed=42, 5 epochs |
| `fixtures/invalid_regularizer_instability.json` | FAIL: REGULARIZER_NUMERIC_INSTABILITY (λ=1e15) |
| `fixtures/invalid_trace_mismatch.json` | FAIL: TRACE_HASH_MISMATCH |

## Gates

1. **Gate 1 — Schema**: fields, types, `algorithm == "rbm_qa_native_orbit_reg_cd1_numpy"`
2. **Gate 2 — Config sanity**: n_visible==784, ranges, lr > 0, lambda_orbit ≥ 0
3. **Gate 3 — Orbit map integrity**: recompute orbit_map_hash from `qa_orbit_map.py`, verify exact match; also checks `coherence_gap_stats` present
4. **Gate 4 — Deterministic replay**: re-run `train_qa_orbit_reg_rbm`, verify `trace_hash` and `reg_trace_hash` exact match
5. **Gate 5 — invariant_diff contract**: REGULARIZER_NUMERIC_INSTABILITY or GRADIENT_EXPLOSION → typed obstruction required; CONVERGED/STALLED → null

## Failure Taxonomy

| fail_type | Gate | Trigger |
|-----------|------|---------|
| `ORBIT_MAP_VIOLATION` | 3 | orbit_map_hash does not match canonical QA enumeration |
| `TRACE_HASH_MISMATCH` | 4 | deterministic replay produces different trace hash |
| `GRADIENT_EXPLOSION` | 5 | weight update norm > 1000.0 |
| `REGULARIZER_NUMERIC_INSTABILITY` | 5 | NaN or Inf in W after regularizer step |

## Additional Trace Fields

- `reg_norm_per_epoch`: Frobenius norm of regularizer gradient `2*(W[i]-μ_O)` per epoch
- `reg_trace_hash`: sha256 of the `reg_norm_per_epoch` list (determinism check)

## CLI

```bash
# Self-test (3 fixtures)
python qa_kona_ebm_qa_native_orbit_reg_v1/validator.py --self-test

# Validate a cert
python qa_kona_ebm_qa_native_orbit_reg_v1/validator.py \
  qa_kona_ebm_qa_native_orbit_reg_v1/fixtures/valid_orbit_reg_stable_run.json

# Generate a cert
python qa_kona_ebm_qa_native_orbit_reg_v1/generate_cert.py \
  --n-samples 1000 --n-epochs 5 --lr 0.01 --lambda-orbit 1e-3 --seed 42

# Multi-seed report (5 seeds × 5 epochs, λ=1e-3)
python qa_kona_ebm_qa_native_orbit_reg_v1/generate_multi_seed_report.py \
  --lambda-orbit 1e-3 --n-epochs 5 --json-out /tmp/reg_report.json
```

## Baseline Result (seed=42, λ=1e-3, n_epochs=5, lr=0.01, n_samples=1000)

With weak regularization (λ=1e-3), orbit coherence scores remain near the
permutation-test null (z ≈ -0.85 COSMOS, +0.51 SATELLITE). This is the expected
result: the regularizer is present but not strong enough to overcome the random
initialization and short training regime.

**Interpretation**: λ=1e-3 is below the threshold needed to visibly align orbit
receptive fields at 5 epochs. The scientific interest is in measuring how coherence
z-scores scale with λ across 1e-5 / 1e-3 / 1e-1 / 1.0 / 10.0 and longer training.

## Trajectory Analysis (λ=10, 50 epochs)

A 5-seed sweep at λ=10 over 50 epochs reveals consistent three-phase dynamics
in the COSMOS orbit coherence signal.

### COSMOS Coherence Peak

COSMOS coherence peaks at **epochs 6–7** across all 5 seeds (peak c ≈ 0.000977–0.000979).
The peak is time-locked and seed-independent: despite different random initializations
the orbit structure creates a reproducible resonance window in the early training phase.

### Three Training Phases

| Phase | Epochs | Coherence | reg_norm | Description |
|-------|--------|-----------|----------|-------------|
| A | 1–7 | Rising | Falling | CD-1 creates initial data structure; regularizer reinforces orbit groupings |
| B | 7–23 | Falling | Falling | Weight vectors converge toward orbit mean, but activation correlations decay — CD-1 noise scrambles representations even as weights cluster |
| C | 23+ | Falling | Rising | CD-1 noise fully overcomes regularizer; weights and activations both diverge from orbit structure |

**Phase A**: The regularizer and CD-1 are cooperative. Early weight movement is large,
orbit pull is strong relative to the noise, and coherence climbs.

**Phase B**: The regularizer continues to reduce reg_norm (weights are clustering),
but the CD-1 stochastic noise is progressively scrambling the hidden activations.
Coherence falls even though the weight geometry is still improving.

**Phase C**: reg_norm reverses and increases monotonically, indicating that the
CD-1 update magnitude has grown large enough to pull individual weights away from
the orbit mean faster than the regularizer can correct.

### reg_norm Trajectory

The reg_norm minimum of ≈1.84 occurs at **epoch 22–23**, after which it increases
monotonically, reaching ≈4.5 by epoch 50.

### Dissociation: Weight Convergence ≠ Activation Coherence

The reg_norm minimum (epoch 23) does not coincide with the coherence peak (epoch 7).
This 16-epoch gap demonstrates that weight-space clustering and activation-space
coherence are **decoupled quantities** under CD-1 noise. Orbit regularization shapes
the weight geometry, but whether that geometry expresses as coherent activations depends
on the current noise level in the contrastive divergence updates.

This is a key finding: a future regularizer design that additionally targets
activation coherence (rather than weight proximity alone) would need a different
loss term.

### Gradient Mass Imbalance (v2 Design Note)

The unscaled regularizer gives COSMOS (72 units) **9× the total gradient mass**
compared to SATELLITE (8 units). This explains why the coherence effect appears
first and most strongly in COSMOS. A v2 regularizer should normalize gradient
contribution per orbit by dividing by `|O|`, making the per-unit pull equal across
orbit types.

## LR Decay Variant (lr_schedule)

Motivated by the three-phase finding, the trainer now supports an optional
step-wise LR schedule via `model_config.lr_schedule`. The intent is to damp CD-1
noise onset at the Phase A→B transition (epoch 7–8), extending the coherence peak
before Phase B decay begins.

**Configuration**: `lr_schedule` is a list of `{"epoch": int, "lr": float}` entries
specifying the learning rate to apply at that epoch boundary. If absent, a flat lr
is used (original behavior).

**Gate 6** validates `lr_per_epoch` in the trace against the declared schedule:
each recorded per-epoch lr must match the schedule value active at that epoch.

New fixtures:

| File | Result | Notes |
|------|--------|-------|
| `fixtures/valid_orbit_reg_lr_decay_stable_run.json` | PASS | λ=10, lr drops at epoch 8 |
| `fixtures/invalid_lr_schedule.json` | FAIL: `LR_SCHEDULE_INVALID` | recorded lr deviates from schedule |

## Relation to Family [63]

Family [63] is the null-result baseline: standard CD-1 with no regularizer.
Family [64] adds λ·R(W) as the intervention. The permutation gap test (n_perm=500,
add-one smoothed p-values) provides the statistical scaffold for comparing the two.

## Next Steps

- Sweep λ ∈ {1e-5, 1e-3, 1e-1, 1.0, 10.0} at n_epochs=20 across 5 seeds
- Check for orbit coherence z-score monotone increase with λ
- Compare reconstruction error [63] vs [64] at matched λ values
- If coherence emerges: cert the first run where z > 2 for both COSMOS and SATELLITE
