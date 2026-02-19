# QA Dynamics Spine

**Status**: Active | **Version**: 1.0 | **Date**: 2026-02-18
**Reference implementation**: Family [64] — QA Kona EBM QA-Native Orbit Reg

---

## Purpose

The QA Dynamics Spine is an opt-in standard for certificate families that evolve a state across time (training epochs, inference steps, control loops) and wish to certify stability or instability structure in a mathematically checkable way.

Before this spine, QA certs certified outcomes (trace hashes, invariant_diff, coherence scores). The Spine adds a new class of certified object: **stability certificates** — structural claims about whether the dynamics were contracting, noise-bounded, or in an instability regime, recomputed at Gate 3 and hash-anchored at Gate 4.

---

## Definition: Dynamics-Compatible Family

A cert family is **Dynamics-Compatible** if it:

1. Evolves a state `x_t` across discrete time steps (epochs, iterations, etc.)
2. Logs a **deviation norm** per step: `dev_norm_per_step[t] ≥ 0`
3. Includes a **generator_curvature** block in its cert result (see §4)
4. Implements Gate 3 curvature recomputation and Gate 4 hash-chain inclusion (see §7)

A Dynamics-Compatible family opts in by declaring `"dynamics_spine": "v1"` in its `mapping_protocol_ref.json` scope_note (informally) and by conforming to this spec.

---

## Core Decomposition

Every Dynamics-Compatible family must identify:

| Concept | Meaning | Family [64] instance |
|---------|---------|----------------------|
| **State** `x_t` | The evolving parameter/state vector | RBM weight matrix `W_t` |
| **Projector** `Q` | Extracts deviation modes from `x_t` | Orbit mean-subtraction: `Q = I - (1/\|O\|)·11ᵀ` per orbit |
| **Deviation** `Δ_t = Q·x_t` | Component of state in the deviation subspace | Within-orbit weight deviation |
| **Restoring generator** | Contracts Δ toward zero | Orbit-coherence regularizer, strength `λ` |
| **Stochastic generator** | Injects noise into Δ | CD-1 gradient (minibatch + negative-phase truncation noise) |
| **Step schedule** `η_t` | Controls the effective temperature | Learning rate per epoch |

---

## Spine Primitive A: Deviation Norm

**Required logged quantity**: `dev_norm_per_step` (list of non-negative floats, one per step)

Canonical mapping:

```
dev_norm_per_step[t] ↔ ‖Δ_t‖_F
```

In family [64], this is `reg_norm_per_epoch` — the Frobenius norm of the intra-orbit weight deviation.

The deviation norm trajectory is the primary observable for the dynamics. It encodes:
- Whether the restoring generator is winning (norm decreasing)
- When stochastic noise dominates (norm increasing or oscillating)
- The concentration of instability (epoch of maximum norm)

---

## Spine Primitive B: Generator Interaction Curvature

The **generator interaction curvature** κ_QA measures whether the restoring generator dominates the stochastic generator at each step:

```
κ_QA > 0  ⟺  deviation contraction (stable attractor)
κ_QA < 0  ⟺  deviation expansion permitted (instability regime)
```

### Required cert block: `result.generator_curvature`

```json
{
  "definition": "string (human-readable formula for kappa_hat)",
  "kappa_hat_per_epoch": [float, ...],
  "min_kappa_hat": float,
  "min_kappa_epoch": int (1-indexed, tie-break: first occurrence),
  "kappa_hash": "hex64 (sha256 of JSON-serialized kappa_hat_per_epoch)",
  "max_dev_norm": float,
  "max_dev_epoch": int (1-indexed, tie-break: first occurrence)
}
```

`kappa_hash` is a tamper-evident seal. Any modification to `kappa_hat_per_epoch` breaks the hash and is caught at Gate 4.

`max_dev_norm` and `max_dev_epoch` attest where instability concentrates — the epoch at which the deviation norm was highest (the "worst stability point").

---

## Curvature Regimes

### Closed-Form Regime (Preferred)

When the restoring potential is a **mean-subtraction quadratic** `R(x) = ½‖Qx‖²_F` with strength `λ`, the deviation subspace Hessian is exactly `H = λ·Q`, and:

```
κ̂_QA = 1 − |1 − η_t · λ|
```

This requires **no probe computation** — it depends only on logged `lr_per_epoch[t]` and `lambda_orbit`. Gate 3 recomputation is a one-liner.

**Stability condition**: `0 < η · λ < 2`

Family [64] numerical examples:
- lr=0.01, λ=10: κ̂ = 1 − |1 − 0.1| = 0.1 (stable)
- lr=0.5, λ=5:   κ̂ = 1 − |1 − 2.5| = −0.5 (unstable → `NEGATIVE_GENERATOR_CURVATURE`)

### Probe-Based Regime (Future Families)

When `H ≠ λQ` (non-quadratic or non-projective restoring term), curvature requires an empirical noise estimate:

```
κ̂_QA = λ_min(H_eff) − 0.5 · η · σ̂²_dev
```

where `σ̂²_dev` is computed from gradient residuals on a fixed probe set (n_probe samples, deterministic seed). The `generator_curvature` block must additionally declare:

```json
"probe": {
  "probe_strategy": "fixed_first_n",
  "n_probe": 256,
  "microbatches": 8,
  "perm_seed": 42,
  "sample_index_sha256": "hex64"
}
```

---

## Obstruction Algebra

Dynamics-Compatible families must declare and handle these obstruction types:

| Obstruction | Gate | Trigger |
|-------------|------|---------|
| `NEGATIVE_GENERATOR_CURVATURE` | 3 | Recomputed κ̂ < 0 at any logged epoch |
| `CURVATURE_RECOMPUTE_MISMATCH` | 3 | Recomputed κ̂ or kappa_hash ≠ cert values |
| `CURVATURE_PROBE_MISMATCH` | 3 | Probe definition or sample_index_sha256 mismatch (probe-based regime only) |
| `MAX_DEV_SPIKE_ATTESTATION_MISMATCH` | 3 | Claimed max_dev_norm/epoch ≠ argmax of dev_norm_per_step |

All obstructions require full `invariant_diff` payloads with `fail_type`, `target_path`, `reason`, and structured `invariant_diff` body (epoch index, expected vs. got values, etc.).

`NEGATIVE_GENERATOR_CURVATURE` is a **structural obstruction** — it triggers when the *recomputed* curvature is negative, regardless of what the cert claims. It cannot be silenced by cert edits alone (Gate 4 hash-chain would also fail).

---

## Gate Contract

### Gate 3: Curvature Recompute + Attestation Integrity

For every epoch listed in `result.generator_curvature.kappa_hat_per_epoch`:

1. **Recompute** `kappa_hat[t]` from logged `lr_per_epoch[t]` and declared `lambda_orbit` (closed-form) or from probe gradient traces (probe-based).
2. **Compare** to cert value — fail `CURVATURE_RECOMPUTE_MISMATCH` on mismatch.
3. **Check negativity** — fail `NEGATIVE_GENERATOR_CURVATURE` if recomputed < 0.
4. **Verify** `min_kappa_hat == min(kappa_hat_per_epoch)` and `min_kappa_epoch == argmin` (tie-break: first).
5. **Verify** `max_dev_norm == max(dev_norm_per_step)` and `max_dev_epoch == argmax` (tie-break: first).
6. Fail `MAX_DEV_SPIKE_ATTESTATION_MISMATCH` on any attestation mismatch.

### Gate 4: Hash-Chain Integrity

The canonical hash payload (whatever the family hashes for trace integrity) **must include** `result.generator_curvature` in full, including:
- `kappa_hat_per_epoch`
- `kappa_hash`
- `max_dev_norm`, `max_dev_epoch`

Any post-generation edit to curvature fields breaks the Gate 4 hash and is caught as `TRACE_HASH_MISMATCH`.

---

## Opt-In Checklist for New Families

To become Dynamics-Compatible, a new family must:

- [ ] Identify the deviation operator `Q` and document it in the human tract
- [ ] Log `dev_norm_per_step` (Frobenius norm of `Δ_t`)
- [ ] Choose curvature regime: closed-form (preferred) or probe-based
- [ ] Emit `result.generator_curvature` block with all required fields
- [ ] Implement Gate 3 curvature recompute check (all four obstruction types)
- [ ] Include `generator_curvature` in Gate 4 hash-chain payload
- [ ] Add PASS fixture demonstrating κ̂ > 0 with `generator_curvature` present
- [ ] Add FAIL fixture demonstrating `NEGATIVE_GENERATOR_CURVATURE`
- [ ] Add FAIL fixture demonstrating `MAX_DEV_SPIKE_ATTESTATION_MISMATCH`

---

## Reference Implementation: Family [64]

Family [64] (`qa_kona_ebm_qa_native_orbit_reg_v1/`) is the canonical Dynamics-Compatible reference:

| Spine element | Implementation |
|--------------|----------------|
| State `x_t` | RBM weight matrix `W_t ∈ ℝ^{81×784}` |
| Projector `Q` | Orbit mean-subtraction (per COSMOS/SATELLITE orbit) |
| Deviation norm | `reg_norm_per_epoch` (Frobenius norm of `W_orbit - μ_orbit`) |
| Curvature regime | **Closed-form**: `κ̂ = 1 − \|1 − lr·lambda_orbit\|` |
| Gate 3 recompute | One-liner from `lr_per_epoch` + `lambda_orbit` |
| Stability condition | `0 < lr·lambda_orbit < 2` |
| Fixtures | `valid_orbit_reg_kappa_stable.json` (PASS), `invalid_negative_generator_curvature.json` (FAIL), `invalid_max_dev_spike_epoch.json` (FAIL) |
| Tag | `af3a430` (κ̂_QA Gate 3 implementation) |

---

## Theoretical Grounding

The stability claims certified by this spine are grounded in:

- **Lyapunov analysis**: `E[‖Δ_{t+1}‖²_F] ≤ (1 − η·λ)·E[‖Δ_t‖²_F] + η²·σ²/2`
- **Noise floor**: `E[‖Δ‖²_F] ≲ η·σ²/(2λ)` — deviation energy scales linearly with lr, inversely with λ
- **Freidlin–Wentzell escape**: `P(escape) ≍ exp(−λ·r²/(2·η·σ²))` — LR decay exponentially suppresses basin escape
- **Mandt SDE limit**: OU stationary covariance `C = η·Σ_dev/(2λ)` — LR is effective temperature
- **Spectral radius** (closed-form): `ρ_dev(A) = |1 − η·λ|`; stable iff `ρ_dev < 1`

Full derivations: `memory/family64_theory.md`, `docs/families/64_kona_ebm_qa_native_orbit_reg.md#theoretical-interpretation`.

---

## See Also

- `docs/families/64_kona_ebm_qa_native_orbit_reg.md` — reference implementation human tract
- `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md` — universal theorem object
- `memory/family64_theory.md` — theory notes and derivations
- `Documents/QA_ADAPTER_PATTERN_SPEC.md` — adapter pattern (non-dynamics families)
