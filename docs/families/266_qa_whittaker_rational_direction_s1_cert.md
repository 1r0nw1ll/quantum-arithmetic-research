# [266] QA Whittaker Rational Direction S¹ Cert

## What this is

**Layer 1** of the Whittaker → QA development ladder (see [bridge spec §8](../specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md)). Layer 1 establishes the QA-rational direction net on the unit circle as a finite, exactly-enumerable rational sample-net with two structural theorems on the closed first quadrant. Future layers extend to 3D, Whittaker 1903 wave-kernel approximation, Whittaker 1904 two-scalar-potential bridge, and Maxwell scalar-pair reconstruction. Those later layer IDs are unassigned until build time.

**Primary source**:
- E. T. Whittaker (Whittaker, 1903). *On the partial differential equations of mathematical physics.* Math. Annalen 57:333–355. DOI: 10.1007/BF01444290. (Whittaker, 1903) is the **motivation** for studying QA-rational angular nets; the cert certifies only the QA-side discretization layer, not Whittaker's wave-equation theorem itself.

**Bridge background only** (not mapped in v1):
- E. T. Whittaker (Whittaker, 1904). *On an Expression of the Electromagnetic Field due to Electrons by Means of Two Scalar Potential Functions.* Proc. London Math. Soc. s2-1:367–372. DOI: 10.1112/plms/s2-1.1.367.

## Claim (narrow)

The QA-rational direction set

```
D_m := { (C/G, F/G) : (b, e) ∈ {1..m}², gcd(b, e) = 1,
                      d = b+e,  a = b+2e,
                      C = 2·d·e,  F = a·b = d² − e²,  G = d² + e² }
```

is finite, exactly enumerable in integer arithmetic, and admits two structural theorems on the closed first quadrant of `S¹`:

- **(W2) Angular separation lower bound** (rigorous, all-pairs): for any two distinct directions in `D_m`, `sin(angular_separation) ≥ 1 / (G_i · G_j) ≥ 1 / G_max(m)²`. Proven from cross-product integrality (the integer `|C_i·F_j − C_j·F_i|` is non-negative and vanishes iff vectors are parallel/equal in Q1, hence is at least 1 for distinct directions).
- **(W3) Lipschitz nearest-neighbor sampling error bound** on the closed quadrant: with virtual boundary anchors `E_0 = (1, 0)` at `θ = 0` and `E_∞ = (0, 1)` at `θ = π/2` added as **observer-side anchors only** (NOT QA seeds, NOT counted in `|D_m|`, NOT in W1/W2), define `D_m⁺ := D_m ∪ {E_0, E_∞}` and the closed max angular gap `Δ_max⁺(m) := max_i (θ̃_{i+1} − θ̃_i)`. Then for any `L`-Lipschitz angular profile `g` on `[0, π/2]`, nearest-neighbor sampling at `D_m⁺` yields:

```
sup_{θ ∈ [0, π/2]} |g(θ) − g(θ_NN(θ))| ≤ L · Δ_max⁺(m)
```

**Claim scope**:
- Cert does **not** prove Whittaker's wave-equation theorem, Maxwell's equations, electromagnetism, two-scalar-potential reductions, or any physics.
- Cert does **not** claim density of `D_m` in `S¹` as `m → ∞` (deferred to v2).
- Cert does **not** cover 3D `S²` direction sets (Layer 2; future ID unassigned).
- Cert does **not** include the half-gap tighter constant `L · Δ_max⁺/2` (loose form is used; sharp-constant improvement reserved).

## Bit-exact predictions (verified)

| m  | `\|D_m\|` | `G_max(m)` | `Δ_max⁺(m)` (rad) |
|----|-----------|------------|-------------------|
| 9  | 55        | 370        | 0.199337          |
| 24 | 359       | 2785       | 0.079957          |
| 72 | 3,175     | 25,633     | 0.027396          |

W2 all-pairs counts: 1,485 (m=9), 64,261 (m=24), 5,038,725 (m=72) — pure integer arithmetic. Self-test runtime is observed locally at ~2.6s and is not part of the cert claim.

## Theorem NT compliance

- **Construction layer**: integer-only / `fractions.Fraction`. Raw `d = b+e`, `a = b+2e` (no mod reduction in element computation per CLAUDE.md HARD rule). `C, F, G` are integers; `C² + F² = G²` exact. Angular ordering uses `Fraction(F, C)` comparison — never float. No `**2` (uses `x*x`).
- **Observer layer**: floats appear only in the W3 Lipschitz test grid, the test function `g`, the `π/2` upper boundary, and sup-error reporting. Each is declared as observer projection. Firewall crossed exactly twice (input `g`, output sup-error report).
- **Boundary anchors `E_0`, `E_∞`** are observer-side, NOT QA seeds, NOT counted in `|D_m|`.

## Validator gates

| Gate | Check |
|------|-------|
| WRD_1 | `schema_version` matches `QA_WHITTAKER_RATIONAL_DIRECTION_S1_CERT.v1` |
| WRD_DECL | required fields present and well-typed; `m ∈ {9, 24, 72}` |
| WRD_W1 | bit-exact cardinality `\|D_m\| == declared_count` |
| WRD_W2 | all `i<j` pairs satisfy `\|C_i·F_j − C_j·F_i\| ≥ 1` (cross-product integrality theorem) |
| WRD_W3 | Lipschitz nearest-neighbor sup error `≤ L · Δ_max⁺(m)` (independently recomputed on a dense observer-side test grid) |
| WRD_HARD | per-seed `gcd(b, e) = 1`; `claimed_extra_seeds` checked for coprimality; `claimed_d_overrides` checked for `d == b+e` |
| WRD_SRC | `source_attribution` cites Whittaker, 1903, and DOI 10.1007/BF01444290 |
| WRD_WIT | at least 1 witness present |
| WRD_F | `fail_ledger` well-formed (where present) |

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s1_cert_v1/qa_whittaker_rational_direction_s1_cert_validate.py` |
| Mapping ref | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s1_cert_v1/mapping_protocol_ref.json` |
| PASS fixture (m=9, canonical) | `.../fixtures/pass_d9_canonical_net.json` |
| PASS fixture (m=24, canonical) | `.../fixtures/pass_d24_canonical_net.json` |
| PASS fixture (m=72, canonical) | `.../fixtures/pass_d72_canonical_net.json` |
| PASS fixture (Lipschitz witness, sin) | `.../fixtures/pass_lipschitz_witness_d9_sin.json` |
| FAIL fixture (non-coprime seed) | `.../fixtures/fail_noncoprime_seed.json` |
| FAIL fixture (non-QA seed form) | `.../fixtures/fail_nonqa_seed_form.json` |
| FAIL fixture (W3 bound too strong) | `.../fixtures/fail_w3_bound_too_strong.json` |
| FAIL fixture (W2 underclaimed separation) | `.../fixtures/fail_w2_underclaimed_separation.json` |
| Bridge spec / design doc | `docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md` |
| Primary-source PDF (1903) | `Documents/whittaker_corpus/whittaker_1903_partial_differential_equations_mathematical_physics.pdf` |
| Bridge-background PDF (1904) | `Documents/whittaker_corpus/whittaker_1904_two_scalar_potential_functions_electromagnetic_field.pdf` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s1_cert_v1
python qa_whittaker_rational_direction_s1_cert_validate.py --self-test
```

Expected: `{"ok": true, "results": [...]}` with all 8 fixtures (4 PASS, 4 FAIL) classified correctly. Runtime observed locally at ~2.6s on this hardware; the 5M-pair W2 check at m=72 dominates, but runtime is not a cert claim.

## Status

**Registered 2026-05-01 as family [266] in `qa_meta_validator.py`.**

Standalone validation passed before registration. See bridge spec §9 for the full build precondition checklist; Layer 1 build prep was authorized 2026-04-30 by Will.

## Future layers (research ladder)

**Layer ID convention.** `[266]` is definite once registered. Later Whittaker layer IDs are **unassigned** until each cert is built and reviewed. Do not reserve `[267]`-`[270]`: the current meta-validator already uses `[267]`-`[272]` for external-validation and doc/linter gate labels.

- `qa_whittaker_rational_direction_s2_cert_v1` — 3D `S²` direction net (paired QA seeds or Wildberger 3D rational).
- `qa_whittaker_wave_kernel_bridge_cert_v1` — Whittaker 1903 angular wave-kernel approximation (claim type: discretization/approximation, not "QA proves Whittaker").
- `qa_whittaker_two_scalar_potential_bridge_v1` — Whittaker 1904 Φ, Ψ scalar-potential mapping (rename Whittaker's scalars to avoid QA `F, G` collision).
- `qa_maxwell_scalar_pair_reconstruction_cert_v1` — reconstruct EM field components from Layer 4's QA carrier pairs; initial framing must be guarded ("QA-compatible representation," NOT "QA derives EM").
- **Layer 6 (research track, no cert ID)** — Hertz/Debye potentials, longitudinal/scalar interpretations, Bearden/Pond/SVP cross-references; primary-source-gated, hostile-reviewed; cert IDs assigned only after Layers 1–5 are stable.
