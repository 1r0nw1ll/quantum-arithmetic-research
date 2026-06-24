# [504] QA Star-G Tensor Orbit Module Cert

## What this is

Machine-checkable certificate that QA's mod-9 state space is a G-module for
G = Z/24Z × Z/8Z × {1}, directly instantiating the ⋆G tensor algebra of
Nguyen et al. (arXiv:2605.20440). Four integer certificates establish the
Kronecker direct-sum structure, the Chinese Remainder decomposition of the
Cosmos sub-action, the Z/8Z antipodal half-period on the Satellite, and the
universal sigma^24 identity on the full state space.

## Claim (narrow)

| Check | Claim |
|-------|-------|
| **SGT_1** | sigma preserves orbit type for all 81 pairs: Cosmos→Cosmos, Satellite→Satellite, Singularity→Singularity (0 violations) |
| **SGT_2** | CRT: sigma^8 has period 3 on all 72 Cosmos pairs; sigma^3 has period 8 on all 72 Cosmos pairs; gcd(3,8)=1, 3×8=24 → Z/24Z ≅ Z/3Z × Z/8Z |
| **SGT_3** | sigma^4 has period exactly 2 on all 8 Satellite pairs (Z/8Z antipodal map) |
| **SGT_4** | sigma^24 = identity on all 81 pairs; lcm(24,8) = 24 (universal period) |

## The ⋆G Tensor Algebra Context

Nguyen et al. (arXiv:2605.20440) prove three machine-verified results for any
finite group G:

1. **Eckart-Young optimality** (Theorem 1): ⋆G-SVD gives the best rank-k
   approximation among all rank-k G-equivariant tensors.
2. **Kronecker factorization** (Theorem 2): If G = G₁ × G₂, then the ⋆G
   decomposition factors as ⋆G₁ ⊗ ⋆G₂.
3. **Lean 4 formalization** (§4): 600-line machine-verified proof of the full
   ⋆G algebra.

For QA: G = Z/24Z (Cosmos) × Z/8Z (Satellite) × {1} (Singularity). This cert
instantiates Theorem 2 by verifying the Kronecker direct-sum structure.

## G-Module Direct Sum (SGT_1)

sigma maps each orbit class to itself — the three classes are invariant
submodules. This is the precondition for the Kronecker factorization: the
⋆G₁×G₂ decomposition requires that the G₁ and G₂ sub-actions operate on
independent invariant subspaces.

```
sigma(Cosmos pair) ∈ Cosmos     ✓  (72/72 pairs)
sigma(Satellite pair) ∈ Satellite  ✓  (8/8 pairs)
sigma(Singularity) = Singularity   ✓  (1/1 pair)
```

## CRT Decomposition of Z/24Z (SGT_2)

The Chinese Remainder Theorem: since gcd(3,8) = 1 and 3×8 = 24,

```
Z/24Z ≅ Z/3Z × Z/8Z
```

For QA's Cosmos orbit:
- sigma^8 generates the Z/3Z sub-action: period 3 on every Cosmos pair
- sigma^3 generates the Z/8Z sub-action: period 8 on every Cosmos pair

Witness (orbit O₁ starting at (1,1)):

| k | sigma^k(1,1) | Notes |
|---|--------------|-------|
| 8 | (7,1) | ≠ (1,1) → sigma^8 ≠ id |
| 16 | (4,1) | ≠ (1,1) → (sigma^8)² ≠ id |
| 24 | (1,1) | = (1,1) → (sigma^8)³ = id, period 3 ✓ |

This is the Kronecker factorization of the Cosmos G-module factor:

```
Z/24Z ≅ Z/3Z × Z/8Z
        ↑          ↑
    sigma^8     sigma^3
    (period 3)  (period 8)
```

## Z/8Z Half-Period Antipodal Map (SGT_3)

On the 8-element Satellite orbit, sigma^4 is the "antipodal map": it maps
each pair to its diametrically opposite pair in the 8-cycle, with period 2.

Witness (Satellite pair (3,3)):
```
sigma^4(3,3) = (6,6) ≠ (3,3)   → antipodal, not identity
sigma^8(3,3) = (3,3)            → period 8 restored
```

Period of sigma^4 on all 8 Satellite pairs = **2** (uniform).

This is the Z/8Z sub-action at level k=4: sigma^4 generates the unique
Z/2Z subgroup of Z/8Z (the unique involution).

## Universal Period (SGT_4)

lcm(Cosmos period, Satellite period) = lcm(24, 8) = 24.

Since 8 | 24, sigma^24 closes the Satellite orbit (8-cycle visited 3 times)
simultaneously with the Cosmos 24-cycle. **sigma^24 = identity on all 81 pairs.**

```
violations = 0 / 81
lcm(24, 8) = 24  (not 192 = 24×8; the Satellite period divides, not extends)
```

This is the ⋆G claim for the combined group: the minimal k with sigma^k = id
on the full G-module is |G_max| = lcm(|G_Cosmos|, |G_Satellite|) = 24.

## Lean 4 Extension of Cert [128]

Cert [128] provides a Lean 4 proof that the Pisano period π(9) = 24, which
anchors the Cosmos orbit period in formal arithmetic. The ⋆G Lean 4
formalization (arXiv:2605.20440 §4) provides the ambient algebra over which
the G-module decomposition (SGT_1–SGT_4) is formally meaningful.

Together: [128] establishes |Z/24Z| = 24 for QA's Cosmos; this cert
establishes the three-factor G-module structure G = Z/24Z × Z/8Z × {1}.

## Theorem NT Compliance

All continuous constructions (character group of Z/24Z, complex 24th roots
of unity, continuous equivariant learning performance) are observer
projections. The G-module equivariance is certified entirely by integer
period arithmetic. No float state appears anywhere in the cert.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_star_g_tensor_cert_v1/qa_star_g_tensor_cert_validate.py` |
| Mapping ref | `qa_star_g_tensor_cert_v1/mapping_protocol_ref.json` |
| PASS fixture | `fixtures/pass_star_g_tensor.json` |
| FAIL: orbit type mismatch | `fixtures/fail_orbit_type_mismatch.json` |
| FAIL: wrong sigma^8 CRT period | `fixtures/fail_crt_wrong_sigma8_period.json` |
| FAIL: wrong satellite half-period | `fixtures/fail_satellite_wrong_half_period.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_star_g_tensor_cert_v1
python3 qa_star_g_tensor_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`, 1 PASS + 3 FAIL fixtures.

## QA Axiom Compliance

- **A1**: tau_ranks, period counts in {1,...,N} (integers, no zero)
- **A2**: Cosmos period 24, Satellite period 8 are derived from sigma dynamics
- **S1**: G-function G(b,e) uses `(b+e)*(b+e) + e*e` not `**2` (cert [501] context)
- **S2**: No float state; all period comparisons are integer
- **T2**: Continuous characters (roots of unity) are observer projections
- **T1**: QA time = integer path length k; sigma^k is pure integer iteration

## Primary Sources

- Nguyen, T.T. et al. (2025). Group-Algebraic Tensors: Provably-optimal Equivariant Learning and Physical Symmetry Discovery. arXiv:2605.20440.
- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.

## Relation to other certs

- **[128]** Lean 4 pi(9)=24 proof — formal anchor for Z/24Z Cosmos period
- **[261]** Orbit stratification — content-ideal link Cosmos/Satellite/Singularity
- **[499]** Pisano All-Initializations — full Pisano period partition {1,8,24}
- **[500]** Cosmos Chamber — three Cosmos sub-orbits distinguished by G-arithmetic
- **[501]** Algebraic Diversity Observer — G-injectivity = Z/24Z minimal matched group
- **[502]** Collatz-Fibonacci Spectral — Singularity as absorbing fixed point
- **[503]** Witt Tower tau-Monotone Ladder — empirical discrimination ladder

## Scope boundary

**The cert does NOT:**
- Claim ⋆G-SVD optimality for QA operators (Theorem 1 of arXiv:2605.20440)
- Provide a Lean 4 proof of the QA G-module decomposition (only cited)
- Assert continuous equivariant learning performance

**The cert DOES:**
- Certify the G-module direct-sum structure (SGT_1)
- Certify the CRT Z/24Z ≅ Z/3Z × Z/8Z factorization with integer witnesses (SGT_2)
- Certify the Z/8Z antipodal half-period action on Satellite (SGT_3)
- Certify sigma^24 = identity on the full 81-pair state space (SGT_4)
