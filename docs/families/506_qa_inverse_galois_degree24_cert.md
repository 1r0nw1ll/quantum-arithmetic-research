# [506] QA Inverse Galois Degree-24 Certificate

**Schema**: `QA_INVERSE_GALOIS_DEGREE24_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_inverse_galois_degree24_cert_v1/`
**Status**: PASS
**Added**: 2026-06-23
**Primary source**: Tao T. et al. (2026) SAIR IGP24 competition — inverse Galois problem for degree-24 transitive groups.

## Competition Context

SAIR Inverse Galois Problem degree-24 challenge, launched June 16, 2026. Stage 1 closes August 15, 2026. Verification tool: Magma. Baseline: 622 of 165,836 (group, signature) pairs resolved at launch. The competition asks: given a transitive subgroup G ≤ S₂₄, find a degree-24 polynomial over Q with Gal(f/Q) ≅ G (as a transitive action on 24 roots).

## Claim

QA's sigma operator on the mod-9 Cosmos orbit provides a concrete realization of the cyclic group C₂₄ as a degree-24 transitive subgroup of S₂₄ — the regular representation.

**IG_1 — degree-24 cyclic action**
sigma restricted to any Cosmos sub-orbit is a 24-cycle in S₂₄:
- orbit length = 24; sigma^24(1,1) = (1,1); sigma^k(1,1) ≠ (1,1) for 0 < k < 24
- Verified for all 3 Cosmos sub-orbits
- Degree = 24; group order = 24 (regular: |stabilizer| = 1)

**IG_2 — regular representation = 0 fixed points**
sigma^k for 0 < k < 24 has no fixed points in any Cosmos sub-orbit. The 24-cycle representation has cycle structure (24) — a single full cycle. Fixed-point count = 0 across all 3 sub-orbits × 23 non-identity powers = 69 checks, all 0.

**IG_3 — 3 independent C₂₄ copies**
The full Cosmos class (72 pairs) decomposes into exactly 3 sub-orbits of length 24: 72 / 24 = 3 = |C₃| = the coset count. Under the CRT C₂₄ ≅ C₃ × C₈, the 3 sub-orbits correspond to the 3 cosets of the C₈ subgroup in C₂₄.

**IG_4 — CRT decomposition C₂₄ ≅ C₃ × C₈**
- sigma^8 restricted to each Cosmos sub-orbit has period 3
- sigma^3 restricted to each Cosmos sub-orbit has period 8
- gcd(3, 8) = 1 and 3 × 8 = 24 — Chinese Remainder factorization

This is the algebraic structure visible in the LMFDB degree-24 group classification: C₂₄ decomposes as a direct product of two coprime cyclic factors.

## Theorem NT Compliance

Galois group characters (roots of unity), splitting field constructions, and Magma's GaloisGroup() function are continuous/symbolic observer projections over the discrete C₂₄ group structure. QA certifies the discrete structure (orbit length, fixed-point count, CRT periods) via integer sigma iteration only.

NOT CLAIMED: specific polynomial over Q with Gal(f/Q) ≅ C₂₄; LMFDB T(24,k) entry disambiguation; degree-24 transitive group enumeration beyond C₂₄.

## Companion Certs

- [128] QA π(9) = 24 Lean proof — Cosmos orbit has period exactly 24
- [499] All-initializations orbit classification — Cosmos 72 pairs period 24, exhaustive
- [504] Star-G tensor CRT — C₂₄ ≅ C₃ × C₈ certified (SGT_2); sigma^8 period 3; sigma^3 period 8

## Fixtures

| File | Expected | Checks |
|------|----------|--------|
| `pass_inverse_galois_degree24.json` | PASS | All four IG claims |
| `fail_wrong_orbit_length.json` | FAIL | IG_1: cosmos_orbit_length=12 (not 24) |
| `fail_fixed_points_present.json` | FAIL | IG_2: cosmos_fixed_points_nonidentity=3 (not 0) |
| `fail_wrong_crt_period.json` | FAIL | IG_4: sigma8_cosmos_period=4 (not 3; C₄×C₈ not C₃×C₈) |
