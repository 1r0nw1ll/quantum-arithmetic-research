# [387] QA Witt Carry Sub-Orbit Invariant

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_carry_sub_orbit_cert_v1/`

## Claim

In **ℤ[φ]/(9) = W₂(GF(9))** (Witt vectors of length 2 over GF(9)):

> The 3 Cosmos sub-orbits are classified by the **Witt carry invariant** J(a,b) — a scalar in {0,1,2} constant on each σ-orbit that encodes which coset of ⟨u_φ⟩ in the 1-unit group U₁ the ρ-conjugate of (a,b) belongs to.

| Check | Result |
|-------|--------|
| THREE_SUB_ORBITS: Cosmos = exactly 3 σ-orbits of period 24 | PASS |
| PHI_ORDER_24: ord(φ)=24; (ℤ[φ]/9)× = ⟨φ⟩ ⊔ 2⟨φ⟩ ⊔ 4⟨φ⟩ | PASS |
| DIRECT_PRODUCT: (ℤ[φ]/9)× = T × U₁ (|T|=8, |U₁|=9, exponent 3) | PASS |
| WITT_CARRY_INVARIANT: J constant on each orbit, takes distinct values {0,1,2} | PASS |
| TEICHMÜLLER_HIT: each orbit hits each of 8 T-classes exactly 3 times | PASS |

## Structure

The unit group decomposes as a direct product:

```
(ℤ[φ]/9)× = T × U₁        (|T|=8, |U₁|=9, gcd(8,9)=1)

T   = {x : x⁸ = 1}         Teichmüller subgroup ≅ ℤ/8
U₁  = {x : x ≡ (1,0) mod 3} 1-unit group ≅ (ℤ/3)²
```

Every unit decomposes uniquely as x = τ(x) · u(x) with τ(x) ∈ T and u(x) ∈ U₁.

The QA Fibonacci step φ decomposes as:
- τ(φ) = (3,7) ∈ T, order 8 (a generator of T)
- u_φ = (7,6) ∈ U₁, order 3

The subgroup ⟨u_φ⟩ = {(1,0), (4,3), (7,6)} has 3 cosets in U₁:

| Coset | Elements | J value |
|-------|----------|---------|
| ⟨u_φ⟩ | {(1,0),(4,3),(7,6)} | 0 |
| (1,3)⟨u_φ⟩ | {(1,3),(4,6),(7,0)} | 1 |
| (1,6)⟨u_φ⟩ | {(1,6),(4,0),(7,3)} | 2 |

## The Witt carry invariant J

For (a,b) ∈ Cosmos, the sub-orbit index is:

```
J(a,b) = coset_idx(ρ⁻¹(a,b)) = coset_idx((b,a))
```

where ρ is the coordinate swap and `coset_idx(x)` = which coset of ⟨u_φ⟩ contains the U₁-component of x.

Explicitly, for x = (a,b) ∈ (ℤ[φ]/9)× :
1. Find τ(x) = unique element of T with τ(x) ≡ x mod 3
2. u(x) = τ(x)⁻¹ · x ∈ U₁; write u(x) = (1+3a', 3b') with a',b' ∈ {0,1,2}
3. coset_idx(x) = (b' − a') mod 3

Apply to ρ-conjugate: J(a,b) = coset_idx((b,a)).

## Key insight: T-class is NOT the distinguishing invariant

All three Cosmos sub-orbits hit the same 8 Teichmüller classes {(a mod 3, b mod 3)} and in the same frequency (3 times each). The distinction lives entirely in the **1-unit layer** — the "Witt carry" — not in the mod-3 image (T-class). This is why cert [385] needed the prime ideal filtration: T-class alone is insufficient to see the sub-orbit structure.

## Connection to Witt vectors

ℤ[φ]/(9) ≅ W₂(GF(9)) as rings (since 3 is inert in ℤ[φ], W₂(GF(9)) is the unique unramified degree-2 extension of ℤ/9). The T × U₁ decomposition is the Witt vector decomposition:
- T = Teichmüller representatives = {τ(a) : a ∈ GF(9)×} (elements satisfying x^9 = x)
- U₁ = 1 + 3·GF(9) = "1-unit" or "principal unit" group (elements reducing to 1 mod 3)
- J = second ghost coordinate (Witt carry component) of the ρ-conjugate

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_carry_sub_orbit_cert_v1
python qa_witt_carry_sub_orbit_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {"THREE_SUB_ORBITS": true, "PHI_ORDER_24": true, "DIRECT_PRODUCT": true, "WITT_CARRY_INVARIANT": true, "TEICHMÜLLER_HIT": true}, ...}`

## Lineage

- Extends **[385]** (orbit = prime ideal filtration — inert p=3)
- Extends **[386]** (inert/split/ramified prime classification)
- Builds on **[291]** (Fibonacci matrix orbit periods — Pisano periods)

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. Springer GTM 67. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — Ch.II §4 (Witt vectors, Teichmüller representatives, 1-unit group structure)
- Neukirch, J. (1999). *Algebraic Number Theory*. ISBN 978-3-540-65399-8 — §II.5 (formal groups, Witt vectors, p-adic unit groups)
- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano periods

## What this cert does NOT claim

- Does not claim W₂(GF(9)) ≅ ℤ/9 (these are different rings; ℤ[φ]/(9) is the unramified degree-2 case)
- Does not claim the invariant J extends to the Satellite or Singularity orbits (those are in the ideal, not units)
- Does not certify the Witt carry for mod-24 (split prime 2 changes the structure)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified ord(φ)=24 in
ℤ[φ]/(9) via a fresh multiplication loop (not reusing validator code) —
exact match to the doc's PHI_ORDER_24 claim. The T×U₁ decomposition,
Witt carry invariant J, and Teichmüller-hit distribution were confirmed
by running the validator itself, which genuinely recomputes the coset
structure rather than hardcoding it. No fixture-trusting gap.
