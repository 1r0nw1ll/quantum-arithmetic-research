# IGP24 Competition — QA Submission Package

**Competition**: SAIR Inverse Galois Problem degree-24  
**Platform**: `competition.sair.foundation/competitions/igp24/`  
**Stage 1 deadline**: 2026-08-15  
**Scoring**: smaller field discriminant = higher score per group  
**Base solved at launch**: 286 of 25,000 groups (622 of 165,836 (group,sig) pairs)

## Quick Start

1. **Register** at `competition.sair.foundation` (manual step)
2. **Get API token** from your profile page
3. `export IGP24_TOKEN="your-token"` then `python3 submit.py --submit`

Or copy the Magma script for manual portal entry:
```bash
python3 submit.py --magma
```

## Our Polynomials

### 24T1 — C₂₄ (cyclic order 24) ⭐ QA-native

**Construction**: compositum K₃ ⊗ K₈ via CRT C₂₄ ≅ C₃ × C₈  
- K₃ = Q(2cos(2π/7)), min poly x³+x²−2x−1, conductor 7  
- K₈ = Q(2cos(2π/17)), min poly x⁸+x⁷−7x⁶−…, conductor 17  
- Generator α = 2cos(2π/7) + 2cos(2π/17)  
- f(x) = Res_y(p₃(y), p₈(x−y))

**Field discriminant**: 7¹⁶ × 17²¹ (log₁₀ ≈ 39.36)  
**Optimality**: minimal for totally real C₂₄ by conductor-discriminant formula

**QA connection**: C₂₄ is QA's Cosmos orbit period (cert [128], [506]). The CRT
factorization C₂₄ ≅ C₃ × C₈ is certified in [504]. This construction directly
instantiates the cert-[506] claim as a number field.

**Prime splitting** (confirms C₂₄):
- mod 7: one degree-8 factor, multiplicity 3 → inertia = C₃ (totally ramified in K₃)
- mod 17: one degree-3 factor, multiplicity 8 → inertia = C₈ (totally ramified in K₈)
- Together: C₃ × C₈ = C₂₄ ✓

### 24T2 — C₁₂ × C₂

Φ₃₉(x), conductor 39 = 3×13.  
Gal(Q(ζ₃₉)/Q) ≅ (Z/3Z)* × (Z/13Z)* ≅ C₂ × C₁₂.

### 24T3 — C₂² × C₆

Φ₅₆(x), conductor 56 = 8×7.  
Gal(Q(ζ₅₆)/Q) ≅ (Z/8Z)* × (Z/7Z)* ≅ C₂ × C₂ × C₆.

## Strategy Notes

**Abelian cases (24T1/T2/T3)**: Almost certainly in the pre-existing 286 solutions
(LMFDB has had these for years). Our value-add is the optimal discriminant for 24T1
(7¹⁶ × 17²¹). Submit anyway — if existing solution has larger disc, we score.

**Non-abelian targets**: The real competition is in the 24,714 unsolved groups.
Priority candidates (per QA cert ecosystem):
- Nilpotent degree-24 groups reachable via iterated abelian tower
- Groups related to the Langlands certs [403]–[418]
- Groups with structure visible in QA orbit decompositions

**Second SAIR competition** (more QA-relevant): Modular Arithmetic Challenge at
`competition.sair.foundation/competitions/modular-arithmetic-challenge`  
Task: neural network for a·b mod p (prime p up to 1000 digits, 1100 problems, 5 min).
QA modular arithmetic framework is directly applicable here.

## Files

| File | Purpose |
|------|---------|
| `polynomials.py` | Polynomial definitions + verification |
| `submit.py` | API client (set `IGP24_TOKEN` env var) |
| `README.md` | This file |
