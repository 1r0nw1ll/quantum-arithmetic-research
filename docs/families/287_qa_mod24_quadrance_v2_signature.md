# [287] QA Mod-24 Quadrance 2-adic Signature Cert

**Family ID**: 287  
**Slug**: `qa_mod24_quadrance_v2_signature_cert_v1`  
**Status**: Active  
**Registered**: 2026-05-31

## Claim (narrow, falsifiable)

For all pairs (b, e) in {1, ..., 24}², the 2-adic valuation of the quadrance G = b² + e² satisfies:

```
v₂(G) = 2·min(v₂(b), v₂(e)) + δ
```

where δ = 1 if v₂(b) = v₂(e), else δ = 0.

Equivalently, the orbit class of (b, e) under mod-24 QA separates v₂(G):

- **Cosmos** (v₂(b) or v₂(e) ≤ 2): v₂(G) ≤ 5
- **Satellite** (8|b AND 8|e, not singularity): v₂(G) ≥ 6
- **Singularity** (b=24, e=24): v₂(G) ≥ 6

Verified exhaustively for all 576 pairs in {1, ..., 24}².

## Algebraic Mechanism

The δ = 1 diagonal enhancement arises from a prime-specific arithmetic fact: odd squares satisfy x² ≡ 1 (mod 8). When v₂(b) = v₂(e) = k, write b = 2^k·b'', e = 2^k·e'' with b'', e'' odd. Then:

```
G = 4^k·(b''² + e''²)
b''² ≡ 1 (mod 8),  e''² ≡ 1 (mod 8)
b''² + e''² ≡ 2 (mod 8)  →  v₂(b''² + e''²) = 1
v₂(G) = 2k + 1
```

When v₂(b) ≠ v₂(e), the smaller-valuation term dominates and v₂(G) = 2·min(v₂(b), v₂(e)).

## Contrast with Mod-9 Cert [283]

Cert [283] proves v₃(G) = 2·v₃(gcd(b, e)) for mod-9 pairs — with **no δ term**. This is because nonzero mod-3 residues square to 1 and 1 + 1 = 2, which is coprime to 3, so no extra factor of 3 arises. The δ = 1 enhancement is **2-adic specific**, driven by the fact that 2 divides 1 + 1 = 2 (mod 8).

## Tightness Witnesses

- Cosmos max: v₂(G) = 5 at (b, e) = (4, 4) — G = 32 = 2⁵
- Satellite min: v₂(G) = 6 at (b, e) = (8, 16) — G = 320 = 2⁶·5
- Satellite diagonal max: v₂(G) = 9 at (b, e) = (16, 16) — G = 512 = 2⁹
- Singularity: v₂(G) = 7 at (b, e) = (24, 24) — G = 1152 = 2⁷·9

## Primary Sources

- Wildberger, N. J. (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry*. Wild Egg Books. ISBN 978-0-9757492-0-8. Chapter 1: quadrance Q(A,B) = (x₂−x₁)² + (y₂−y₁)²; for QA direction vector (b,e), G = b² + e².
- Wall, D. D. (1960). Fibonacci series modulo m. *Amer. Math. Monthly*, 67(6). DOI: 10.1080/00029890.1960.11989541. Orbit period classification.

## Mechanism Chain

- [279] QA Orbit Access Theorem — provides orbit_family classifier on (b, e, 24)
- [283] QA RT Quadrance Orbit Divisibility Cert (mod-9) — established the v_p(G) = 2·v_p(gcd) pattern; this cert shows the 2-adic case departs by δ = 1

## Checks

| ID     | Description                                                    |
|--------|----------------------------------------------------------------|
| V2Q_1  | v₂(G) formula: 2·min(v₂(b),v₂(e)) + δ                       |
| V2Q_2  | Cosmos threshold: v₂(G) ≤ 5                                   |
| V2Q_3  | Satellite/Singularity threshold: v₂(G) ≥ 6                    |
| V2Q_4  | Equal-valuation diagonal enhancement: v₂(b)=v₂(e) → δ=1     |
| V2Q_5  | Unequal-valuation: v₂(b)≠v₂(e) → δ=0                        |
| SRC    | Primary-source exempt marker present                           |
| F      | Fixture format and schema validation                           |

**Fixtures**: 6 PASS + 4 FAIL  
**Self-test**: exhaustive sweep of all 576 pairs in {1,...,24}²
