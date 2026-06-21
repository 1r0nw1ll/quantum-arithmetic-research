# [496] QA-E8 Satellite Chamber Theorem

**Cert slug**: `qa_e8_satellite_chamber_cert_v1`  
**Family ID**: 496  
**Derived**: 2026-06-21

## Theorem Statement

For QA mod m=9, the Satellite orbit has 8 states with period 8, canonically anchored at (6,3) — the unique step yielding the primitive (3,4,5) triple with C<F. Label the 8 axes e₁,...,e₈ in Satellite order starting from (6,3).

For height function h = α·d² + β·e² (α,β > 0), h lies in the E₈ Weyl chamber where **(6,3) is the branch node** and **(3,6) [Grant LRT] is the terminal leaf of the long arm** (distance 4 from branch) **if and only if**:

> **7/12 < α/β < 13/12**

The **unique isotropic** choice α = β gives α/β = 1 ∈ (7/12, 13/12).  
Normalizing α = β = 1: **h = G = d² + e²** (Wildberger quadrance).

## Sub-Claims

| Check | Claim |
|-------|-------|
| ESC_PYTH | C² + F² = G² for all 576 QA pairs (b,e) ∈ {1..9}² |
| ESC_PARITY | b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs |
| ESC_ROOTS | E₈ root system: 112 Type-1 (±2 entries) + 128 Type-2 (±1, even −) = 240 |
| ESC_GRAM | G-chamber simple system Cartan matrix det = 1 |
| ESC_BRANCH | h = G places (6,3) at E₈ branch node (degree-3 node) |
| ESC_GRANT | (3,6) [Grant LRT] is at distance 4 from branch under h = G |
| ESC_WALL_LOWER | WALL_LOWER·G = +45 > 0 → lower bound α/β > 7/12 |
| ESC_WALL_UPPER | WALL_UPPER·G = −9 < 0 → upper bound α/β < 13/12 |
| ESC_ISO_INTERVAL | α/β = 1 ∈ (7/12, 13/12) strictly |
| ESC_G2_EXITS | G² has opposite sign on 3 Type-2 wall roots (G-chamber ≠ G²-chamber despite same axis ordering) |
| ESC_ELEM_UNIQUE | G is unique in {b,e,d,a,C,F,G} with strict ordering + E₈ genericity + branch=(6,3) |

## Wall Roots

**Lower wall** (root `[-1,-1,+1,-1,+1,-1,+1,+1]`):  
A = 108·α − 63·β > 0 → α/β > 63/108 = **7/12**

**Upper wall** (root `[+1,-1,+1,-1,−1,−1,+1,+1]`):  
A = 108·α − 117·β < 0 → α/β < 117/108 = **13/12**

Both binding walls have the same A-coefficient (108), with B-values −63 (lower) and −117 (upper).  
The center of the interval (in A-B space) is at −(−63−117)/(2·108) = 90/108 = 5/6 ≈ 0.833.  
The isotropic point (ratio = 1) lies at 0.833 + 0.167, interior to the interval.

## E₈ Dynkin Diagram Under h = G

```
(9,3)—(3,3)—(6,3)—(9,6)
               |
            (6,6)—(3,9)—(6,9)—(3,6)
         [b=e diag]        [Grant LRT]
```

- **(6,3)** = branch node (degree 3), G = 90 (smallest on long arm)
- **(3,6)** = Grant LRT, (12,5,13) triple, distance 4 from branch
- Short arm: (6,3) → (9,6)
- Medium arm: (6,3) → (3,3) → (9,3)
- Long arm: (6,3) → (6,6) → (3,9) → (6,9) → (3,6)

## Elementary Uniqueness Selection Chain

```
{b,e,d,a,C,F,G}
  → {C,F,G}    (strict ordering: b,e,d,a have repeated values on Satellite)
  → {F,G}      (genericity: C has zero-projection Type-2 root)
  → {G}        (correct branch: F places branch at (9,6), G places it at (6,3))
```

## G vs G² (Metric Sensitivity)

G = (90,225,153,45,117,306,261,180) and G² = (8100,50625,23409,2025,13689,93636,68121,32400) give **identical Satellite axis orderings** but are in **different Weyl chambers**: 3 Type-2 wall roots change sign:

| Root | G·r | G²·r |
|------|-----|------|
| `[-1,-1,+1,-1,+1,-1,+1,+1]` | +45 | −16767 |
| `[-1,-1,+1,+1,-1,+1,-1,+1]` | −9 | +10935 |
| `[+1,+1,+1,+1,+1,-1,-1,+1]` | +243 | −31509 |

This proves chamber selection depends on metric values, not axis order alone — making d²+e² (Wildberger quadrance) geometrically canonical.

## Primary Sources

- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.
- Bourbaki (1968). *Groupes et algèbres de Lie*, Ch. 4–6.
- Humphreys, J.E. (1972). *Introduction to Lie Algebras and Representation Theory*. Springer. ISBN 978-0-387-90053-7.

## Parents

[244] (QA–E8 Orbit Embedding), [249] (E8 Embedding Orbit Classifier), [250] (ADE Mutation Game)
