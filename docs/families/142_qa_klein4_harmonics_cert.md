# Family [142] QA_KLEIN4_HARMONICS_CERT.v1

## One-line summary

The four sign-changes of (F,C,G) form a Klein 4-group K4=Z₂×Z₂ that preserves F²+C²=G² and permutes the harmonic packet {H, I, −H, −I}.

## Mathematical content

### The four operations

For any QA triple (F,C,G) = (d²−e², 2de, d²+e²):

| Element | Action on (F,C) | Action on (H,I) | Coordinate |
|---------|----------------|-----------------|------------|
| I₀ | (F, C) → (F, C) | (H,I) → (H, I) | (d,e)→(d,e) |
| I₁ | (F, C) → (−F, C) | (H,I) → (I, H) | (d,e)→(e,d) |
| I₂ | (F, C) → (F, −C) | (H,I) → (−I, −H) | (d,e)→(d,−e) |
| I₃ | (F, C) → (−F, −C) | (H,I) → (−H, −I) | (d,e)→(e,−d) |

where H = C+F, I = C−F.

### Klein 4-group table

```
∘   I₀  I₁  I₂  I₃
I₀  I₀  I₁  I₂  I₃
I₁  I₁  I₀  I₃  I₂
I₂  I₂  I₃  I₀  I₁
I₃  I₃  I₂  I₁  I₀
```

Every element is its own inverse. I₁∘I₂ = I₃. The group is abelian (Z₂×Z₂).

### Pythagorean invariance

**Proof:** F²+C² = G². Sign changes are transparent to squares:
- (−F)²+C² = F²+C² = G² ✓
- F²+(−C)² = F²+C² = G² ✓
- (−F)²+(−C)² = F²+C² = G² ✓

All four operations preserve the null/Pythagorean condition.

### Harmonic action proofs

**I₁ (F→−F):** H' = C+(−F) = C−F = I, I' = C−(−F) = C+F = H → swaps H↔I ✓

**I₂ (C→−C):** H' = (−C)+F = −(C−F) = −I, I' = (−C)−F = −(C+F) = −H → (H,I)→(−I,−H) ✓

**I₃ (F→−F, C→−C):** H' = (−C)+(−F) = −(C+F) = −H, I' = (−C)−(−F) = F−C = −(C−F) = −I → (H,I)→(−H,−I) ✓

### The K4 orbit is the complete harmonic packet

For any direction (d,e), the K4 orbit of (H,I) is exactly:

```
{ (H,I), (I,H), (−I,−H), (−H,−I) }
```

These are the four "corners" of the harmonic packet associated with (d,e).

### Fundamental example: (d,e)=(2,1)

(F,C,G) = (3,4,5), H=7, I=1. Harmonic orbit:

| Element | (H',I') | Triple (F',C',G) |
|---------|---------|-----------------|
| I₀ | (7, 1) | (3, 4, 5) |
| I₁ | (1, 7) | (−3, 4, 5) |
| I₂ | (−1, −7) | (3, −4, 5) |
| I₃ | (−7, −1) | (−3, −4, 5) |

All satisfy F'²+C'²=25=G². The orbit {(7,1),(1,7),(−1,−7),(−7,−1)} lives on the circle of radius G=5 in (H,I)-space.

### Coordinate interpretation of I₁

I₁ is the only physically natural operation: (d,e)→(e,d) swaps the direction coordinates. This:
- maps F=d²−e² → e²−d² = −F (changes sign of red quadrance)
- leaves C=2de = 2ed = C (green quadrance unchanged)
- leaves G=d²+e² = e²+d² = G (blue quadrance unchanged)

I₂ and I₃ are formal algebraic operations (require negative e, which is non-physical in QA), but they are algebraically well-defined and preserve F²+C²=G².

### Connection to conic type

I₁ swaps H↔I. Since I=C−F is the conic discriminant (cert [140]):
- I>0 (hyperbolic direction) maps to I'=H>0 (also hyperbolic, with H>I)
- I<0 (elliptic direction) maps to I'=H<0 (the negated outer-square value)

I₁ is not a conic-type flip — it is a harmonic packet symmetry.

### Pell boundary structure

At the Pell boundary |I|=1 (cert [141]):
- (2,1): H=7, I=1 → K4 orbit {(7,1),(1,7),(−1,−7),(−7,−1)}
- (5,2): H=41, I=−1 → K4 orbit {(41,−1),(−1,41),(1,−41),(−41,1)}

The I₁ image of the (2,1) fundamental (H=7,I=1) is (H',I')=(1,7), which is exactly the harmonic packet of the (2,1) direction under I₁.

## Checks

| ID | Description |
|----|-------------|
| K4_1 | schema_version == 'QA_KLEIN4_HARMONICS_CERT.v1' |
| K4_2 | F=d²−e², C=2de, G=d²+e², F²+C²=G² |
| K4_3 | Group table 4×4 matches Z₂×Z₂ |
| K4_ACT | All four K4 images of (F,C,G) satisfy F'²+C'²=G² |
| K4_HARM | I₁ swaps H↔I; I₂: (H,I)→(−I,−H); I₃: (H,I)→(−H,−I) |
| K4_W | ≥3 direction witnesses |
| K4_F | Fundamental (2,1): H=7, I=1; orbit {(7,1),(1,7),(−1,−7),(−7,−1)} |

## Connection to other families

- **[137] QA_KOENIG_TWISTED_SQUARES_CERT.v1**: H=C+F and I=C−F; K4 acts on the twisted-squares harmonic pair
- **[125] QA_CHROMOGEOMETRY_CERT.v1**: F=Qr (red), C=Qg (green); I₁ = red reflection; I₂ = green reflection
- **[140] QA_CONIC_DISCRIMINANT_CERT.v1**: I=C−F is the discriminant; K4 permutes discriminant ±values
- **[141] QA_PELL_NORM_CERT.v1**: I=−(x²−2y²); I₁ maps I→H, equivalent to negating the Pell norm
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: I₁ corresponds to (d,e)→(e,d), which is related to M_A and M_C tree moves

## Source grounding

- **elements.txt** (Dale/Ben): H and I defined as H=C+F, I=C−F in the 26-element table
- **[137] cert**: H²−G²=G²−I²=2CF — H and I appear symmetrically in the Koenig identity
- **QA axiom S1**: F=d\*d−e\*e (never d\*\*2); C=2\*d\*e — purely integer arithmetic

## Fixture files

- `fixtures/k4_pass_group_axioms.json` — group table + algebraic proofs + 3 witnesses
- `fixtures/k4_pass_witnesses.json` — 6 general witnesses at H/E/Pell-boundary/large
