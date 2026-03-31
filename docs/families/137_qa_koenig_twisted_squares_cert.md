# Family [137] QA_KOENIG_TWISTED_SQUARES_CERT.v1

## One-line summary

Certifies the (I²,2CF,G²,H²) arithmetic progression structure: H²-G²=G²-I²=2CF=24L for any QA direction, where H=C+F and I=C-F are the outer and inner Koenig square elements.

## Mathematical content

For any QA direction (d,e) with triple (F,C,G):

```
H = C + F   (outer Koenig square)
I = C − F   (inner Koenig square; sign = conic type)
L = CF/12   (QA L-element, always integer for primitive direction)
```

**Identities:**
```
H² − G² = 2CF = 24L
G² − I² = 2CF = 24L
```

**Corollary — arithmetic progression:**
```
(I², 2CF, G², H²)  with step 2CF
```
i.e., I² + 2CF = G² and G² + 2CF = H².

### Algebraic proof

```
H² − G² = (C+F)² − (C²+F²) = 2CF     [using C²+F² = G²]
G² − I² = (C²+F²) − (C−F)² = 2CF     [same]
L = CF/12 ∈ Z:
  • 8 | C = 2de  (d−e odd → exactly one of d,e even → 4|de → 8|2de)
  • 3 | F = (d−e)(d+e)  (gcd(d,e)=1 → one of d−e, d+e divisible by 3)
  → 24 | CF/1  (but CF = 12L so 2CF = 24L ✓)
```

### Fundamental example: (d,e)=(2,1), triple (3,4,5)

```
H = C+F = 4+3 = 7   (Will: "outer square side")
I = C−F = 4−3 = 1   (inner square side; I>0 → hyperbola)
L = CF/12 = 12/12 = 1

H² − G² = 49 − 25 = 24 = 2·4·3 = 2CF ✓
G² − I² = 25 −  1 = 24 = 2·4·3 = 2CF ✓

Quadruple: (1, 24, 25, 49) — step 24
```

The number 24 here is the **origin of mod-24** (certified independently in [130]): every primitive Pythagorean direction has 2CF divisible by 24, and the minimum is 24 at the fundamental (2,1).

### Conic type from I

| Sign of I | Conic type |
|-----------|-----------|
| I > 0     | Hyperbola |
| I = 0     | Parabola  |
| I < 0     | Ellipse   |

(From elements.txt: I=C−F; I<0=ellipse, I=0=parabola, I>0=hyperbola.)

## Checks

| ID | Description |
|----|-------------|
| KTS_1 | schema_version == 'QA_KOENIG_TWISTED_SQUARES_CERT.v1' |
| KTS_2 | F=d²-e², C=2de, G=d²+e² for declared triple |
| KTS_3 | F²+C²=G² for triple |
| KTS_4 | H=C+F, I=C-F computed correctly |
| KTS_5 | H²-G² = 2·C·F |
| KTS_6 | G²-I² = 2·C·F |
| KTS_7 | L=C·F/12 is integer |
| KTS_8 | 24·L = 2·C·F |
| KTS_9 | 2·C·F ≡ 0 (mod 24) |
| KTS_W | ≥3 witness entries |
| KTS_F | Fundamental (d,e)=(2,1) present with 2CF=24 |

## Geometric interpretation — Twisted Squares

The name comes from the "twisted squares" construction (Mathologer, 2024):

Place a square of side (d+e)=√(H²) rotated 45° inside a square of side (d+e). The four right triangles in the corners each have legs d and e, area=de. The inner square has side |d−e|. The outer square has side d+e.

But in QA, H=C+F and I=C−F are QA *elements* (not the geometric sides d±e). The QA identity H²−G²=2CF is the algebraic shadow of the twisted-squares area formula:

```
outer_area − circle_inscribed = 4 × triangle_area
(C+F)²     − (C²+F²)         = 2CF  =  4 × (CF/2)
```

where CF/2 is the "triangle area" in QA element space.

## Koenig chain

The Koenig series generates a chain of directions by taking H and I as seeds for the next step. From (d,e)=(2,1):
```
H=7, |I|=1  →  next direction contains (7,1)
```

This generates all prime-producing triples. The chain structure means every triple in the Koenig series satisfies: its H²−G² equals 24 times its L, connecting the arithmetic divisibility to the geometric construction.

## Historical chain

| Date | Source | Contribution |
|------|--------|-------------|
| Antiquity | Babylonian | Pythagorean triples (F,C,G) |
| QA Law 15 | Iverson (Pyth-1) | L=CF/12, W=d(e+a); H²−I²=48L identified |
| 2024 | Mathologer | Twisted-squares video: outer²−inner²=4×area |
| 2026-03-30 | Will Dale | (I²,2CF,G²,H²) QA quadruple corollary; H=C+F=outer, I=C−F=inner |

## Connection to other families

- **[130] QA_ORIGIN_OF_24_CERT.v1**: certifies H²−G²=2CF=24L is always divisible by 24 for primitive directions. This cert certifies the full arithmetic progression structure.
- **[134] QA_EGYPTIAN_FRACTION_CERT.v1**: the Koenig descent path passes through the twisted-squares structure.
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: the Barning-Hall tree generates all (d,e) whose twistedness (2CF) is certified here.
- **[125] QA_CHROMOGEOMETRY_CERT.v1**: 2CF = 2·Qg·Qr = double the product of the green and red quadrances.

## Fixture files

- `fixtures/kts_pass_fundamental.json` — anchor: (2,1), 2CF=24, quadruple=(1,24,25,49)
- `fixtures/kts_pass_witnesses.json` — 5 witnesses including both ellipse (I<0) and hyperbola (I>0) directions + Koenig chain example
