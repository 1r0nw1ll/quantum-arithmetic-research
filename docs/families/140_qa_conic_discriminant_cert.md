# Family [140] QA_CONIC_DISCRIMINANT_CERT.v1

## One-line summary

Certifies I=C-F as the QA conic discriminant: I>0→hyperbola, I=0→parabola (impossible for integers — requires the silver ratio), I<0→ellipse; the silver-ratio convergents 2/1,5/2,12/5,29/12,... alternate H/E with |I|=1.

## Mathematical content

For any QA direction (d,e) with triple (F,C,G):

```
I = C − F = 2de − (d²−e²)
```

| Sign of I | Conic type | Condition |
|-----------|-----------|-----------|
| I > 0 | **Hyperbola** | C > F, i.e., d/e < 1+√2 |
| I = 0 | **Parabola** | C = F — **impossible for integers** |
| I < 0 | **Ellipse** | C < F, i.e., d/e > 1+√2 |

### Parabola impossibility

I=0 requires 2de = d²-e², i.e., d²-2de-e²=0. Treating this as a quadratic in d/e:

```
(d/e)² − 2(d/e) − 1 = 0
→  d/e = 1 + √2   (the silver ratio ≈ 2.41421...)
```

The discriminant is 8, which is not a perfect square → no rational root → no integer solution. Every primitive integer direction (d,e) falls strictly on one side of the parabolic boundary.

### The silver ratio as the QA conic threshold

The continued fraction expansion of 1+√2 = [2;2,2,2,...] gives convergents:

| Convergent | d/e | I | Type | Note |
|------------|-----|---|------|------|
| 2/1 | 2.000 | +1 | H | Fundamental QA direction |
| 5/2 | 2.500 | -1 | E | Closest integer ellipse |
| 12/5 | 2.400 | +1 | H | Plimpton 322 Row 1 |
| 29/12 | 2.4167 | -1 | E | One step closer |
| 70/29 | 2.4138 | +1 | H | Two steps closer |

Each convergent has **|I|=1** — the closest an integer direction can get to the parabola. The types alternate H/E/H/E/... converging to the irrational boundary from both sides.

### Chromogeometry connection

From cert [125]: F=Qr(d,e) (red quadrance), C=Qg(d,e) (green quadrance).

```
I = Qg − Qr
```

- **I>0 (hyperbolic)**: green dominates — more "rotational" character
- **I<0 (elliptic)**: red dominates — more "stretching" character
- The conic type is literally the question: which color metric is larger?

## Checks

| ID | Description |
|----|-------------|
| CD_1 | schema_version == 'QA_CONIC_DISCRIMINANT_CERT.v1' |
| CD_2 | I=C-F computed correctly for each direction |
| CD_3 | conic_type matches sign(I) |
| CD_4 | F²+C²=G² for each triple |
| CD_PARA | Parabola impossibility block present |
| CD_W | ≥3 witnesses of each type (hyperbola and ellipse) |
| CD_F | Fundamental (d,e)=(2,1) present as hyperbola (I=1) |

## Examples

**Fundamental hyperbola** (2,1): I=4-3=1>0. The 3-4-5 triangle, d/e=2 < 2.414.

**Closest ellipse** (5,2): I=20-21=-1<0. The 21-20-29 triple, d/e=2.5 > 2.414.

**Plimpton 322 Row 1** (12,5): I=120-119=1>0. Just barely hyperbolic — d/e=2.4, only 0.014 below the silver ratio. The Babylonians arranged their table with Row 1 near (but not at) the parabola boundary.

## Connection to other families

- **[125] QA_CHROMOGEOMETRY_CERT.v1**: I=Qg-Qr is the difference of green and red chromogeometric quadrances
- **[137] QA_KOENIG_TWISTED_SQUARES_CERT.v1**: I=C-F appears there as the inner Koenig square element; this cert makes its geometric meaning first-class
- **[138] QA_PLIMPTON322_CERT.v1**: Row 1 (12,5) has I=1 — barely hyperbolic; the parabola boundary explains why the Plimpton table starts near that direction
- **[127] QA_UHG_NULL_CERT.v1**: the conic type of a null point [F:C:G] in UHG is determined by the position relative to the null conic — I encodes which side

## Fixture files

- `fixtures/cd_pass_fundamental.json` — 4 directions straddling the silver ratio; parabola impossibility proof
- `fixtures/cd_pass_witnesses.json` — 4 hyperbola + 4 ellipse witnesses + convergent sequence to silver ratio
