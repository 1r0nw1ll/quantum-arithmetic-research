# Family [141] QA_PELL_NORM_CERT.v1

## One-line summary

Certifies I=C-F=-(x²-2y²) where x=d-e, y=e: the QA conic discriminant is the negated Pell norm, and the M_B Pythagorean tree move is the Pell-sign-flip map.

## Mathematical content

### The identity

For any QA direction (d,e), let **x = d−e** and **y = e**. Then:

```
I = C − F = -(x² − 2y²) = -P(d−e, e)
```

where P(x,y) = x²−2y² is the **Pell norm** for D=2.

**Proof:**
```
I = 2de − (d²−e²)
  = 2(x+y)y − ((x+y)²−y²)       [substituting d=x+y]
  = 2xy+2y² − (x²+2xy+y²−y²)
  = 2y² − x²
  = -(x²−2y²)  ✓
```

### The Pell boundary

The Pell equation x²−2y²=±1 gives the integer directions closest to the parabola boundary (from cert [140]):

| Pell norm | Meaning | I | Conic |
|-----------|---------|---|-------|
| P = -1 | Half-Pell solution | I = +1 | Hyperbola boundary |
| P = 0 | Impossible for primitive integers | I = 0 | Parabola (impossible) |
| P = +1 | Full Pell solution | I = -1 | Ellipse boundary |

**Fundamental**: (x,y)=(1,1) satisfies x²−2y²=−1 (half-Pell), giving direction (d,e)=(2,1), I=+1 — the QA fundamental direction is the simplest Pell solution.

### The M_B map is the Pell-sign-flip

The Pythagorean tree M_B move (cert [135]): M_B(d,e) = (2d+e, d).

In Pell variables (x,y)=(d−e,e), this becomes:

```
x' = (2d+e)−d = d+e = (x+y)+y = x+2y
y' = d = x+y

P(x',y') = (x+2y)² − 2(x+y)²
         = x²+4xy+4y² − 2x²−4xy−2y²
         = 2y² − x²
         = -(x²−2y²) = -P(x,y)  ✓
```

**M_B flips the Pell sign.** Starting from (2,1) with P=−1:

| Direction | x | y | Pell P | I | Type |
|-----------|---|---|--------|---|------|
| (2,1)   | 1  | 1  | -1 | +1 | H |
| (5,2)   | 3  | 2  | +1 | -1 | E |
| (12,5)  | 7  | 5  | -1 | +1 | H |
| (29,12) | 17 | 12 | +1 | -1 | E |
| (70,29) | 41 | 29 | -1 | +1 | H |
| (169,70)| 99 | 70 | +1 | -1 | E |

This is the **Pell equation solution sequence** for x²−2y²=±1, generated entirely by the QA M_B tree move.

### Pell norm levels

For general (d,e), |P| measures how far the direction is from the parabola boundary:

- |P|=1: boundary tier (certs [140] convergents)
- |P|=7: next tier — includes (3,2) H, (4,1) E, (9,4) H, (8,3) E
- |P|=17: next — includes (4,3) H, ...

Directions at the same |P| level are "equidistant" from the parabola in Pell norm units.

## Checks

| ID | Description |
|----|-------------|
| PN_1 | schema_version == 'QA_PELL_NORM_CERT.v1' |
| PN_2 | F=d²-e², C=2de, G=d²+e² |
| PN_3 | F²+C²=G² |
| PN_IDEN | I = -(x²-2y²) where x=d-e, y=e |
| PN_MB | Consecutive chain entries: (d₁,e₁)=M_B(d₀,e₀) and P₁=-P₀ |
| PN_W | ≥3 general witnesses |
| PN_F | Fundamental (2,1): P=-1, I=1 |

## Connection to other families

- **[140] QA_CONIC_DISCRIMINANT_CERT.v1**: I=C-F is the discriminant; this cert gives its algebraic origin as a Pell norm
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: M_B is the Pell-flip map; the M_B spine of the tree is the Pell solution sequence
- **[138] QA_PLIMPTON322_CERT.v1**: Row 1 (12,5) is a Pell-1 direction (x=7,y=5, P=-1, I=1); the Babylonians started their table at a Pell boundary point
- **[139] QA_48_64_CERT.v1**: at the Pell boundary |P|=1, the discriminant |I|=1 — the minimum non-zero distance

## Wildberger connection

Wildberger arXiv:0806.2490 ("Pell's equation and the Pell group") establishes that Pell solutions correspond to the isometry group structure of the real number line in rational geometry — directly connecting to the QA direction classification.

## Fixture files

- `fixtures/pn_pass_fundamental.json` — identity proof, M_B Pell chain 6 steps, algebraic proofs
- `fixtures/pn_pass_witnesses.json` — 6 general witnesses at various Pell norm levels
