# Family [145] QA_PATH_SHAPE_CERT.v1

## One-line summary

Generator sequences in the Pythagorean tree partition into four shape classes — UNIFORM_A, UNIFORM_B, UNIFORM_C, MIXED — each with distinct structural invariants.

## Mathematical content

### The Pythagorean tree generators

Three moves act on primitive directions (d,e) with d>e>0, gcd(d,e)=1:

| Move | Formula | Effect |
|------|---------|--------|
| M_A | (2d−e, d) | Slow growth; consecutive integers |
| M_B | (2d+e, d) | Pell chain; exponential growth |
| M_C | (d+2e, e) | Arithmetic stride; e constant |

Every primitive Pythagorean direction is reachable from root (2,1) by a unique sequence of these moves (Barning 1963 / Hall 1970).

### Shape classes

A **path shape** is the generator sequence (g₁, g₂, ..., gₙ). Four classes:

**UNIFORM_A** — only M_A: (2,1)→(3,2)→(4,3)→(5,4)→... Each step increments: (d,e)→(d+1,d). Produces consecutive-integer directions.

**UNIFORM_B** — only M_B: (2,1)→(5,2)→(12,5)→(29,12)→... The Pell chain. Pell norm P(d−e, e) = (d−e)²−2e² alternates sign at every step (cert [141]).

**UNIFORM_C** — only M_C: (2,1)→(4,1)→(6,1)→(8,1)→... e remains constant; d grows by +2e each step (arithmetic progression).

**MIXED** — two or more distinct generators. Most Pythagorean tree paths. Example: (2,1)→M_B→(5,2)→M_A→(8,5)→M_C→(18,5).

### Invariants

| Class | Invariant |
|-------|-----------|
| ALL | Primitivity: gcd(d,e)=1 preserved by all three generators |
| ALL | F²+C²=G² at every step (identity for QA directions) |
| UNIFORM_B | Pell norm alternates: P_{n+1} = −P_n |
| UNIFORM_C | e constant along entire path |
| UNIFORM_A | d/e → 1 (consecutive integers) |

## Checks

| ID | Description |
|----|-------------|
| PS_1 | schema_version == 'QA_PATH_SHAPE_CERT.v1' |
| PS_2 | all directions d>e>0, gcd=1; steps recomputed correctly |
| PS_CLASS | declared shape class matches actual generator sequence |
| PS_INV_B | Pell norm alternates for UNIFORM_B paths |
| PS_INV_C | e constant for UNIFORM_C paths |
| PS_W | ≥4 paths total (one per shape class) |
| PS_F | fundamental root (2,1) present |

## Source grounding

- **Barning (1963)** / **Hall (1970)**: three generators produce all primitive Pythagorean triples as a ternary tree
- **Price (2008)**: Fibonacci-box connection; M_A/M_B/M_C linked to k-identification
- **Ben Iverson**: Koenig series = M_A descent direction
- **Cert [135]** QA_PYTHAGOREAN_TREE_CERT.v1: the three generators and their properties
- **Cert [134]** QA_EGYPTIAN_FRACTION_CERT.v1: k-sequence = path encoding
- **Cert [141]** QA_PELL_NORM_CERT.v1: M_B = Pell-flip; Pell norm alternation

## Connection to other families

- **[135] Pythagorean Tree**: defines M_A/M_B/M_C; this cert classifies the paths they generate
- **[134] Egyptian Fraction**: k-sequence (k₁,k₂,...) = shape encoding in a different alphabet
- **[141] Pell Norm**: UNIFORM_B invariant (Pell alternation) is the Pell norm cert's content
- **[137] Koenig Twisted Squares**: Koenig descent = inverse of UNIFORM_A
- **[146] Path Scale**: companion cert — shape is the combinatorial layer, scale is the magnitude layer

## Fixture files

- `fixtures/ps_pass_four_shapes.json` — one witness per shape class (3 steps each) from root (2,1)
- `fixtures/ps_pass_invariants.json` — 5 paths: 5-step UNIFORM_B/C/A + two MIXED with different orderings
