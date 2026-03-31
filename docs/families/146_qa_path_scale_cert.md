# Family [146] QA_PATH_SCALE_CERT.v1

## One-line summary

G=d²+e² grows exponentially along UNIFORM_B paths (ratio → 3+2√2 = silver ratio squared) and polynomially along UNIFORM_A and UNIFORM_C paths (ratio → 1).

## Mathematical content

### Scale profile

The **scale profile** of a generator path is the sequence of G values G₀, G₁, ..., Gₙ where Gₖ = dₖ²+eₖ² is the blue quadrance at step k.

### Two scale classes

**EXPONENTIAL** — G ratio Gₙ₊₁/Gₙ converges to a constant > 1.

Only UNIFORM_B (M_B-only) paths exhibit this. M_B's matrix [[2,1],[1,0]] has dominant eigenvalue 1+√2, so d grows by factor (1+√2) per step. Since G ~ d²(1 + (e/d)²) and e/d → 1/(1+√2), the G ratio converges to (1+√2)² = 3+2√2 ≈ 5.828427.

| Step | Direction (d,e) | G | Ratio |
|------|----------------|---|-------|
| 0 | (2,1) | 5 | — |
| 1 | (5,2) | 29 | 5.800 |
| 2 | (12,5) | 169 | 5.828 |
| 3 | (29,12) | 985 | 5.828 |
| 4 | (70,29) | 5741 | 5.828 |
| 5 | (169,70) | 33461 | 5.828 |

Convergence is rapid: by step 2, the ratio is within 0.001 of the limit.

**POLYNOMIAL** — G ratio → 1 (polynomial growth, not exponential).

UNIFORM_A: G = 2n²+6n+5 (from (n+2,n+1) directions). Quadratic growth.
UNIFORM_C: G = (d₀+2ne)²+e² ≈ 4n²e² for large n. Quadratic growth.

Contrast after 5 steps from (2,1):
- UNIFORM_B: G = 33461 (exponential)
- UNIFORM_A: G = 85 (polynomial)
- UNIFORM_C: G = 145 (polynomial)

### The silver ratio connection

3+2√2 = (1+√2)² is the square of the silver ratio. This same constant appears in:
- **[140] Conic Discriminant**: d/e = 1+√2 is the parabola boundary (irrational, unreachable)
- **[141] Pell Norm**: M_B generates Pell solutions; Pell continued fraction convergents approach 1+√2
- **[138] Plimpton 322**: (12,5) with d/e = 2.4 is close to 1+√2 = 2.414...

### Growth monotonicity

All three generators increase G from any primitive direction with d>e>0:
- M_A: G_new = 5d²−4de+e² > d²+e² = G when d > e (always true)
- M_B: G_new = 5d²+4de+e² > G (trivially, since 4de > 0)
- M_C: G_new = d²+4de+5e² > G (trivially, since 4de > 0)

Therefore all forward paths have G strictly increasing.

## Checks

| ID | Description |
|----|-------------|
| SC_1 | schema_version == 'QA_PATH_SCALE_CERT.v1' |
| SC_2 | all G values recomputed correctly as d²+e² |
| SC_GROWTH | G strictly increasing at every step |
| SC_RATIO | declared G ratios match computed within tolerance |
| SC_CONV_B | EXPONENTIAL path final ratio within 0.01 of 3+2√2 |
| SC_W | ≥3 paths |
| SC_F | root (2,1) present |

## Source grounding

- **Pell equation theory**: (1+√2)² = 3+2√2 as the fundamental growth rate of Pell denominators
- **Silver ratio**: 1+√2 = continued fraction [2;2,2,2,...]; convergents are M_B chain d/e ratios
- **Cert [141]** QA_PELL_NORM_CERT.v1: M_B = Pell-flip; chain generates Pell solution sequence
- **Cert [145]** QA_PATH_SHAPE_CERT.v1: companion cert defining the four shape classes

## Connection to other families

- **[145] Path Shape**: shape = combinatorics of generator sequence; scale = magnitude evolution
- **[141] Pell Norm**: M_B chain is the Pell sequence; silver ratio is the Pell eigenvalue
- **[140] Conic Discriminant**: 1+√2 = parabola boundary = scale growth eigenvalue
- **[135] Pythagorean Tree**: M_A/M_B/M_C generators whose scale behavior is certified here
- **[138] Plimpton 322**: Plimpton Row 1 (12,5) sits near the silver ratio boundary; its G=169=13² is step 2 of the Pell chain

## Fixture files

- `fixtures/sc_pass_scale_classes.json` — three 5-step paths (one per generator type) showing EXPONENTIAL vs POLYNOMIAL
- `fixtures/sc_pass_pell_convergence.json` — 8-step Pell chain demonstrating rapid convergence to 3+2√2
