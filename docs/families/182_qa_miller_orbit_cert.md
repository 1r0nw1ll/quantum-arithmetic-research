# Family [182] QA_MILLER_ORBIT_CERT.v1

## One-line summary

QA mod-9 orbit classification applied to Miller indices of crystal reflections yields four results: cosmos dominance in interplanar spacing, satellite quadrance restricted to QR(9), singularity quadrance as perfect squares, and satellite green quadrance = 3× cosmos.

## Mathematical content

### Miller indices as QA input

Miller indices (h, k, l) define crystal planes. The quadrance Q_M = h² + k² + l² encodes interplanar geometry. QA maps each reflection to an orbit via (b, e) = (h mod 9, k mod 9) → orbit classification.

### Four certified results

| # | Result | Evidence |
|---|--------|----------|
| 1 | Cosmos d_spacing > satellite d_spacing | 21/21 minerals tested |
| 2 | Satellite Q_M mod 9 ∈ QR(9) = {0, 1, 4, 7} | All satellite reflections |
| 3 | Singularity Q_M = perfect squares | All singularity reflections |
| 4 | Satellite green quadrance = 3× cosmos green | Chromogeometry ratio |

### Dataset

13,055 reflections across 4 crystal systems (cubic, tetragonal, orthorhombic, hexagonal) from standard crystallographic databases.

### Chromogeometry connection

Green quadrance C = 2de governs the satellite-to-cosmos ratio. The factor of 3 arises from |satellite| / |cosmos| period ratio = 8/24 = 1/3, inverted in the quadrance domain.

## Checks

| ID | Description |
|----|-------------|
| MO_1 | schema_version == 'QA_MILLER_ORBIT_CERT.v1' |
| MO_ORDER | cosmos d > satellite d for all 21 minerals |
| MO_QR | satellite Q_M mod 9 ∈ {0, 1, 4, 7} |
| MO_SQUARE | singularity Q_M is a perfect square |
| MO_CHROMO | satellite green = 3× cosmos green |
| MO_W | ≥3 witnesses (distinct minerals) |
| MO_F | ≥1 falsifier (non-QR satellite rejected) |

## Source grounding

- **Ben Iverson**: QA orbit classification and mod-9 structure
- **Wildberger**: chromogeometry — green/red/blue quadrance framework
- **Standard crystallographic databases**: Miller index reflections

## Connection to other families

- **[160] Bragg RT**: Bragg condition as rational trigonometry in QA framework
- **[181] Satellite Product Sum**: satellite orbit algebraic identity used in Q_M analysis

## Fixture files

- `fixtures/mo_pass_mineral_orbits.json` — 21 minerals with orbit-classified reflections
- `fixtures/mo_fail_bad_qr.json` — falsifier with satellite Q_M outside QR(9)
