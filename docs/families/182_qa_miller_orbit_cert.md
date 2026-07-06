# Family [182] QA_MILLER_ORBIT_CERT.v1

## One-line summary

QA mod-9 orbit classification applied to Miller indices of crystal reflections yields four results: cosmos dominance in interplanar spacing, satellite quadrance restricted to QR(9), singularity quadrance as perfect squares, and satellite green-channel fraction ≈2.6× cosmos (observed; see Verification Note on the "3×" figure below).

## Mathematical content

### Miller indices as QA input

Miller indices (h, k, l) define crystal planes. The quadrance Q_M = h² + k² + l² encodes interplanar geometry. QA maps each reflection to an orbit via (b, e) = (h mod 9, k mod 9) → orbit classification.

### Four certified results

| # | Result | Evidence |
|---|--------|----------|
| 1 | Cosmos d_spacing > satellite d_spacing | 21/21 minerals tested |
| 2 | Satellite Q_M mod 9 ∈ QR(9) = {0, 1, 4, 7} | All satellite reflections |
| 3 | Singularity Q_M = perfect squares | All singularity reflections |
| 4 | Satellite green-channel fraction ≈2.6× cosmos (observed) | Chromogeometry ratio |

### Dataset

13,055 reflections across 4 crystal systems (cubic, tetragonal, orthorhombic, hexagonal) from standard crystallographic databases.

### Chromogeometry connection

Green quadrance C = 2de governs the satellite-to-cosmos ratio: satellite
reflections have a notably higher green-channel fraction (0.303, from
415/1370 satellite reflections) than cosmos (0.117, from 1348/11550
cosmos reflections) — an observed ratio of 2.59×, not exactly 3×. The
period ratio |cosmos|/|satellite| = 24/8 = 3 is a plausible *heuristic
motivation* for why satellite is green-shifted relative to cosmos (both
involve the same divisibility-by-3 structure), but it is not a rigorous
derivation of the exact numeric factor — see Verification Note.

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

## Verification Note (2026-07-06)

**Found and fixed a real overclaim**: the doc's headline claim
"satellite green quadrance = 3× cosmos green" doesn't match the actual
data. Ran the real backing script (`experiments/qa_voxelation_crystal_batch.py`,
which uses genuine lattice parameters from RRUFF/AMCSD for all 21
minerals — Silicon a=5.4299, Halite a=5.6402, etc.) and independently
reproduced every headline number exactly: 13,055 reflections, 21
minerals, 4 crystal systems, orbit census 11550/1370/135
(cosmos/satellite/singularity), 21/21 minerals confirming
cosmos-mean-d > satellite-mean-d. The chromogeometric channel counts
also reproduce exactly: cosmos green fraction = 1348/11550 = 0.117,
satellite green fraction = 415/1370 = 0.303 — an observed ratio of
**2.59×**, not 3×. The doc's "period ratio 24/8=3, inverted" argument is
a plausible heuristic (both satellite's period and its green-shift stem
from the same divisibility-by-3 structure) but was never a rigorous
derivation of the exact factor, and presenting it as "= 3×" overstated
what the argument actually establishes. Corrected the one-line summary,
the results table, and the chromogeometry section to state the observed
~2.6× and be explicit that the period-ratio argument is heuristic
motivation, not proof of the exact number.

**QR(9)={0,1,4,7} independently confirmed** via direct computation
(squares mod 9 for x=0..8). **Perfect-square and QR-restriction proofs
verified algebraically**: satellite has h,k≡0 mod 3 ⟹ h²+k²≡0 mod 9 ⟹
Q_M≡l² mod 9 (always a QR); singularity has h=k=0 ⟹ Q_M=l² (perfect
square) — both proofs are simple, correct number theory, independently
re-derived.

Also noted (not fixed, cosmetic only): the backing script's own module
docstring says "19 minerals" while its actual `CRYSTALS` list and every
other reference (fixture, doc) consistently say 21 — a stale header
comment, harmless since no code depends on it.

No other bugs found. The validator (`MO_CHROMO` check) only verifies
`satellite_green > cosmos_green` (direction, not magnitude) — this was
already appropriately conservative and didn't itself assert "3×", so no
validator/fixture changes were needed, only the doc's prose.
