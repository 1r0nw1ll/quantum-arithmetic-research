<!-- PRIMARY-SOURCE-EXEMPT: reason=primary source is Chase's locally-received Nucleus.html (2026-04-19); this doc is the mapping worksheet from that artifact to QA, not a derivation from published literature -->

# QA Nucleus — Mapping from Chase's DIAZAI / HSTM OMNIMANIFOLD v28.0

**Primary source**: `~/Downloads/Nucleus.html` (Chase, received 2026-04-19), 423 lines, Three.js + Plotly + GSAP.

**Companion port**: `tools/qa_viz/threejs/qa_nucleus.html`.

**Methodology**: "Map Best-Performing to QA" (`memory/feedback_map_best_to_qa.md`). Every continuous-float or thematic component in the source has integer or rational structure underneath — catalog it, convert floats to their rational forms, identify the QA object each stands in for, then port the clean pieces. Pieces with broken geometry or unmapped magic constants are flagged, not dropped.

---

## Component mapping table

| Source element | Original form | QA structure | Rational form |
|---|---|---|---|
| `digitalRoot(n)` (L149) | `(n % 9) \|\| 9` | A1 mod-9: `((n−1) % 9) + 1` (equivalent for n ≥ 1) | — |
| `v-boost` input (L100) | `0.729` | ratio on mod-9 path generator | `3⁶/10³ = 729/1000` |
| `gamma = 1/√(1−v²)` (L302) | Lorentz factor | path-length rescale on discrete orbit | integer rescale table per boost |
| `si_gap` (L176) | `1.111…` | **Keely / Volk lifted ratio** | `10/9` |
| `dia_gap` (L177) | `5.555…` | **3D collapse threshold** = 5 lifted units | `50/9 = 5·(10/9)` |
| `eta` (L178) | `0.982` | efficiency loss on recycled path | `491/500` — **magic; flag** |
| `this.energy` (L170) | float accumulator in "eV" | **integer path-step counter on 10:9 orbit** | `k ∈ ℤ≥0` |
| `this.recycled` (L171) | float reservoir | unredeemed reverse-orbit steps | `r ∈ ℤ≥0` |
| `res` (L306) | `(10/9)^((cycle%111)·10/111)` | **10 lifted steps per 111 clock ticks** | `(10/9)^(s·10/111)` |
| `0.007` growth const (L307) | scalar | 7/1000 — **magic; flag** | `7/1000` |
| `0.45` feedback (L308) | scalar | `9/20` — factor-5 leakage | `9/20 = 3²·5 / 2²·5` |
| `halts / (cycle+1)` (L371) | "Halting Omega" | **orbit return density Ω_QA** | halts/k |
| "5D KK CORE" (L6) | Kaluza-Klein 4+1 | **QA 4-tuple (b,e,d,a) + path-time k** = 5 integer indices | natively integer |
| `base = 27` (L92) | KK radius | `27 = 3³ = 3·9` — mod-9 tiled 3× in each direction | 27 |
| `base × base` grid (L251) | 27² lattice | `3⁶ = 729` nodes — matches `boost` numerator | 729 |
| `i, j, k` kernel (L253–255) | `(x%9)+1, (y%9)+1, ((x+y)%9)+1` | `(b, e, k_chase)` — **k_chase ≠ A2 `d`** (off by shift) | A2-fixed: `d = ((b+e−1)%9)+1` |
| color by `dr(i+j+k)` (L256–258) | digital-root hue map | after A2 fix: `dr(b+e+d) = dr(2d)` | — |
| "Metatron 13 nodes" (L211–244) | 13 spheres in hex ring + all-pairs lines | **geometry broken** (Metatron is Fruit-of-Life derived, not hex ring); intent maps to Fuller VE 12-around-1 | 12 + 1 |
| chiral "2:1 R-to-L" (L328) | gsap scale tween | torus winding numbers | (2, 1) |
| `phonon-freq` (L101) | 1–1000 Hz | phase-index rational on mod-9 | `f/100` integer-scaled |
| `branch-factor` (L102) | 0–10 | orbit branching per T-step | integer 0–10 |
| `ijk × kji` (L379–381) | palindromic product | **invariant: `dr(ijk·kji) = dr(i+j+k)²`** | pure integer |
| `phason-flip` hue 180° (L391–397) | HSL rotate | mod-9 complementation `b ↔ 10−b` | integer inversion |
| inner `for s = 0..10` (L305) | 10 iters | the **10** of the 10:9 lifted ratio | integer 10 |
| `energy.toFixed(18)` (L369) | 18-digit float | display as `num/den` | `p/q` |

---

## Detailed mappings

### 1. The 10:9 lifted ratio is the central QA object

`si_gap = 10/9` and `dia_gap = 50/9 = 5·(10/9)`. The collapse threshold is **5 lifted units** — factor 5 is the "3D smoking gun" (Vibes 2026-04-08, `memory/project_vibes_corrections_apr08.md`): without factor 5, QA is a 2.5D projection. Dia_gap being `5·si_gap` is therefore the **dimensionally consistent 3D collapse threshold** for a 10:9 lifted path.

The inner `for s = 0..10` loop is the 10-step walk per outer tick, matching the **10** in 10:9. The 111-tick period is `111 = 3·37` — three copies of prime 37. Ten lifted steps per 111 ticks means each full 10-cycle spans `111` clock units. Not yet mapped to a known QA invariant; flag.

### 2. 27×27 kernel — the k off-by-shift correction

Chase uses `k = ((x + y) % 9) + 1`. This yields values in {1..9} but does **not** equal the A2-derived `d = b + e`. Example:

- `x = 5, y = 5` → `b = 6, e = 6`, so `b + e = 12`, and A2-compliant `d = ((12−1)%9)+1 = 3`.
- Chase's `k = ((5+5)%9)+1 = 2`. Off by 1 from QA `d`.

The relationship: `k_chase = qa_mod(b + e − 1)` whereas A2 `d = qa_mod(b + e)`. Off-by-one shift, systematic. Easy fix. Port uses A2 `d`.

### 3. Halting Omega as orbit return density

`Ω = halts / (cycle + 1)` is well-defined as soon as you specify when a halt fires. Chase halts on `energy ≥ 5.555` (continuous). Port halts on **landing on Singularity class (b=9, e=9)** under the walk — this is a discrete, A1-compliant criterion.

Expected density for a sequential walk over the 27×27 grid: 9 singularity nodes out of 729 total → `Ω → 9/729 = 1/81 ≈ 0.01235`. Port displays this live and tests convergence.

### 4. Palindrome `ijk × kji` — dr identity

Claim: for any digits `i, j, k ∈ {1..9}` forming integers `N = 100i + 10j + k` and `M = rev(N) = 100k + 10j + i`, **`dr(N·M) = dr(i+j+k)²`**.

Proof: digital root is invariant under digit permutation, so `dr(N) = dr(rev(N)) = dr(i+j+k)`. Digital root is multiplicative mod 9, so `dr(N·M) = dr(N)·dr(M) mod 9 = dr(i+j+k)² mod 9`.

Verified numerically:
- `b=3, e=4, d=7`: `347 × 743 = 257,821`, `dr = 7`. `dr(3+4+7) = 5`, `5² = 25`, `dr(25) = 7` ✓
- `b=1, e=2, d=3`: `123 × 321 = 39,483`, `dr = 9`. `dr(1+2+3) = 6`, `6² = 36`, `dr(36) = 9` ✓

Chase's code computes `ijk × kji` but displays only "prime proximity" — the dr identity is the real invariant. Port shows both.

### 5. Metatron 13 → Fuller VE 12+1

Chase's "Metatron's Cube" at L211–244 is 13 spheres arranged as a center + hex ring + offset in z (the 13-vertex count is the only Metatron-accurate feature). Real Metatron's Cube is derived from Fruit of Life — 13 spheres in a specific 2D packing, not a 6+6+1 ring. The intent — 13 nodes around a center — maps cleanly to **Fuller's Vector Equilibrium (VE) / cuboctahedron**: 12 spheres tangent to 1 central sphere, all 12 equidistant. This is the `12 + 1 = 13` canonical closest-packing configuration.

The QA mapping of VE to mod-9 is non-trivial and connects to the Fuller voxelation initiative (`memory/project_strategic_pivot_202604.md`). **Deferred**: port does not include this piece; revisit when voxelation track is active.

### 6. Lorentz gamma as discrete path-length rescale

`γ = 1/√(1 − v²)` for `v = boost = 729/1000`:

- `v² = 531,441 / 1,000,000`
- `1 − v² = 468,559 / 1,000,000`
- `γ = √(1,000,000 / 468,559) ≈ 1.4611`

The rational `1 − v² = 468,559 / 10⁶` has numerator `468,559 = 7 · 66,937 = 7 · 61 · 1097`. No clean QA signature. The **form** (Lorentz) is aesthetic; the **numerical value** `γ ≈ 1.461` doesn't correspond to a known QA rescaling ratio. Port treats boost as a rational slider and logs γ as an integer-approximation telemetry, but does not feed γ as a continuous input into any orbit rule (Theorem NT compliance).

---

## What's ported to `qa_nucleus.html`

- **27×27 kernel** with A2-compliant `d = ((b+e−1)%9)+1` and full 4-tuple `(b, e, d, a = ((b+2e−1)%9)+1)`.
- **Integer path-step counter `k`** replacing the float energy accumulator.
- **Orbit return density Ω_QA = halts / k**, where halt = cursor landing on Singularity class.
- **Palindrome dr-invariant panel**: current (b,e,d), `bed × deb` product, `dr(product)` and `dr(b+e+d)²` side-by-side.
- **Color palette** from Chase's `Colors.blendCharge` (preserves his dr→hue aesthetic).
- **Orbit class filter** (Cosmos / Satellite / Singularity) — reused from `qa_torus.html`.
- **Rationals display**: `si_gap = 10/9`, `dia_gap = 50/9`, `boost = 729/1000 = 3⁶/10³` shown symbolically, not as decimals.

## Deferred (flagged, not ported)

- **Metatron / Fuller VE 12+1 geometry** — revisit with voxelation initiative.
- **5D KK path-time rendering as an animated extra dimension** — needs design pass.
- **Lorentz γ as a discrete rescaling table** — needs integer boost catalog.
- **Magic constants** `0.007` (L307) and `0.982` (L178) — not yet mapped; mark as unresolved.
- **111-tick period** (= 3·37) — role not yet mapped to known QA invariant.
- **Phonon cymatics / Hz slider** — needs phase-index encoding before porting.
