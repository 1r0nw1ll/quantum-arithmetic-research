# Family [205] QA_GRID_CELL_RNS_CERT.v1

## One-line summary

The entorhinal grid cell system is a biological residue number system (RNS) structurally isomorphic to QA modular arithmetic: each grid module computes position mod its period (= QA state mod m), CRT decoding reconstructs position from residues (= QA multi-modulus join), and the ratio of QA's two canonical moduli 24/9 = 2.667 falls within 2% of Euler's number e — the proven optimal period ratio for neuron-economy (Wei et al. 2015).

## Background

**Grid cells** (Hafner & Moser, Nobel Prize 2014) in the medial entorhinal cortex fire at regular spatial intervals, forming hexagonal lattices. Different grid modules have different spatial periods (30 cm to 10+ m). Fiete et al. (2008) identified this multi-module system as a **Residue Number System** — a number representation where a value is encoded as its residues modulo several coprime (or near-coprime) moduli, with unique decoding via the Chinese Remainder Theorem.

## Mathematical content

### RNS: Grid cell phase code = QA modular state

Each grid module α with period λ_α computes: `χ_α(x) = x mod λ_α`

QA computes: `b_new = ((b + e - 1) % m) + 1`

Both represent state as a residue modulo a period. The grid cell code is a vector of residues across modules; QA uses moduli m=9 (theoretical) and m=24 (applied).

### CRT: Reconstruction from residues

Grid cells: position uniquely recoverable from `(x mod λ₁, ..., x mod λ_N)` via CRT when moduli are sufficiently coprime. Capacity = LCM(λ₁, ..., λ_N).

QA: `qa_pim/crt.py` implements `crt_join_coprime()` and `crt_join_general()` — exactly the decoding operation.

### RATIO: 24/9 ≈ e (within 2%)

Wei et al. (eLife 2015) proved that the optimal period ratio for a 1D neuron-economy principle is **r = e = 2.718**. The derivation: minimize total neurons N = d·r·log_r(R) over r; setting dN/dr = 0 gives ln(r) = 1, hence r = e.

QA's ratio: **24/9 = 8/3 = 2.667**. Deviation from e: **1.9%**.

For 2D: optimal ratio = √e = 1.649. Measured in rats: 1.42–1.64 (Barry 2007, Stensola 2012).

**Additional**: φ² = 2.618 is also within 2% of 24/9. QA's norm field Q(√5) contains φ, and Vago & Ujfalussy (2018) proved φ is the optimal ratio for two-module coding via Hurwitz's theorem.

### LCM: LCM(9,24) = 72 = Cosmos orbit

gcd(9,24) = 3, so the moduli are NOT coprime. LCM(9,24) = 9×24/3 = **72**. This is exactly the number of distinct (b,e) pairs in the 24-cycle Cosmos orbit. The grid cell literature shows non-coprime modules have capacity = LCM, not product.

### CARRY: Independent module updates

Grid cells: each module updates phase independently from velocity input. No inter-module communication ("carry") needed. QA: each state (b,e) updates within its modulus without cross-modulus communication. Both exploit the carry-free property of modular arithmetic for parallel computation.

### ABS: Grid codes for abstract concepts

Constantinescu et al. (Science 2016): fMRI shows 6-fold hexagonal symmetry in vmPFC and entorhinal cortex when subjects navigate a 2D "bird space" (neck length × leg length). The grid code is NOT specific to spatial navigation — it organizes any continuous 2D feature space. This supports QA's application to non-spatial domains (finance, EEG, audio, climate).

### TORUS: Toroidal state space

Grid modules fire on hexagonal lattices with periodic boundaries = 2D torus. QA pairs (b,e) ∈ (Z/mZ)² form a discrete 2D torus. φ(9) = 6 = hexagonal symmetry order.

### HEX27: Hexagonal encoding for m=9

Kymn et al. (Neural Computation 2025): hexagonal encoding with modulus m achieves 3m²−3m+1 states using 3m codebook vectors. For m=9: **217 states with 27 vectors**. The number 27 = 3×9 = the three enneads of the Hebrew/Greek letter-number system ([202]).

## Checks

| ID | Description |
|----|-------------|
| GCR_1 | schema_version == 'QA_GRID_CELL_RNS_CERT.v1' |
| GCR_RATIO | \|24/9 − e\| / e < 10% (actual: 1.9%) |
| GCR_LCM | LCM(9,24) = 72 = Cosmos orbit pairs |
| GCR_PHI6 | φ(9) = 6 = hexagonal symmetry |
| GCR_HEX | 3×9²−3×9+1 = 217; 3×9 = 27 |
| GCR_NUM | numerical checks pass |
| GCR_W | >= 5 witnesses |
| GCR_F | falsifier: wrong modulus (24/7 = 3.43, 26% deviation) |

## Source grounding

- **Fiete et al.** (2008): *J. Neuroscience* 28(27):6858. Foundational RNS identification.
- **Sreenivasan & Fiete** (2011): *Nature Neuroscience* 14:1330. Analog error-correcting code.
- **Vago & Ujfalussy** (2018): *PLOS Comp Bio*. Number theory, golden ratio optimality.
- **Wei et al.** (2015): *eLife*. Economy principle, r=e optimal, hexagonal lattice.
- **Constantinescu et al.** (2016): *Science* 352:1464. Grid codes for abstract concepts.
- **Kymn et al.** (2025): *Neural Computation* 37:1. HD-vector RNS, hexagonal encoding.

## Connection to other families

- **[202] Hebrew Mod-9 Identity**: 27 codebook vectors = three enneads
- **[192] Dual Extremality**: π(9)=24 grounds the 24/9 ratio
- **[199] Grokking Eigenvalue**: Neural networks discover QA-compatible Fourier circuits
- **[191] Bateson Learning Levels**: Orbit invariance = error detection
- **[130] Origin of 24**: 24 as applied modulus, now also ≈ 9e
- **[154] T-Operator Coherence**: QCI coherence as grid cell phase consistency

## Fixture files

- `fixtures/gcr_pass_core.json` — 9 claims with full witnesses and sources
- `fixtures/gcr_pass_numerical.json` — ratio, LCM, totient, hex encoding verification
- `fixtures/gcr_fail_wrong_ratio.json` — wrong modulus (24/7) exceeds 10% threshold
