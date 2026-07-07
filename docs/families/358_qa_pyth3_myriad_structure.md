# [358] QA Pyth-3 Myriad and Octave Structure

**Family**: `qa_pyth3_myriad_structure_cert_v1`  
**Source**: Iverson & Elkins (2006) *Pythagorean Arithmetic Vol III* Chapters 1 and 4

> *(Ch.1 p.1)*: "This Myriad of Sound is divided into seven octaves in which each octave is double the frequency of the one below it... the number values remain exactly the same, (1 to 10,000), in this new unit-of-measure."

> *(Ch.1 p.2)*: "The four primary notes of a bugle are C, F, A, and C."

> *(Ch.4 p.13-14)*: "Each Myriad consists of seven octaves. Each octave contains a maximum of 10,000 different quantum frequencies... 7 Myriads, each with 7 octaves, equals 49 sub-levels."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Myriad=1..10000=2^4×5^4=10^4; 10000 mod 24=16 (Cosmos class 16-par); digital root=1 | PASS |
| C2 | 7 octaves per Myriad; octave=factor 2; 2^7=128 ratio; 7 is first prime not dividing QA year 360=2^3×3^2×5 | PASS |
| C3 | 7 Myriads × 7 octaves = 49=7^2 total levels; 49 mod 24=1 (Singularity orbit class); 49 mod 9=4 | PASS |
| C4 | Bugle C,F,A,C: just-intonation ratios 3:4:5:6; unique primitive QA chain (b,d,a,a+e) with a+e=2b (octave); 3^2+4^2=5^2 (3-4-5 triple embedded) | PASS |
| C5 | Doubling (b,e)→(2b,2e): all 11 quadratic identities scale by 4; quartic L scales by 16; (6,2,8,10)=2×(3,1,4,5) has gcd=2 | PASS |

## Mathematical Details

### C1: Myriad as Integer Range

The Greek *myriad* (μυριάς) = 10,000 = 10^4 = 2^4 × 5^4. Iverson uses it as the count of integer quantum frequencies within each octave.

| Property | Value |
|----------|-------|
| 10000 = | 2^4 × 5^4 |
| 10000 mod 24 | 16 (Cosmos orbit class 16-par) |
| Digital root (mod 9) | 1 (since 1+0+0+0+0=1) |
| 10000 mod 4 | 0 (4-par) |
| Count of integers in [1..10000] | 10,000 |

### C2: The 7-Octave Structure

An octave doubles frequency. Seven octaves give 2^7 = 128-fold ratio from bottom to top.

| Property | Value |
|----------|-------|
| Octave ratio | 2:1 |
| 7 doublings | 2^7 = 128 |
| 128 < 10,000 | Myriad range covers full 128-fold span |
| Primes dividing 360 | {2, 3, 5} only |
| First prime NOT dividing 360 | **7** (since 360 = 2^3 × 3^2 × 5) |

The 7-fold structure of Myriads and octaves has a number-theoretic root: 7 is precisely the smallest prime outside 360's factorization. The QA year (360 = 24 × 15) organizes Cosmos orbit cycles; 7 operates at a different level.

### C3: The 49 = 7^2 Hierarchy

| Total levels | 49 = 7 × 7 = 7^2 |
|---|---|
| 49 mod 24 | **1** — Singularity orbit class (the fixed point of Cosmos dynamics) |
| 49 mod 9 | **4** (since 7^2 = 49 = 5×9 + 4) |
| 7 celestial rotation types (Ch.4) | axial, diurnal, monthly, annual, precessional, galactic, universal |

The 49-level hierarchy lands on orbit class 1 (mod 24) — the Singularity, the fixed point. The maximum possible hierarchy collapses to the simplest QA state.

### C4: The Bugle Chord and the Unique QA Octave Chain

The four primary bugle notes in just intonation (from C = 1):

| Note | Ratio | Integer (×3) |
|------|-------|--------------|
| C (lower) | 1 | **3** |
| F | 4/3 | **4** |
| A | 5/3 | **5** |
| C (upper) | 2 | **6** |

The integer ratios 3:4:5:6 form the arithmetic sequence (b, b+e, b+2e, b+3e) for **b=3, e=1**. This is the QA bead chain:

- b = 3 (lower C)
- d = b+e = 4 (F)
- a = d+e = 5 (A)
- a+e = 6 = 2×3 = 2×b (upper C = octave of lower C)

**Uniqueness proof**: The octave condition requires a+e = 2b, i.e., b+3e = 2b → b = 3e. Primitivity requires gcd(b,e) = gcd(3e,e) = e = 1. Therefore e = 1 and b = 3 is the unique primitive solution.

**Pythagorean structure of the ratios**:

- 3^2 + 4^2 = 9 + 16 = 25 = 5^2 ✓ (the 3-4-5 right triangle)
- Ratios 3:4:5:6 embed the fundamental Pythagorean triple as their first three terms

**QA bead triple for (b=3, e=1, d=4, a=5)**:
- C = 2de = 8, F = ab = 15, G = d^2+e^2 = 17
- 8^2 + 15^2 = 64 + 225 = 289 = 17^2 ✓ (8-15-17 right triangle)

The bugle frequency ratios and the bead-triple Pythagorean sides are distinct structures sharing the same generative tuple.

### C5: Bead Doubling and Identity Scaling

Doubling all beads (b,e,d,a) → (2b,2e,2d,2a) scales every Pythagorean identity by a power of 4:

| Identity type | Degree (in beads) | Scale factor |
|---------------|-------------------|--------------|
| A, B, D, E (squares) | 2 | 4 |
| C=2de, F=ab, G=d²+e², H,I (sums) | 2 | 4 |
| J=bd, K=ad | 2 | 4 |
| L = abde/6 | **4** | **16** |

Example: (b=3,e=1,d=4,a=5) → (b=6,e=2,d=8,a=10):

| Identity | Original | Doubled | Ratio |
|---------|----------|---------|-------|
| C | 8 | 32 | 4 |
| F | 15 | 60 | 4 |
| G | 17 | 68 | 4 |
| H | 23 | 92 | 4 |
| I | 7 | 28 | 4 |
| J | 12 | 48 | 4 |
| K | 20 | 80 | 4 |
| **L** | **10** | **160** | **16** |

The doubled tuple has gcd(6,2)=2 — it is NOT primitive. Every primitive tuple's "octave doubling" leaves the primitive layer.

## Theorem NT Note

"Sound," "octave," "Myriad," "light frequency," "celestial rotation" are observer projection labels for integer arithmetic structure (Cosmos orbit class, prime factorization, bead chain arithmetic). Causal layer is (b,e,d,a) integer arithmetic. No continuous frequency values enter the QA causal layer.

**Depends on**: [356] Pyth-1 Conclusions (quantum year 360=24×15); [355] Formal Proofs (divisibility structure); [357] Twenty Identities (bead-chain identities); [318] Synchronous Harmonics Ceiling (harmonic integer structure)  
**Uniqueness claim**: the bugle 4-note just-intonation chain is the unique primitive arithmetic QA octave sequence — a structural singularity in the space of all such chains

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced all 5 claims: 10000
mod 24=16, digital root=1; 360=2³×3²×5 has prime factors {2,3,5}, first
prime not dividing it is 7; 49 mod 24=1, 49 mod 9=4; the bugle chain
(3,1,4,5) with a+e=6=2b, 3²+4²=5², bead triple C=8/F=15/G=17 satisfying
8²+15²=17²; and the doubling-scaling law (all quadratic identities ×4,
`L` ×16) verified exactly for (3,1,4,5)→(6,2,8,10). The validator
(`qa_pyth3_myriad_structure_cert_validate.py`) is genuinely computed, no
fixture-trusting gap.
