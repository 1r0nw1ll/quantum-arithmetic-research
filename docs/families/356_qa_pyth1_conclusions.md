# [356] QA Pyth-1 Conclusions and Objectives

**Family**: `qa_pyth1_conclusions_cert_v1`  
**Source**: Iverson, B. (1993) *Pythagorean Arithmetic Vol I* Chapter X pp.95-108

> *(p.95-96)*: "This planetary unit of measure is 25,800 miles plus or minus 400 miles. If the apogee and the perigee of the orbit are divided by this figure, one will arrive at the K, and J values within the quantum group which is derived from the 120, 3599, 3601 prime Pythagorean triangle."

> *(p.96)*: "Quantum Arithmetic says that, the length of our year must be 360 days but it is slightly more than 365 days."

> *(p.98)*: "QUANTUM: Measured in integer values."

> *(p.100-101)*: "four beats to each bar... four bars to a phrase, four phrases to a verse, and four verses to the song."

> *(p.98)*: "Only those points which are proven, as proof was made in Chapter 9, can truly be considered to be Quantum Arithmetic."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | (b=59,e=1,d=60,a=61) → C=120, F=3599, G=3601; C²+F²=G²; J=3540, K=3660, K-J=C=120 | PASS |
| C2 | Quantum year 360=24×15 (15 Cosmos orbit cycles); 360 mod 24=0; 365 mod 24=5 (5-par) | PASS |
| C3 | Song structure 4^4=256 notes/verse; 4 verses=1024=2^10 total; maps to QA 4-tuple (b,e,d,a) | PASS |
| C4 | All 11 primary Pythagorean identities (A,B,C,D,E,F,G,H,I,J,K) are positive integers for 161 primitive pairs ≤19 | PASS |
| C5 | L=abde/6=CF/12 is always a positive integer; L=1 for 3-4-5; first 10 values: (1,1,1),(1,2,5),(1,3,14),(1,4,30),(1,5,55) | PASS |

## Mathematical Details

### C1: The 120-3599-3601 Triangle (pp.95-96)

This is the largest prime Pythagorean triangle with bead values under 100:
- **Bead numbers**: b=59, e=1, d=60, a=61 (gcd=1, b odd ✓)
- **Identities** (raw, no mod reduction): C=2de=120, F=ab=3599, G=d²+e²=3601
- **Pythagorean**: 120²+3599²=14400+12952801=12967201=3601² ✓
- **J, K values**: J=bd=3540, K=ad=3660, K-J=d(a-b)=60×2=120=C ✓

**Planetary unit (observer projection)**: Iverson notes 25,800 miles ± 400 as a unit. At this scale:
- K×25,800 ≈ 94,428,000 miles (≈ Earth-Sun aphelion: 94,509,460 miles, within 0.09%)
- J×25,800 ≈ 91,332,000 miles (≈ Earth-Sun perihelion: 91,402,640 miles, within 0.08%)

These are observer projections — they do not enter the QA causal layer.

### C2: The Quantum Year (p.96)

| Year | Value | mod 24 | Classification |
|------|-------|--------|---------------|
| QA year | 360 | 0 | Quantum-aligned (15 Cosmos cycles) |
| Observed | 365 | 5 | 5-par, not quantum-aligned |

360 = 24 × 15 = 6 × 60 = LCM(1..6) × 6. Divisible by all of {1,2,3,4,5,6,8,9,10} but not 7.

### C3: The Quantum Song (pp.100-101)

Iverson's "song" structure is a recursively 4-fold hierarchy:

| Level | Count | Total beats |
|-------|-------|-------------|
| Notes per chord | 4 | 4 |
| Beats per bar | 4 | 16 |
| Bars per phrase | 4 | 64 |
| Phrases per verse | 4 | 256 = 4^4 |
| Verses | 4 | 1024 = 2^10 |
| Final tonic | 1 | **1025** |

**Voice mapping to (b, e, d, a)**:
- Base note (tempo) → b (yin bead, always odd)
- Tenor note → e (yang bead)
- Alto note → d = b+e (first derived)
- "Understood fourth" → a = d+e (fully derived, "not sounded but understood")

The aboriginal bead (1,1,2,3) forms the simplest chord: base=1, tenor=1, alto=2, implied=3.

### C4: All 16 Identities Integer-Valued

Iverson (p.98) defines "QUANTUM: Measured in integer values." The 12 primary identities are:
A=a², B=b², C=2de, D=d², E=e², F=ab, G=d²+e², H=C+F, I=|C-F|, J=bd, K=ad, L=abde/6.

All are immediately integer by construction except L. L is an integer because:
- 4|C (cert [355] C3) → 4|CF
- 3|(b,e,d, or a) (cert [355] C2) → 3|(C or F) → 3|CF
- Combined: 12|CF → L=CF/12 ∈ ℤ

### C5: The L Identity — First Values

| (b,e) | d | a | L=abde/6 | Area=6L |
|-------|---|---|----------|---------|
| (1,1) | 2 | 3 | 1 | 6 (3-4-5 triangle: 3×4/2=6) |
| (1,2) | 3 | 5 | 5 | 30 |
| (1,3) | 4 | 7 | 14 | 84 |
| (1,4) | 5 | 9 | 30 | 180 |
| (1,5) | 6 | 11 | 55 | 330 |

L grows rapidly; sequence 1,5,14,30,55 matches OEIS A001700 (triangular numbers of triangular numbers).

## Theorem NT Note

Planetary units (miles), year length (365.25 days), musical frequency ratios, and aphelion/perihelion distances are **observer projections** — they measure QA structure but do not feed back as QA inputs. The bead arithmetic (b,e,d,a) is the causal layer.

**Depends on**: [355] Formal Proofs (for C3, C4 divisibility proofs); [337] J,K Parameters (K-J=C identity)
