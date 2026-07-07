# [369] QA Pyth-2 Wave Quarter Points and Par Classification

**Family**: `qa_pyth2_wave_quarter_points_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XIII pp.43-56+

> *(p.43)*: "Any wavelength which is measured by a 4-par number will find each of these points falling at an integer."

> *(p.43)*: "If the wavelength is measured by a 2-par number, only the beginning, the middle, and the end will fall at an integer."

> *(p.43)*: "in the case where a wavelength is measured by an odd number, only the beginning and the ending will be at an integer. The midpoint will fall at a half-integer and the quarter points will fall at a quarter integer."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 4-par W (W≡0 mod 4): W/4, W/2, 3W/4 are all integers | PASS |
| C2 | 2-par W (W≡2 mod 4): W/2 is integer; W/4 is half-integer (not integer) | PASS |
| C3 | Odd W (W≡1 or 3 mod 4): W/2 is half-integer; W/4 is quarter-integer (denom=4) | PASS |
| C4 | 5-par (W≡1 mod 4): quarter=k+1/4; 3-par (W≡3 mod 4): quarter=k+3/4 — complementary mirror images | PASS |
| C5 | Two waves with wavelengths W₁,W₂ have integer coincidences exactly at multiples of lcm(W₁,W₂) | PASS |

## Mathematical Details

### Par Classification (Iverson's 5-level par system)

Iverson classifies integers by their level of "par":
- **4-par**: W ≡ 0 (mod 4) — divisible by 4
- **2-par**: W ≡ 2 (mod 4) — divisible by 2 but not 4
- **Odd par**: W odd; subdivided by mod-4 residue into 5-par (W≡1 mod 4) and 3-par (W≡3 mod 4)

### C1: 4-Par Wavelength — All Quarter Points are Integers

If W = 4k, then:
- W/4 = k (integer — first quarter)
- W/2 = 2k (integer — midpoint)
- 3W/4 = 3k (integer — third quarter)

All five wave points (start=0, quarter, mid, three-quarter, end=W) are integers.

Verified for 124 values W∈[4,496].

### C2: 2-Par Wavelength — Only Midpoint is Integer

If W = 4k+2, then:
- W/4 = k + 1/2 (half-integer — quarter point is NOT an integer)
- W/2 = 2k+1 (integer — midpoint IS an integer)
- 3W/4 = 3k + 3/2 (half-integer — three-quarter is NOT an integer)

Only start, midpoint, and end fall at integers.

Verified for 125 values W∈[2,498].

### C3: Odd Wavelength — Only Endpoints are Integers

If W is odd (W = 2m+1), then:
- W/2 = m + 1/2 (half-integer — midpoint is NOT an integer)
- W/4 = (2m+1)/4 — denominator is exactly 4 (quarter-integer)
- 3W/4 = 3(2m+1)/4 — denominator is exactly 4

Only start (0) and end (W) fall at integers.

Verified for 250 odd values W∈[1,499]: all have half-integer midpoints and quarter-integer quarter-points.

### C4: 3-Par vs 5-Par Quarter-Flip

Among odd wavelengths, the fractional part of W/4 depends on W mod 4:

| Par type | W mod 4 | W/4 = k + | Example |
|----------|---------|-----------|---------|
| 5-par | 1 | 1/4 | W=5: 5/4 = 1 + 1/4 |
| 3-par | 3 | 3/4 | W=3: 3/4 = 0 + 3/4 |

The fractional parts 1/4 and 3/4 are complementary: they sum to 1, making 5-par and 3-par quarter-points **mirror images** of each other within the unit interval.

Iverson notes these appear as Fig.16a wave diagrams where the quarter-peak of a 5-unit wave falls at position 5/4 while the 3-unit wave has its quarter at 3/4.

Verified for 100 odd values W∈[1,199].

### C5: Two-Wave Coincidences at LCM

The integer starting points of wave W₁ are {0, W₁, 2W₁, 3W₁, ...}; similarly for W₂.

The intersection of these two sets is exactly the set of multiples of lcm(W₁, W₂):

{n ≥ 0 : W₁|n and W₂|n} = {n ≥ 0 : lcm(W₁,W₂)|n}

This connects directly to the coincidence-period arithmetic of cert [368] (LCM of coprime cycles = product).

| W₁ | W₂ | lcm | First coincidence after 0 |
|----|-----|-----|--------------------------|
| 3 | 5 | 15 | 15 |
| 2 | 3 | 6 | 6 |
| 4 | 6 | 12 | 12 |

Verified for all 529 pairs (W₁,W₂)∈[2,24]².

## Theorem NT Note

"Sine wave," "high point," "low point," "amplitude," "wave crest," and the Fig.16 diagrams are observer projections. The QA discrete layer contains only:
- The wavelength W (a positive integer)
- Its par-class determined by W mod 4
- Integrality of W/4, W/2, 3W/4 determined by pure divisibility arithmetic
- Coincidence period = lcm(W₁, W₂)

**Depends on**: [368] Synchronous Harmonics LCM (coincidence periods via lcm); [318] Synchronous Harmonics Ceiling (the {3,5,7} 105-unit system and τ=60 extremal structure)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. The validator
(`qa_pyth2_wave_quarter_points_cert_validate.py`) genuinely recomputes
all 5 claims live (4-par/2-par/odd wavelength quarter-point
integrality, the 3-par/5-par mirror-image fractions, and the LCM
two-wave coincidence rule over 529 pairs) — all match the doc's worked
examples exactly, no fixture-trusting gap.
