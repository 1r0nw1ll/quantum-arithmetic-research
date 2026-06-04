# [318] QA Synchronous Harmonics Ceiling

**Family**: `qa_sh_5040_ceiling_cert_v1`  
**Depends on**: [147] QA Synchronous Harmonics (wavelet sync + par rules), [151] QA Par Number (double parity)

> *"5040 ... has more divisors than any other integer below it."*  
> — Iverson, QA-2 Ch.6 p.84

> *"In order to jump from one Quantum Number to the next, requires a change in 6 units."*  
> — Iverson, QA-2 Ch.6 p.88

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 5040=2⁴·3²·5·7 has τ(5040)=60 divisors — more than any positive integer below 5040 (max below: 48); divisible by every integer 1-10; 5039 prime; 5041=71² has 3 divisors | PASS |
| C2 | All 21 harmonic fractions {k/d × 5040} for d∈{2,3,4,5,6,7}, k∈{1,...,d−1} are positive integers, ranging from 720 to 4320 — the "harmonic points of a wavelength of 5040 units" (Iverson p.85) | PASS |
| C3 | Minimum quantum wavelength with exactly 5 distinct prime factors containing 2 and 3 is 2310=2·3·5·7·11; with 7 is 510510=2·3·5·7·11·13·17; both primorials of first k primes; both divisible by 6 | PASS |
| C4 | Integers 5041–5045 are each not divisible by 6 (necessary condition for a quantum wavelength); 5046=2·3·29² has only 3 distinct prime factors, below the 5-factor minimum | PASS |
| C5 | Theorem NT: continuous frequency spectra and spectral lines are observer projections; harmonic fractions, τ(5040)=60, minimum QW wavelengths are discrete integer claims; Samekh open per Iverson | PASS |

## The ceiling number 5040

5040 = 7! = 1×2×3×4×5×6×7 = **2⁴ × 3² × 5 × 7**

It has 60 divisors: (4+1)(2+1)(1+1)(1+1) = 5×3×2×2 = 60.

**The extremal property**: No positive integer smaller than 5040 has as many divisors. The previous record holder had only 48. The nearest higher competitor needs to be substantially larger.

**Divisibility by 1–10** (all exact):

| Divisor | 5040 ÷ d |
|---|---|
| 1 | 5040 |
| 2 | 2520 |
| 3 | 1680 |
| 4 | 1260 |
| 5 | 1008 |
| 6 | 840 |
| 7 | 720 |
| 8 | 630 |
| 9 | 560 |
| 10 | 504 |

**The cliff at 5041**: τ(5040)=60 → τ(5041)=3. The number 5041=71² has only three divisors: 1, 71, 5041. This is the sharpest possible drop — from the most divisors of any number below it to the minimum possible for a composite. 5039 is prime (2 divisors). 5040 is a local maximum in every sense.

## The harmonic fractions of 5040

The 21 fractions {k/d × 5040} for denominators d = 2, 3, 4, 5, 6, 7:

| d | Fractions × 5040 |
|---|---|
| 7 | 720, 1440, 2160, 2880, 3600, 4320 |
| 6 | 840, 1680, 2520, 3360, 4200 |
| 5 | 1008, 2016, 3024, 4032 |
| 4 | 1260, 2520, 3780 |
| 3 | 1680, 3360 |
| 2 | 2520 |

All 21 are positive integers. Range: 720 (=5040/7) to 4320 (=6·5040/7).

These are what Iverson calls the "harmonic points of a wavelength of 5040 units" — the frequencies of Music of the Spheres. He connects them to the "Cattle of the Sun" fractions in the problem ascribed to Archimedes, and to the Chinese *Song Celestial* (I-Ching fractions of the fundamental wavelength).

## Minimum quantum wavelengths

Every quantum wavelength must contain prime factors 2 AND 3 (divisible by 6). The minimum with exactly 5 distinct prime factors is the primorial of the first 5 primes:

**2310** = 2 × 3 × 5 × 7 × 11

The minimum with 7 distinct prime factors:

**510510** = 2 × 3 × 5 × 7 × 11 × 13 × 17

Iverson: *"With 7 factors it is 2×3×5×7×11×13×17 = a half million units, for the minimum wavelength. This is too big a project to perform."* (p.75)

## The 5040 neighbourhood gap

| n | Divisible by 6? | Distinct prime factors | Quantum eligible? |
|---|---|---|---|
| 5040 | ✓ | 4 ({2,3,5,7}) | No — only 4 distinct primes |
| 5041 | ✗ | 1 ({71}) | No |
| 5042 | ✗ | 2 ({2,2521}) | No |
| 5043 | ✗ | 2 ({3,41}) | No |
| 5044 | ✗ | 3 ({2,13,97}) | No |
| 5045 | ✗ | 2 ({5,1009}) | No |
| 5046 | ✓ | 3 ({2,3,29}) | No — only 3 distinct primes |

The minimum quantum wavelength above 5040 requires at least 5 distinct prime factors including 2 and 3. 5040+6=5046 fails (only 3 distinct primes). The actual next minimum-5-prime QW above 5040 is 5460=2²·3·5·7·13, then 5544=2³·3²·7·11, etc.

## Samekh — the open frontier

Iverson proposes that somewhere above 5040, there exists a number "Samekh" that *"divides into prime factors in more than one way"* — contradicting Euclid's unique factorization theorem. He bounds it: *"above 5040 for linear numbers, above 71 for root structure"* and *"somewhere below 700,000."*

Iverson does not claim to have found Samekh. This cert records the open question as stated in QA-2 p.83. **Samekh is not a cert claim here.**

The cert [318] certifies only the integer arithmetic claims that are verifiable from the primary source.
