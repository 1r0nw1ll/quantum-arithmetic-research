# Rank-Quantized Modular Fibonacci Recurrence: A Combinatorial Aliasing Theorem

**Track A — Algebraic Foundations**
Date: 2026-03-27
Author: Will Dale

---

## Abstract

We prove a closed-form theorem characterizing the Orbit Follow Rate (OFR) of a sine wave under
rank-based modular quantization. For a sine wave sampled at rate SR with frequency f = k·SR/m
(exact resonance), the fraction of consecutive sample triples satisfying the mod-m Fibonacci
recurrence converges to an exact rational number Fib_hits(π_k, m) / m, where π_k is a permutation
derived by modular subsampling of the rank permutation of sin(2πj/m). The result is purely
combinatorial — independent of amplitude, signal dynamics, or temporal context. This subsumes and
corrects earlier empirical claims that OFR measures dynamical orbit coherence.

---

## Setup

### 1. Quantization

Let a signal x = (x_0, x_1, ..., x_{N-1}) be **equalized rank-quantized** at modulus m:

    state_j = ⌊m · rank(x_j) / N⌋

where rank(x_j) is the position of x_j in the sorted order of all N samples (0-based). By
construction, each of the m states is occupied by exactly ⌊N/m⌋ or ⌈N/m⌉ samples.

### 2. Orbit Follow Rate (OFR)

    OFR(x, m) = |{j : state_{j+2} ≡ state_j + state_{j+1} (mod m)}| / (N - 2)

This is the fraction of consecutive triples satisfying the mod-m Fibonacci recurrence.

---

## The Rank Permutation π₁

For modulus m, define the **fundamental rank permutation**:

    π₁[j] = rank of sin(2πj/m) among {sin(2πi/m) : i = 0, ..., m-1}

**Scope condition:** π₁ is well-defined (no ties) when m is odd, since sin(2πj/m) = sin(2πj'/m)
requires j + j' = m/2, which has no integer solution for odd m.

Computed values:

| m  | π₁ |
|----|-----|
| 3  | [1, 2, 0] |
| 5  | [2, 4, 3, 1, 0] |
| 7  | [3, 5, 6, 4, 2, 0, 1] |
| 9  | [4, 6, 8, 7, 5, 3, 1, 0, 2] |

**Structure of π₁:** The permutation follows an "arcsin CDF" pattern:
- j = 0: state ≈ m/2 (sin(0) = 0, midpoint of range)
- j = 0..m/4: ascending (ascending sine, ascending rank)
- j = m/4..3m/4: descending (sine peaks and descends, rank descends from m-1 to 0)
- j = 3m/4..m: re-ascending (sine returns to 0 from below)

For large odd m the pattern is: [⌊m/2⌋, ⌊m/2⌋+2, ..., m-1, m-2, ..., 0, 1, 3, ..., ⌊m/2⌋-1]

---

## Main Theorem

**Theorem (OFR for resonant sine, odd m).**

Let m be odd. Let x be a sine wave at frequency f = k·SR/m sampled at rate SR for N = n·m
samples (n integer). Then as n → ∞:

    OFR(x, m)  →  Fib_hits(π_k, m) / m

where:

    π_k[j]  =  π₁[k·j mod m]             (modular subsampling of π₁)

    Fib_hits(π, m)  =  |{j ∈ {0,...,m-1} :
                           π[(j+2) mod m] ≡ π[j] + π[(j+1) mod m]  (mod m)}|

**Proof sketch:**
For N = n·m with integer n, the sample x_j = sin(2πk·j/m) is periodic with period m. The
equalized rank within one period satisfies state_j = π₁[k·j mod m] = π_k[j] (rank permutation
under modular index substitution). The OFR counts Fibonacci-compatible triples within one period,
which is Fib_hits(π_k, m)/m, independent of n. □

---

## Subsampling Corollary

For gcd(k, m) = 1 (k and m coprime), the map j ↦ k·j mod m is a bijection on {0,...,m-1}.
Therefore π_k is a genuine permutation of {0,...,m-1}, and the Fibonacci hit count is a
well-defined integer in {0, 1, ..., m}.

For gcd(k, m) = g > 1, π_k[j] = π₁[k·j mod m] is periodic with period m/g and takes only
m/g distinct values. In this case the formula still holds analytically, but the empirical
convergence is disrupted by tie-breaking in the rank computation.

---

## Exact Fib_hits Table (odd m)

Computed values for m ∈ {3, 5, 7, 9}, all k ∈ {1, ..., m-1}:

### m = 5

| k | gcd(k,5) | Fib_hits | OFR(k,5) | Excess |
|---|----------|----------|----------|--------|
| 1 | 1 | **0** | **0/5 = 0.000** | **−0.200** |
| 2 | 1 | **2** | **2/5 = 0.400** | **+0.200** |
| 3 | 1 | **1** | **1/5 = 0.200** | **0** |
| 4 | 1 | **2** | **2/5 = 0.400** | **+0.200** |

### m = 9

| k | gcd(k,9) | Fib_hits | OFR(k,9) | Excess |
|---|----------|----------|----------|--------|
| 1 | 1 | **2** | **2/9 ≈ 0.222** | **+1/9** |
| 2 | 1 | **0** | **0/9 = 0.000** | **−1/9** |
| 3 | 3 | 0 | 0 | −1/9 |
| 4 | 1 | **2** | **2/9 ≈ 0.222** | **+1/9** |
| 5 | 1 | **1** | **1/9 ≈ 0.111** | **0** |
| 6 | 3 | 0 | 0 | −1/9 |
| 7 | 1 | **1** | **1/9 ≈ 0.111** | **0** |
| 8 | 1 | **1** | **1/9 ≈ 0.111** | **0** |

(Bold = gcd = 1, exact theorem applies)

### m = 7

| k | gcd(k,7) | Fib_hits | OFR(k,7) | Excess |
|---|----------|----------|----------|--------|
| 1 | 1 | 1 | 1/7 ≈ 0.143 | 0 |
| 2 | 1 | 2 | 2/7 ≈ 0.286 | **+1/7** |
| 3 | 1 | 1 | 1/7 ≈ 0.143 | 0 |
| 4 | 1 | 2 | 2/7 ≈ 0.286 | **+1/7** |
| 5 | 1 | 0 | 0/7 = 0.000 | **−1/7** |
| 6 | 1 | 2 | 2/7 ≈ 0.286 | **+1/7** |

---

## Key Special Cases

### k = m/2 (Nyquist) — even m only
When k = m/2: f = SR/2. The signal x_j = sin(πj) = 0 for all integer j. All states are
equal, and OFR = 0/m = 0. Verified for m ∈ {8, 12, 16, 24}.

### k = m − 1 (complement)
k = m−1 ≡ −1 (mod m). The permutation π_{m-1}[j] = π₁[−j mod m] = π₁[m−j]. This is the
time-reversed permutation. Empirically: Fib_hits ≥ 1 for all tested m; excess ≥ 0.

| m | Fib_hits(π_{m-1}) | excess |
|---|-------------------|--------|
| 5 | 2 | +0.200 |
| 7 | 2 | +0.143 |
| 9 | 1 | 0 |

---

## Triangle Wave Equivalence

**Corollary:** A triangle wave at frequency f = k·SR/m has the same OFR as a sine wave at the
same frequency. Both waveforms are unimodal per cycle (monotone ascending then descending), so
they induce the same rank ordering on samples within each period. The equalized quantized states
are identical, and hence Fib_hits is identical.

More generally, any **strictly unimodal** periodic waveform at f = k·SR/m yields the same
π_k and the same OFR.

---

## Sawtooth Immunity

A sawtooth wave at frequency f is monotone-ascending then discontinuously jumps. Its rank
ordering within each period is uniform: state_j ≈ j·m/N_period (a linear ramp). This creates
an almost-uniform distribution over states with no correlation structure. Empirically:
OFR(sawtooth, m) ≈ 1/m (chance) at all frequencies, with |excess| < 0.001.

---

## Phase Dependence

For a sine wave sin(2πft + φ) at exact resonant frequency, the OFR depends on phase φ
(verified empirically for m=9). This confirms the result is an aliasing/combinatorial effect:
the phase determines which sample-grid alignment applies, shifting the rank permutation.
In the limit N → ∞ with exact rational f·m/SR, the phase selects among a finite set of
distinct rank sequences.

---

## Relationship to QA Arithmetic

The mod-m Fibonacci recurrence (state_{j+2} ≡ state_j + state_{j+1} mod m) is the defining
recurrence of the QA orbit structure. Specifically:

- The QA cosmos orbit is the Fibonacci sequence mod m (m=9 or m=24).
- OFR measures what fraction of quantized triples are on this orbit.

**What OFR is:**
A combinatorial count of how often a discrete rank sequence falls on the QA Fibonacci orbit.
For exact resonant frequencies, this count is determined algebraically by the rank permutation
of the fundamental sine phase space.

**What OFR is not:**
A detector of dynamical coherence, musical structure, or orbit-following behavior in the
signal-source. The effect is entirely in the quantization and aliasing, not the signal.

---

## Placement

This result belongs in **Track A: Algebraic Foundations**, as a characterization of the
combinatorial structure of rank permutations under modular Fibonacci testing.

It should **not** be presented as Track D (empirical coherence detection) or as evidence
that audio signals "follow QA orbits." The correct framing is:

> **Theorem (Rank-Quantized Modular Fibonacci Recurrence):** For unimodal periodic signals at
> exact QA-resonant frequencies f = k·SR/m, the fraction of consecutive triples satisfying the
> mod-m Fibonacci recurrence is given exactly by a combinatorial count over the rank permutation
> of sin(2πj/m). The result is algebraically determined and independent of signal content.

---

## Closed Form for k = 1 (Fundamental)

Computed for all odd m from 3 to 49 (24 values). The sequence Fib_hits(π₁, m) is:

    m:  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
    h:  1  0  1  2  1  2  1  2  1  2  1  2  1  2  1  2  1  2  1  2  1  2  1  2

**Theorem (Fundamental Fib_hits):**

    Fib_hits(π₁, m) = { 0  if m = 5
                       { 1  if m ≡ 3 (mod 4)
                       { 2  if m ≡ 1 (mod 4) and m ≠ 5

Equivalently: for odd m ≥ 7, Fib_hits(π₁, m) = 1 + [m ≡ 1 mod 4].

**The m = 5 exception:** π₁ for m=5 is [2,4,3,1,0]. Checking all 5 consecutive triples
confirms zero Fibonacci-compatible triples. m=5 is the Golden Ratio modulus and the sole
odd m with Fib_hits = 0. Its rank permutation is maximally anti-Fibonacci.

**Consequence for OFR(k=1, m):**

    OFR(k=1, m) → { 0      if m = 5
                   { 1/m    if m ≡ 3 (mod 4)       [= chance, no excess]
                   { 2/m    if m ≡ 1 (mod 4), m≠5  [excess = 1/m above chance]

For the QA modulus m=9: m ≡ 1 (mod 4), so OFR(k=1, 9) = 2/9 exactly. ✓

---

## Open Questions

1. **Prove the m mod 4 formula:** Why does m ≡ 1 (mod 4) give exactly 2 hits and m ≡ 3
   give exactly 1? The answer likely lies in the symmetry of the arcsin CDF permutation
   under rotation by π (which maps sin(2πj/m) → sin(2π(j+m/2)/m) = −sin(2πj/m)).

2. **Symmetry class of π_k:** For which k does Fib_hits(π_k) = 0 (complete suppression)?
   For m=9: k=2 gives 0 hits. Is there an algebraic criterion beyond gcd(k,m)?

3. **Even m with tie-breaking specification:** Extend the theorem to even m by specifying a
   canonical tie-breaking rule that makes the empirical and theoretical values agree.

4. **Multitone signals:** What is OFR for a sum of two resonant sinusoids? Does the
   permutation structure generalize?

---

*This theorem closes the audio-coherence research thread. The mathematical content is real
and publishable; the dynamical-systems interpretation is not supported.*
