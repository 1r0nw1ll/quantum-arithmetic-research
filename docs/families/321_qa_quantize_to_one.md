# [321] QA Quantize-to-ONE

**Family**: `qa_quantize_to_one_cert_v1`  
**Depends on**: [320] QA Quantize Algorithm (midpoint identity b+a=2d, DO=d²)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Cosmos ratio range: b/a ∈ (0,1) for all 72 Cosmos pairs — b < a=b+2e since e≥1; Iverson: "perigee divided by apogee will always be a decimal less than 1" | PASS |
| C2 | Satellite canonical ratio 1/3: all 8 Satellite pairs (b=e) give b/a = b/(3b) = 1/3 exactly; no Cosmos pair achieves ratio 1/3 (b/a=1/3 implies b=e = Satellite condition) | PASS |
| C3 | Within-d-class monotonicity: for fixed d=b+e, the ratio b/(2d−b) is strictly increasing in b; all pairs in the same d-class have distinct ratios | PASS |
| C4 | Seven-prime factorization bound: ω(b·e·d·a) ≤ 5 for all 72 Cosmos pairs — confirms Iverson's "no more than seven successful divisions" (p.40) with margin; empirical max = 5 at e.g. (5,8): 5·8·13·21=10920=2³·3·5·7·13 | PASS |
| C5 | Theorem NT round-trip: given ratio r=b/a and Myriad scale d, reconstruct b via 2dr/(1+r) — exact integer for all 72 Cosmos pairs (uses midpoint identity a+b=2d from cert [320]) | PASS |

## The "Quantize to ONE" extension

"Quantize to ONE" is the application of the Quantize algorithm where **KO = THE ONE** — the Myriad reference unit. Instead of two absolute measurements, the input is a ratio JO/KO = b/a ∈ (0,1).

Iverson: *"In order to QUANTIZE TO ONE, the ratio of the perigee divided by the apogee would be entered in Line 20. Then a number representing THE ONE would be entered in line 30."* (QA-3 Ch.3, p.31)

This is the natural complement to cert [320]: where [320] recovers (b,e) from absolute measurements (DO=d², JO=d·b), cert [321] shows the ratio structure that makes "Quantize to ONE" work.

## Why the Satellite ratio is exactly 1/3

For any Satellite pair (b=e):

```
a = b + 2e = b + 2b = 3b
b/a = b/(3b) = 1/3
```

This is the unique fixed ratio of the Satellite orbit. The ratio 1/3 is **excluded** from the Cosmos: if b/a = 1/3, then 3b = a = b+2e, so 2b = 2e, so b = e — the Satellite condition. The Cosmos and Satellite orbits are cleanly separated by this ratio.

## Within-d-class monotonicity

For fixed d = b+e, the Cosmos pairs with that d-value have b ranging from 1 to d−1 (with e=d−b ≥ 1, b ≠ d−b). The ratio:

```
b/a = b/(b+2e) = b/(b+2(d-b)) = b/(2d-b)
```

has derivative with respect to b: d/db [b/(2d-b)] = 2d/(2d-b)² > 0. So the ratio is strictly increasing in b within each d-class. No two Cosmos pairs in the same d-class share a ratio.

## The 7-prime factorization bound

Iverson states: *"There should be no more than seven successful divisions. Each prime number obtained (up to 7 of them) will represent a wavelet in the upper Myriad of energy."* (p.40)

For mod-9 Cosmos, the empirical maximum is ω(b·e·d·a) = **5**, achieved at several pairs including (5,8):

```
5 · 8 · 13 · 21 = 10920 = 2³ · 3 · 5 · 7 · 13    [5 distinct primes]
```

Ben's claim of ≤7 is a conservative bound — the actual maximum for mod-9 is 5. This connects the arithmetic of the BEDA 4-tuple to the Seven Myriads model: each distinct prime factor represents a wavelet component in the energy hierarchy.

## Ratio reconstruction formula

Given the Myriad scale d and the ratio r = Fraction(b, a), recover (b, e) via:

```
b_rec = 2·d·r / (1 + r)
e_rec = d - b_rec
```

**Proof**: 2dr/(1+r) = 2d·(b/a)/((a+b)/a) = 2d·b/(a+b) = 2d·b/(2d) = b. The midpoint identity a+b = 2d (cert [320] C1) guarantees the denominator cancels exactly, giving integer b for all Cosmos pairs. ∎

## The Seven Myriads (QA-3 Ch.3 context)

Iverson describes 7 nested energy scales (Myriads), each spanning ~7 octaves of 12 frequencies:

| Myriad | Name | Frequency range |
|--------|------|-----------------|
| 1 | Creative | Highest (shortest wavelength) |
| 2 | Psi / Metaphysical | Thought energy |
| 3 | Light | IR + Visible + UV |
| 4 | Chemism | Periodic Table of Elements |
| 5 | Ultra Sound | Above audible |
| 6 | Sound | Music (audible) |
| 7 | Mentalism | Brain waves, 0.1–60 Hz |

Energy "cascades" from higher to lower Myriads: groups of 5–7 wavelets bond to form a single composite wave in the next lower Myriad. The ≤7 prime factor bound on b·e·d·a is the QA arithmetic expression of this 7-component bonding limit.
