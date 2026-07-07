# [366] QA Pyth-1 Proofs: Bead Arithmetic Laws

**Family**: `qa_pyth1_proofs_bead_arithmetic_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter IX pp.94-100

> *(Statement 1)*: "Either d or e must be an even number."

> *(Statement 2)*: "The value of a is always an odd number."

> *(Statement 4)*: "The factor 3 will be represented in every set of bead numbers."

> *(Statement 9)*: "The area of every prime Pythagorean triangle is divisible by 6."

> *(Statement 12)*: "d is prime to b and e. And a is prime to b, d, and e."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Exactly one of {d, e} is even for every primitive pair (b, e) | PASS |
| C2 | a is always odd (both b and a are always odd; {d,e} always split even/odd) | PASS |
| C3 | Factor 3 divides at least one of {b, e, d, a} for every primitive pair | PASS |
| C4 | Area = CF/2 ≡ 0 (mod 6) for all prime Pythagorean pairs | PASS |
| C5 | All four bead numbers {b, e, d, a} are pairwise coprime (6 pairwise gcds all = 1) | PASS |

## Mathematical Details

### C1: Parity Lemma — Exactly One of {d, e} is Even

b is always odd (primitive pair axiom). d = b + e.

- If e is **even**: d = odd + even = **odd**. So {d,e} = {odd, even}: one even (e), one odd (d).
- If e is **odd**: d = odd + odd = **even**. So {d,e} = {even, odd}: one even (d), one odd (e).

In both cases exactly one of {d, e} is even. ✓

Verified for all 4047 prime pairs (b,e)≤100.

### C2: a is Always Odd

a = d + e. By C1, {d, e} always contains one even and one odd number. Therefore a = even + odd = **odd**. ✓

Summary: Among the four bead numbers, b and a are always odd; d and e always have opposite parities (one even, one odd).

Verified for all 4047 prime pairs (b,e)≤100.

### C3: Factor 3 in Every Bead Set

Every primitive pair has at least one of {b, e, d, a} divisible by 3. Proof by complete case analysis on (b mod 3, e mod 3):

| b mod 3 | e mod 3 | Who is 0 mod 3? | Why |
|---------|---------|-----------------|-----|
| 0 | any | b | given |
| any | 0 | e | given |
| 1 | 1 | a | a=b+2e≡1+2=3≡0 |
| 1 | 2 | d | d=b+e≡1+2=3≡0 |
| 2 | 1 | d | d=b+e≡2+1=3≡0 |
| 2 | 2 | a | a=b+2e≡2+4=6≡0 |

All 6 cases are exhaustive (b,e ∈ {0,1,2} mod 3) and each yields at least one divisible-by-3 element. ✓

Verified for all 4047 prime pairs (b,e)≤100. Distribution for (b,e)≤50: b divisible by 3 in 248 pairs; e in 255; d in 267; a in 259.

### C4: Area Divisible by 6

Area = CF/2 where C = 2de and F = ab.

**Divisibility by 2**: C = 2de is always even. One of {d,e} is even (C1), say e = 2k. Then C = 2d·2k = 4dk. Area = CF/2 = 4dk·F/2 = 2dk·F — always an even integer. ✓

**Divisibility by 3**: By C3, factor 3 divides some bead number:
- If 3|d or 3|e: then 3|C = 2de, so 3|area = CF/2.
- If 3|b or 3|a: then 3|F = ab, so 3|area = CF/2.

In all cases 3|area. Since 2|area and 3|area and gcd(2,3)=1: 6|area. ✓

Minimum area (b=1,e=1): C=4, F=3; area=6 (the 3,4,5 triangle). Verified for all 4047 prime pairs (b,e)≤100.

Note: L = abde/6 = area/6 is exactly Iverson's identity L — this confirms that L is always an integer.

### C5: Pairwise Coprimality of {b, e, d, a}

The primitive pair axiom states gcd(b,e)=1. All six pairwise gcds reduce to this:

| Pair | Algebraic reduction | Result |
|------|---------------------|--------|
| gcd(b, e) | axiom | 1 |
| gcd(b, d) | gcd(b, b+e) = gcd(b, e) | 1 |
| gcd(e, d) | gcd(e, b+e) = gcd(e, b) = gcd(b, e) | 1 |
| gcd(b, a) | gcd(b, b+2e) = gcd(b, 2e); b odd → gcd(b,2)=1; gcd(b,e)=1 → gcd(b,2e)=1 | 1 |
| gcd(e, a) | gcd(e, b+2e) = gcd(e, b) = gcd(b, e) | 1 |
| gcd(d, a) | gcd(b+e, b+2e) = gcd(b+e, e) = gcd(b, e) | 1 |

All four bead numbers are mutually coprime. This is stronger than Iverson's Statement 12, which only asserts that d is coprime to {b,e} and a is coprime to {b,d,e}.

Verified for all 4047 prime pairs (b,e)≤100.

## Theorem NT Note

The five claims in this cert are pure number-theoretic properties of the integer bead arithmetic (b,e,d,a) — parity, divisibility by 3, divisibility by 6, coprimality. They hold for all primitive pairs by algebraic necessity, not by any geometric construction. Chapter IX labels these as "proofs" in the context of Pythagorean triangles, but the claims are properties of the bead numbers themselves.

**Depends on**: [359] Nightside Energy (par classification); [360] Prime Triangle Structure (a always odd, G always 5-par); [361] Primeness Parity Shape (C always 4-par)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced all 5 claims over
4047 primitive pairs (b,e≤100): exactly one of {d,e} even; a always
odd; factor 3 always present in {b,e,d,a} (exact mod-3 distribution
248/255/267/259 for (b,e)≤50); area=CF/2 always divisible by 6; all 6
pairwise gcds of {b,e,d,a} equal 1. The validator
(`qa_pyth1_proofs_bead_arithmetic_cert_validate.py`) is genuinely
computed, no fixture-trusting gap.
