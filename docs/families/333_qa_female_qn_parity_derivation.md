# [333] QA Female QN Derivation: First-Fourth Parity

**Family**: `qa_female_qn_parity_derivation_cert_v1`  
**Sources**: Iverson (1991) *QA Vol I* p.27 Ch.2 Ex.8 + Iverson (1993) *QA Book 2* Ch.1 "Doubling"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Female QN = (2e, b, a, 2d) from male (b,e,d,a); d_f=2e+b=d, a_f=2e+2b=2d | PASS |
| C2 | gcd(e,d)=gcd(b,e)=1; e and d have opposite parities for all male QNs | PASS |
| C3 | e odd → b_f=2e is 2-par, a_f=2d is 4-par (d=b+e even, so 2d≡0 mod 4) | PASS |
| C4 | e even, b odd → b_f=2e is 4-par, a_f=2d is 2-par (d=b+e odd, so 2d≡2 mod 4) | PASS |
| C5 | First-fourth parity swap (2-par ↔ 4-par) holds for all 230 male QNs up to {1..23} | PASS |

## Core Structural Result

### Derivation Formula (C1)

From male (b, e, d, a) with d = b+e and a = b+2e (raw, A2, no mod reduction):

> "Double the two **intermediate** numbers (e, d) of the male QN  
> and place them at the **two ends**."

$$\text{female} = (b_f, e_f, d_f, a_f) = (2e,\ b,\ a,\ 2d)$$

Consistency check:
- $d_f = b_f + e_f = 2e + b = b+e = d$ ✓
- $a_f = b_f + 2e_f = 2e + 2b = 2(b+e) = 2d$ ✓

### Par-Type Parity Rule (C2–C4)

Since gcd(b, e) = 1 with b odd, e and d = b+e always have **opposite parities**:

| e parity | d parity | b_f = 2e | a_f = 2d | par swap |
|----------|----------|----------|----------|----------|
| odd | even | 2-par (≡ 2 mod 4) | 4-par (≡ 0 mod 4) | ✓ |
| even | odd | 4-par (≡ 0 mod 4) | 2-par (≡ 2 mod 4) | ✓ |

This is the exact claim from QA-2 Ch.1: *"If the first integer is 2-par, the fourth integer will be 4-par. If the first integer is 4-par, then the fourth will be 2-par."*

### Worked Examples

| Male (b,e,d,a) | Female (2e,b,a,2d) | b_f par | a_f par |
|----------------|---------------------|---------|---------|
| (1, 2, 3, 5) | (4, 1, 5, 6) | 4-par | 2-par |
| (1, 4, 5, 9) | (8, 1, 9, 10) | 4-par | 2-par |
| (3, 2, 5, 7) | (4, 3, 7, 10) | 4-par | 2-par |
| (1, 3, 4, 7) | (6, 1, 7, 8) | 2-par | 4-par |
| (3, 4, 7, 11) | (8, 3, 11, 14) | 4-par | 2-par |
| (5, 2, 7, 9) | (4, 5, 9, 14) | 4-par | 2-par |
| (1, 6, 7, 13) | (12, 1, 13, 14) | 4-par | 2-par |

## Observer Projection Note (Theorem NT)

"Male", "female", "2-par", "4-par" are observer-layer labels. The causal structure: integer gcd, divisibility by 4, and the identity gcd(e, b+e) = gcd(b, e). The parity swap is a pure modular arithmetic theorem, not a continuous function.

**Depends on**: [252] QA Par-Types; [282] Female Triangle Pythagorean Structure
