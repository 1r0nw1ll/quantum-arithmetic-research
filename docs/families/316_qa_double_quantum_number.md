# [316] QA Double Quantum Number for Diadic Fractions

**Family**: `qa_double_quantum_number_cert_v1`  
**Depends on**: [310] Rational Surveying (BEDA squaring map), [315] Rhind 2/n Unit Fraction

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | General 3-term formula 2/n=1/(e₁k)+1/(e₁n)+1/(d₁n) with k=d₁n/(C₁−a₁) holds exactly as Fraction for all male QN matches and the double-female case (2/71) | PASS |
| C2 | Exactly 7 Rhind 3-term Cosmos entries match the male Double QN formula (C₁=a₁+n, k=d₁); 2/97 has two valid male chains: m=56 (1/56+1/679+1/776) and m=60 (1/60+1/291+1/1940) | PASS |
| C3 | Female transformation (b,e,d,a)→(2e,b,a,2d) produces a valid BEDA tuple; double-female (2,8,10,18) from male (1,4,5,9) gives k=5 and 2/71=1/40+1/568+1/710 (Rhind match) | PASS |
| C4 | All 8 four-term Rhind entries n∈{29,43,61,73,79,83,89,101} satisfy 2/n=1/p+Σ1/(cᵢn) with inner (2p−n)/p exact; n=29 BEDA closure: BEDA(4,2,6,8), p=C₁=24, inner {e,d,a}={2,6,8}, n=2p−2a−p/a=29 | PASS |
| C5 | Theorem NT: scribal notation, choice of p, hieroglyphic form are observer projections; QN chain, integer k, exact Fraction sums are discrete QA claims; inner QN for 4-term n≠29 explicitly open per Iverson QA-1 | PASS |

## Core concept (Iverson's own words)

> *"In the case of Rhind Mathematical Papyrus and the diadic fractions, these numbers were extended by multiplying the two middle numbers of the first set, and using that for the third number of the second group."*  
> — Iverson, QA-1 p.2

> *"Much more research and study is in order to learn the methods used to derive the complete table."*  
> — Iverson, QA-1 p.55

**Diadic** = "doubled" fraction, any 2/n. The Double Quantum Number is Iverson's term for the 7-element chain formed by two linked BEDA 4-tuples.

## General 3-term formula

For any BEDA tuple (b₁, e₁, d₁, a₁) and target n:

```
C₁ = 2·e₁·d₁
k  = d₁·n / (C₁ − a₁)        [must be a positive integer]

2/n = 1/(e₁·k) + 1/(e₁·n) + 1/(d₁·n)
```

Different (b₁,e₁,d₁,a₁) choices give different k values and thus different decompositions of 2/n.

## Male QN family (b₁ odd, C₁ = a₁ + n)

When C₁ = a₁ + n, then k = d₁ and the first denominator is e₁·d₁ = C₁/2 = d₂.

The 7-element Double QN chain: **(b₁, e₁, d₁, a₁=b₂, e₂, d₂, n)** where e₂=(n−a₁)/2, d₂=e₁·d₁.

| n | Chain | QA decomp | Rhind |
|---|---|---|---|
| 17 | (1,3,4,7,5,12,17) | 1/12+1/51+1/68 | ✓ |
| 31 | (1,4,5,9,11,20,31) | 1/20+1/124+1/155 | ✓ |
| 37 | (5,3,8,11,13,24,37) | 1/24+1/111+1/296 | ✓ |
| 47 | (7,3,10,13,17,30,47) | 1/30+1/141+1/470 | ✓ |
| 59 | (5,4,9,13,23,36,59) | 1/36+1/236+1/531 | ✓ |
| 67 | (3,5,8,13,27,40,67) | 1/40+1/335+1/536 | ✓ |
| **97** | **(1,7,8,15,41,56,97)** | **1/56+1/679+1/776** | **✓** |

For 2/97: two valid male chains exist (Iverson explicitly notes both):
- m=56=7×8 → chain (1,7,8,15,41,56,97) → 1/56+1/679+1/776 **(Rhind choice)**
- m=60=3×20 → chain (17,3,20,23,37,60,97) → 1/60+1/291+1/1940

The Rhind scribes chose m=56 because it gives the **largest first fraction** (smallest first denominator).

## Female QN family

**Female transformation**: male (b,e,d,a) → female **(2e, b, a, 2d)**

Derivation (Iverson p.27): "Double the two intermediate numbers of the male QN and place them at the two ends."

| Male | → Female | Verified |
|---|---|---|
| (1,4,5,9) | (8,1,9,10) | ✓ (d'=9, a'=10) |
| (1,2,3,5) | (4,1,5,6) | ✓ |
| (1,1,2,3) | (2,1,3,4) | ✓ |
| (1,3,4,7) | (6,1,7,8) | ✓ |

**Double female** = 2 × male: (2b, 2e, 2d, 2a)

For 2/71: double-female of (1,4,5,9) is **(2,8,10,18)**:
- C₁ = 2×8×10 = 160 = 18 + 2×71 → C₁ = a₁ + 2n (**double case**)
- k = d₁·n/(C₁−a₁) = 10×71/142 = **5**
- 2/71 = 1/(8×5) + 1/(8×71) + 1/(10×71) = **1/40+1/568+1/710** (Rhind match ✓)

The standard male formula gives 1/42+1/426+1/497 (valid, but not Rhind). The double-female is what the Rhind scribes used.

## 4-term two-level structure

Every 4-term Rhind entry has the form:

```
2/n = 1/p + (2p−n)/(p·n)
     = 1/p + 1/(c₁·n) + 1/(c₂·n) + 1/(c₃·n)
```

where (2p−n)/p = 1/c₁ + 1/c₂ + 1/c₃ (inner 3-unit-fraction sum).

| n | p | inner fraction | {c₁,c₂,c₃} | BEDA origin |
|---|---|---|---|---|
| 29 | 24 | 19/24 | {2,6,8} | **BEDA(4,2,6,8)**, p=C₁=24, a=d+e=8, n=2p−2a−p/a |
| 43 | 42 | 41/42 | {2,3,7} | open |
| 61 | 40 | 19/40 | {4,8,10} | open |
| 73 | 60 | 47/60 | {3,4,5} | open |
| 79 | 60 | 41/60 | {3,4,10} | open |
| 83 | 60 | 37/60 | {4,5,6} | open |
| 89 | 60 | 31/60 | {4,6,10} | open |
| 101 | 101 | 1 | {2,3,6} | open |

### n=29: complete BEDA closure

BEDA(b=4, e=2, d=6, a=8):
- p = C₁ = 2·e·d = 24 (even Pythagorean leg)
- BEDA identity: a = d + e = 8 ✓
- a divides p: 24/8 = 3
- **n = 2p − 2a − p/a = 48 − 16 − 3 = 29** ✓
- Inner {c₁,c₂,c₃} = {e,d,a} = {2,6,8}
- 2/29 = 1/24 + 1/58 + 1/174 + 1/232 ✓

The n=29 case is the only Rhind 4-term entry where the inner decomposition comes directly from the same BEDA tuple that generates p. For all other 4-term entries, the inner QN structure is open per Iverson.

## Open questions (per Iverson QA-1, p.55)

1. What QN family generates the inner {c₁,c₂,c₃} for n=43,61,73,79,83,89,101?
2. Do 4-term cases use a "triple-QN" chain or a different construction entirely?
3. Is there a unified QN formula covering both 3-term and 4-term cases?
