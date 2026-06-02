# [305] QA Reactive Power Versor Coupling

**Family**: `qa_reactive_power_versor_coupling_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [296] SL(2,Z) Versor Isomorphism, [299] Cayley-Hamilton Fibonacci-Lucas, [303] Three-Phase Cosmos Cancellation, [304] Polyphase Sum Structure

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Grade-reactive isomorphism: det(Mᵏ) = (−1)ᵏ (Cassini); odd k = odd-grade versor (reactive); even k = even-grade rotor (active); alternates at every T-step on the 24-clock | PASS |
| C2 | Rational spread = QA power factor: s(b,e) = e·e/(b·b+e·e) ∈ (0,1)∩ℚ for all A1 states; (1−s)+s=1 exactly; active fraction = 1−s = b·b/(b·b+e·e); replaces sin²(φ) in the observer layer | PASS |
| C3 | 45° locus = b=e diagonal: s(k,k)=½ for k=1..9; Singularity (9,9) on this locus; Satellite spreads {1/10, 1/5, 4/13, ½, 9/13, 4/5, 9/10} closed under s→1−s (reactive symmetry) | PASS |
| C4 | Reactive complement: s(e,b) = 1−s(b,e) exactly; Cosmos closed under (b,e)→(e,b); minimum + maximum Cosmos spread = 1 (at (9,1) and (1,9)) | PASS |
| C5 | T-step cross-spread: cs = (b·d−e²)²/(G·G′) where d=A1_mod(b+e,9), G=b²+e², G′=e²+d²; exact Fraction in [0,1] for all 72 Cosmos states; Singularity (9,9) has cs=0 (self-coupling) | PASS |

## Rational trigonometry as the QA observer language

**Why rational trig (not sin/cos/tan or hyperbolic trig):**

The QA observer layer requires exact arithmetic with no transcendental functions. Wildberger's spread and quadrance achieve this by replacing angle-based quantities with squared ratios:

| Standard phasor | QA rational-trig substitute | Expression |
|---|---|---|
| sin²(φ) | Wildberger spread | s(b,e) = e·e/G |
| cos²(φ) | Quadrance (active fraction) | 1−s = b·b/G |
| sin²(φ)+cos²(φ)=1 | s + (1−s) = 1 | exact Fraction identity |
| power factor = cos(φ) | PF² = 1−s | exact rational, no sqrt |

Hyperbolic trig (sinh/cosh/tanh) applies to the blue chromogeometric layer (hyperbolic/Minkowski geometry). The Euclidean polyphase power setting is elliptic; hyperbolic trig has no natural role here.

## Grade-reactive correspondence (C1)

The Cassini identity det(Mᵏ) = (−1)ᵏ divides the 24-clock into reactive (odd) and active (even) steps:

```
k=0 (0°):   det=+1  even-grade rotor  — active reference
k=1 (15°):  det=−1  odd-grade versor  — reactive step
k=6 (90°):  det=+1  even-grade rotor  — quarter-wave (90°)
k=12 (180°): det=+1 even-grade rotor  — M¹²=−I (grade inversion)
k=24 (360°): det=+1 even-grade rotor  — M²⁴=I (full cycle)
```

Note: k=6 (90°) is even-grade (det=+1), not reactive — this is the correct result because the 90° advance M⁶ is a **rotation** (rotor), not a **reflection** (versor). Pure reactive coupling requires an **odd-grade** step, not a 90° angular displacement.

## Spread interpretation for Cosmos

The Cosmos spread range is [1/82, 81/82] = [s(9,1), s(1,9)], achieved at the near-unity (small angle) and near-zero (large angle) extremes. Neither endpoint is reachable (A1 guarantees b,e ≥ 1). The Singularity (9,9) sits at the midpoint s=½, representing the 45° equal-reactive/active state.

**Satellite symmetry (C3):** The 8 Satellite states have spreads that come in complementary pairs {s, 1−s} around ½. This is the discrete signature that Satellite orbit states "know" both their reactive and active aspects equally — each Satellite state has a natural complement within the same orbit stratum.

## T-step cross-spread (C5)

The mutual reactive coupling between state (b,e) and its T-image (e,d) is:

```
cs(b,e) = (b·d − e·e)² / ((b²+e²)·(e²+d²))
```

where d = A1\_mod(b+e, 9). This is the Wildberger cross-spread between the two directions. Sample values:

| State (b,e) | T-image (e,d) | Cross-spread cs |
|---|---|---|
| (1,1) | (1,2) | 1/10 |
| (2,3) | (3,5) | 1/442 |
| (5,8) | (8,4) | 121/445 |
| (7,1) | (1,8) | 121/130 |
| (9,9) | (9,9) | 0 (self) |

Small cross-spread = states nearly aligned (small reactive coupling per step). Large cross-spread = states nearly orthogonal (large reactive coupling per step).

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.X
- Wildberger, N.J. (2005) *Divine Proportions: Rational Trigonometry*, Wild Egg Books, ISBN 978-0-9757492-0-8
- Hestenes, D. and Sobczyk, G. (1984) *Clifford Algebra to Geometric Calculus*, Reidel, ISBN 978-90-277-1673-6
