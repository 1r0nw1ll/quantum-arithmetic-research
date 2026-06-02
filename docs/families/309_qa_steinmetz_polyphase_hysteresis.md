# [309] QA Steinmetz Polyphase Hysteresis

**Family**: `qa_steinmetz_polyphase_hysteresis_cert_v1`  
**Depends on**: [298] Orbit Grade Decomposition, [303] Three-Phase Cosmos Cancellation, [304] Polyphase Sum Structure, [305] Reactive Power Versor Coupling

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Single-cycle orbit coupling sum S = Σ_{k=0}^{23} cs(T^k(1,1)) is an exact Fraction; cs(b,e)=(b·d−e²)²/((b²+e²)(e²+d²)); S = 191484369324836748/23958756907457125 | PASS |
| C2 | Maximum single-step coupling: cs_max = 1600/1681 = (40/41)² at state (1,9) (k=22); symmetric case G=G'=82; numerator=(1·1−81)²=6400; largest cross-spread in the orbit | PASS |
| C3 | Polyphase coupling linearity: three 3-phase copies T⁰=(1,1), T⁸=(7,1), T¹⁶=(4,1) each give orbit sum S; total 3-phase = 3·S (exact Fraction; parallel to cert [304] 3I theorem) | PASS |
| C4 | Steinmetz exponent n≈1.6 in P_h=k_h·f·B_max^n is observer; the discrete orbit provides exact cross-spread values without parameterization; the exponent approximates cs_k variation at the observer layer | PASS |
| C5 | Theorem NT: P_h = k_h·f·S_observer (float); k_h and B_max are material constants (observer); S is Fraction (QA layer); neither re-enters T-step orbit logic | PASS |

## Key result

Steinmetz (1892) identified the hysteresis loss per cycle as proportional to ∮H·dB. In QA, the analogous quantity is the sum of T-step cross-spreads over one complete 24-step Cosmos orbit:

**S = Σ cs(T^k(1,1)) = 191484369324836748/23958756907457125 ≈ 7.993 (exact Fraction)**

This sum is orbit-independent: any of the 72 Cosmos states gives the same S (the orbit traverses the same 24 states in a different order). 

### Maximum coupling state

The state (1,9) at k=22 achieves cs = **(40/41)² = 1600/1681 ≈ 0.952**, the maximum single-step cross-spread in the entire orbit. The elegance:

- State (1,9): b=1, e=9, d=A1_mod(10,9)=1 → T(1,9)=(9,1)
- G = b²+e² = 82, G' = e²+d² = 82  (symmetric: G=G')
- cs = (1·1 − 81)² / (82·82) = 80²/82² = (40/41)²

The symmetry G=G' makes this a perfect-square Fraction.

### Polyphase linearity (3-phase)

The three 3-phase orbit representatives {T⁰=(1,1), T⁸=(7,1), T¹⁶=(4,1)} form a triad (cert [303]). Since T is a bijection on Cosmos, each traverses the full 24-orbit state-set — in different order, same sum. Therefore:

**3-phase hysteresis sum = 3·S** (exact, not approximate)

This is the QA counterpart of the Steinmetz polyphase formula Σ P_h,i = n·P_h,1.

### Theorem NT boundary

| Layer | Expression | Type |
|-------|-----------|------|
| QA (discrete) | S = Σ cs_k | exact Fraction |
| Observer | P_h = k_h · f · S_observer | float (material constant) |
| Steinmetz exponent | n ≈ 1.6 | empirical, observer only |
