# [313] QA Salient-Pole Reluctance Torque

**Family**: `qa_salient_pole_reluctance_cert_v1`  
**Depends on**: [298] Orbit Grade Decomposition, [305] Reactive Power Versor Coupling, [307] Induction Motor Slip, [308] Scott T-Transformer

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | d-q saliency ratio: s₈/s₆ = (9/25)/(1/50) = 18 (exact integer); d-axis spread is 18× the q-axis spread; positions from cert [308] Scott-T word | PASS |
| C2 | Rational double-spread 4s_k(1−s_k) ∈ (0,1)∩Q for k≠0,12 (Fraction, Wildberger analog of sin²(2δ)); Δ₀=Δ₁₂=0; all Δ_k are exact Fractions | PASS |
| C3 | argmin_k Δ_k = 9; Δ_min = −479993/515450 (exact); T⁹=(1,8) is the maximum-braking position: s₉=49/130 ≈ 0.377 but s₁₈=1/122 ≈ 0.008 | PASS |
| C4 | Sign partition: Δ_k > 0 for exactly k∈{4,10,11,13,14} (5 near-Singularity acceleration zones); Δ_k=0 for {0,12}; Δ_k < 0 for 17 positions; argmax k=11, Δ_max=324551/862025 | PASS |
| C5 | Theorem NT: reluctance torque T_r ∝ Δ_k is observer; saliency X_d−X_q and sin(2δ) are observer projections; QA orbit defect Δ_k is the discrete structural pre-image | PASS |

## Key result

The **reluctance defect** Δ_k = s_{2k mod 24} − 4s_k(1−s_k) measures how far the Fibonacci orbit deviates from ideal uniform rotation at each T-step. For an ideal uniform rotation, s_{2k} = 4s_k(1−s_k) exactly (Wildberger double-spread identity). The Fibonacci orbit never satisfies this — it alternates between "braking" (Δ_k < 0) and "acceleration" (Δ_k > 0) phases.

### d-q Saliency Ratio = 18

From cert [308] (Scott T-Transformer), the d-axis position is T⁸=(7,1) and q-axis is T⁶=(4,3):

| Axis | Position | Spread from (1,1) |
|------|----------|-------------------|
| d-axis (k=8) | (7,1) | s₈ = 9/25 |
| q-axis (k=6) | (4,3) | s₆ = 1/50 |
| **Ratio** | | **s₈/s₆ = 18** |

The Fibonacci orbit has extreme intrinsic saliency at the Scott-T split: the d-axis direction is angularly 18× more "distant" from the seed (1,1) than the q-axis direction.

### Defect sign partition

| Category | k values | Count | Interpretation |
|----------|----------|-------|----------------|
| Zero | {0, 12} | 2 | Synchronous + antipodal fixed points |
| Positive (acceleration) | {4, 10, 11, 13, 14} | 5 | Near-Singularity escape: small s_k but large s_{2k} |
| Negative (braking) | {1,2,3,5,6,7,8,9,15,…,23} | 17 | Orbit slower than ideal rotation |

The 5 acceleration zones all occur where the orbit has just passed through a near-Singularity region (s_k tiny) and the doubled step lands at a high-spread state (s_{2k} relatively large). The orbit "compresses and then expands" rather than rotating uniformly.

### Maximum braking at k=9

State T⁹=(1,8): spread s₉ = 49/130 ≈ 0.377 (near-maximum), but its doubled step T¹⁸=(5,6) has spread s₁₈ = 1/122 ≈ 0.008 (near-zero). The orbit "slams the brakes" — a large spread state maps to a near-zero spread state at double step. This gives Δ₉ = −479993/515450 ≈ −0.931, the most extreme defect in the orbit.

### Physical analog

In a salient-pole synchronous machine, reluctance torque T_r = (V²/2ω_s)·(1/X_q − 1/X_d)·sin(2δ). The QA analog replaces sin(2δ) with the rational double-spread 4s_k(1−s_k), and the saliency term (1/X_q − 1/X_d) is proportional to the defect Δ_k. The integer saliency ratio s_d/s_q = 18 and the sign-asymmetric defect profile are the discrete QA structure underlying the continuous reluctance formula.
