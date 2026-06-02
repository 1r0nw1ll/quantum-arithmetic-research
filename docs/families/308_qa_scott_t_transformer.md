# [308] QA Scott T-Transformer

**Family**: `qa_scott_t_transformer_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [294] SL(2,Z) Spine, [296] SL(2,Z) Versor Isomorphism, [303] Three-Phase Cosmos Cancellation

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 3-phase subgroup {0,8,16} ≤ Z/24Z: M^8 has order 3 (M^8≠I, M^16≠I, M^24=I mod 24); 8 T-steps = 120° equivalent in the 24-step orbit | PASS |
| C2 | 2-phase subgroup {0,6,12,18} ≤ Z/24Z: M^6 has order 4 (M^6,M^12,M^18≠I, M^24=I mod 24); 6 T-steps = 90° equivalent | PASS |
| C3 | Scott-T SL(2,Z) word: W = R⁻¹·L⁻¹ = [[2,−1],[−1,1]] = M^{−2}; M²·W=I (exact); M^8·W=M^6 mod 24; unique 2-step correction bridging 3-phase to 2-phase | PASS |
| C4 | Rational Scott-T coefficient: direction spread s(T⁶(1,1), T⁸(1,1)) = 289/1250 (exact Fraction); T⁶=(4,3), T⁸=(7,1); s=(4·1−3·7)²/(25·50) = 289/1250 | PASS |
| C5 | √3/2 ≈ 0.866 is observer projection; 3→2 phase transform = group operation W=R⁻¹L⁻¹ on Z/24Z orbit clock; no transcendental arithmetic in QA layer | PASS |

## Key result

The Scott-T transformer (3-phase → 2-phase conversion) corresponds to a single SL(2,Z) word in the orbit clock algebra:

```
3-phase: M^8 (order 3, positions {0, 8, 16} in Z/24Z)
2-phase: M^6 (order 4, positions {0, 6, 12, 18} in Z/24Z)
Scott-T: W = M^{-2} = R^{-1}·L^{-1} = [[2,-1],[-1,1]]
```

The key identity: **M⁸ · W = M⁶** (verified mod 24).

The C-phase position (k=8) maps to the Q-axis position (k=6) by applying the Scott-T word W exactly once. The offset is 8 − 6 = 2 T-steps, and M^{−2} = R^{−¹}L^{−1} is the unique length-2 inverse SL(2,Z) word.

### Group-theoretic structure

| System | Subgroup | Generator | Order | T-step spacing |
|--------|----------|-----------|-------|----------------|
| 3-phase | {0,8,16} | M^8 | 3 | 8 (= 120° equiv) |
| 2-phase | {0,6,12,18} | M^6 | 4 | 6 (= 90° equiv) |

### Physical analog

The physical Scott-T transformer uses a √3/2 ≈ 0.866 tap ratio. In QA, this becomes the rational direction spread 289/1250 ≈ 0.231 between the 3-phase C-position T⁸=(7,1) and the 2-phase Q-axis T⁶=(4,3). The correction is algebraically exact: W = M^{−2} = R^{−1}L^{−1}, no irrational arithmetic required.
