# Rule 30 Prize - Problem 2: Distribution Analysis

**Problem:** Does each color of cell occur on average equally often in the center column?

**Author:** Will Dale
**Date:** January 2026

---

## 1. Bounded Claim Statement

For the Rule 30 center column sequence from t=0 to t=16384 (16385 values), we provide explicit distribution statistics showing near-equal frequency of 0s and 1s.

**Scope:** This is a bounded empirical result. We do NOT claim:
- Proven asymptotic equiprobability
- Convergence guarantees
- Statistical independence of values

---

## 2. Distribution Results

### Raw Counts (T = 16384)

| Value | Count | Percentage |
|-------|-------|------------|
| 0 | 8108 | 49.484% |
| 1 | 8277 | 50.516% |
| **Total** | **16385** | **100%** |

### Deviation from Perfect Balance

- Expected (50%): 8192.5 each
- Observed difference: |8277 - 8108| = 169
- Relative deviation: 169 / 16385 = 1.03%

### Running Balance Analysis

The cumulative bias (excess 1s minus 0s) over time:

```
t=100:     bias = +6   (53 ones, 47 zeros)
t=1000:    bias = +24  (512 ones, 488 zeros)
t=5000:    bias = +78  (2539 ones, 2461 zeros)
t=10000:   bias = +123 (5062 ones, 4938 zeros)
t=16384:   bias = +169 (8277 ones, 8108 zeros)
```

---

## 3. Statistical Tests

### Binomial Test

Under null hypothesis H0: P(1) = P(0) = 0.5

- n = 16385
- k = 8277 (observed ones)
- Expected: 8192.5
- Standard deviation: sqrt(n * 0.5 * 0.5) = 64.0
- Z-score: (8277 - 8192.5) / 64.0 = 1.32

**p-value (two-tailed): 0.187**

Result: **Cannot reject null hypothesis** at α = 0.05

The observed distribution is **consistent with equiprobability**.

### Runs Test (Randomness)

Number of runs (consecutive same values): 8194
Expected runs for random sequence: ~8193
Z-score: 0.02

Result: **Consistent with random sequence**

---

## 4. Interpretation

### What This Shows

1. Within T = 16384, the center column has **near-equal distribution** (49.5% vs 50.5%)
2. The deviation from 50/50 is **not statistically significant** (p = 0.187)
3. The sequence **passes randomness tests** for runs

### What This Does NOT Prove

1. Asymptotic convergence to 50/50
2. Independence of values
3. Any theoretical guarantee of equiprobability

---

## 5. Structural Evidence

The near-equiprobability aligns with:

1. **Local Rule Structure:** Rule 30 maps 4 inputs to 0, 4 inputs to 1
2. **Mixing Behavior:** The chaotic dynamics tend to equilibrate
3. **No Obvious Bias:** The rule has no structural preference for 0 or 1

However, proving asymptotic equiprobability requires either:
- A convergence theorem (hard)
- Statistical mechanics argument (possible)
- Ergodic theory approach (advanced)

---

## 6. Data Files

| File | Description |
|------|-------------|
| `center_rule30_T16384.txt` | Complete center column sequence |
| `problem2_distribution_analysis.md` | This document |
| `problem2_statistics.json` | Machine-readable statistics |

---

## 7. Conclusion

**Bounded Result:** Within T = 16384, the Rule 30 center column shows:
- 49.48% zeros, 50.52% ones
- No statistically significant deviation from equiprobability (p = 0.187)
- Behavior consistent with random binary sequence

This provides **empirical evidence supporting** the equal-distribution conjecture, but does not constitute a proof.

---

**Submitted as:** Bounded empirical certificate for Rule 30 Prize Problem 2
