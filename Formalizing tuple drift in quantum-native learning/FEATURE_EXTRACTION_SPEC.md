# QA World Model Feature Extraction Specification
## Paper 2 Implementation - Canonical Reference

This document defines the exact feature extraction from QA states to neural network inputs.
**Mathematical rigor required**: This must preserve the QA axioms from Paper 1.

---

## State Representation (21-Element Invariant Packet)

From Paper 1, a QA state is **primitives (b, e)** with derived values:

### Primitives
- `b`: integer ≥ 1
- `e`: integer ≥ 1

### Derived Base
- `d = b + e`
- `a = b + 2e = e + d`

### 21-Element Invariant Packet
```
1.  B = b²
2.  E = e²
3.  D = d²
4.  A = a²
5.  X = e·d
6.  C = 2·e·d
7.  F = b·a
8.  G = D + E = d² + e²
9.  L = (C·F)/12        [RATIONAL - exact Fraction]
10. H = C + F
11. I = |C - F|         [ABSOLUTE VALUE]
12. J = d·b
13. K = d·a
14. W = X + K = d(e+a)  [CANONICAL FORM]
15. Y = A - D
16. Z = E + K = e² + (a·d)
17. h² = d²·a·b         [EXACT INTEGER, no sqrt]
18. φ₉ = digital_root(a)    [1..9]
19. φ₂₄ = a mod 24          [0..23]
20. N = b² + e²         [For compatibility]
21. S = b + e + d + a   [Sum invariant]
```

---

## Feature Extraction Strategy

Following ChatGPT's 3-bucket approach:

### Bucket A: Small/Raw Features (Normalized to [0,1])
**Input**: `b, e, d, a, φ₉, φ₂₄`

For a given Cap bound `N` (e.g., N=30):
```rust
fn normalize_small(val: i64, max_val: i64) -> f64 {
    val as f64 / max_val as f64
}

// Features:
feat[0] = normalize_small(b, N)           // b/N
feat[1] = normalize_small(e, N)           // e/N
feat[2] = normalize_small(d, 2*N)         // d/(2N)
feat[3] = normalize_small(a, 3*N)         // a/(3N) since a=b+2e ≤ N+2N
feat[4] = normalize_small(φ₉, 9)          // φ₉/9
feat[5] = normalize_small(φ₂₄, 24)        // φ₂₄/24
```
**Dimension**: 6 features

### Bucket B: Large Invariants (Log-Scaled)
**Input**: `A, B, C, D, E, F, G, H, I, J, K, W, Y, Z, h²`

```rust
fn log_scale(val: i64) -> f64 {
    (1.0 + val as f64).ln()  // log1p(x)
}

// Features:
feat[6]  = log_scale(B)
feat[7]  = log_scale(E)
feat[8]  = log_scale(D)
feat[9]  = log_scale(A)
feat[10] = log_scale(X)
feat[11] = log_scale(C)
feat[12] = log_scale(F)
feat[13] = log_scale(G)
feat[14] = log_scale(H)
feat[15] = log_scale(I)
feat[16] = log_scale(J)
feat[17] = log_scale(K)
feat[18] = log_scale(W)
feat[19] = log_scale(Y)
feat[20] = log_scale(Z)
feat[21] = log_scale(h²)
feat[22] = log_scale(N)
feat[23] = log_scale(S)
```
**Dimension**: 18 features

### Bucket C: Rational Features (L = C·F/12)
**Input**: `L` as exact Fraction

```rust
fn extract_rational(frac: &Fraction) -> (f64, f64) {
    let num = frac.numer();
    let den = frac.denom();
    (
        (1.0 + (*num as f64).abs()).ln(),  // log1p(|numerator|)
        (1.0 + *den as f64).ln()            // log1p(denominator)
    )
}

// Features:
let (ln_num, ln_den) = extract_rational(&state.L);
feat[24] = ln_num
feat[25] = ln_den
```
**Dimension**: 2 features

---

## Total State Feature Dimension

**Total**: 6 + 18 + 2 = **26 features**

(Note: The document mentioned state_dim=128, which likely includes padding or additional derived features. We can expand with interaction terms if needed.)

### Potential Expansion to 128 Features

If we need to reach 128 dimensions, add:
1. **Pairwise products** (selected pairs like `b·e`, `d·a`, etc.)
2. **Ratio features** (`b/e`, `C/F`, etc.)
3. **Modular features** (additional phase information)
4. **Polynomial features** (squares/cubes of normalized values)
5. **Padding with zeros** if needed

---

## Generator Encoding

**Generators**: `{σ, μ, λ₂, ν}`

### One-Hot Encoding (4 dimensions)
```rust
fn generator_to_onehot(gen: &str) -> [f64; 4] {
    match gen {
        "sigma"   => [1.0, 0.0, 0.0, 0.0],
        "mu"      => [0.0, 1.0, 0.0, 0.0],
        "lambda2" => [0.0, 0.0, 1.0, 0.0],
        "nu"      => [0.0, 0.0, 0.0, 1.0],
        _ => panic!("Unknown generator")
    }
}
```

### Index Mapping
```rust
fn generator_to_index(gen: &str) -> usize {
    match gen {
        "sigma"   => 0,
        "mu"      => 1,
        "lambda2" => 2,
        "nu"      => 3,
        _ => panic!("Unknown generator")
    }
}
```

(Note: Document mentions gen_dim=16, suggesting possibly expanded generator features with embeddings or additional metadata.)

---

## Critical Correctness Properties

### ✅ MUST Preserve
1. **I = |C - F|** - Use absolute value, not raw difference
2. **W = X + K** - Use canonical form, not alternative expressions
3. **h² as integer** - Never take sqrt, keep exact
4. **L as Fraction** - Maintain num/den separately, no premature division
5. **No arbitrary reduction** - Preserve QA "no reduction" axiom

### ✅ Feature Boundary Only
- Exact arithmetic in oracle (integers/fractions)
- Convert to f64 ONLY at feature extraction
- Log-scaling prevents numerical overflow

### ✅ Deterministic
- Same state → same features (no randomness)
- Reproducible across runs

---

## Implementation Checklist

- [ ] Implement `construct_qa_state(b, e)` with all 21 invariants
- [ ] Verify I, W, h² match Paper 1 definitions exactly
- [ ] Implement 3-bucket feature extraction
- [ ] Test on known states (e.g., (b=3, e=4, N=30))
- [ ] Verify feature ranges are sensible (no NaN, no Inf)
- [ ] Confirm dimension matches model input (26 or 128)
- [ ] Validate against Python implementation (if available)

---

## Example Test Case

For state `(b=3, e=4)` with `N=30`:

### Derived Values
```
d = 7, a = 11
B = 9, E = 16, D = 49, A = 121
X = 28, C = 56, F = 33
G = 65, L = 56·33/12 = 1848/12 = 154
H = 89, I = 23, J = 21, K = 77
W = 105, Y = 72, Z = 93
h² = 49·11·3 = 1617
φ₉ = digital_root(11) = 2
φ₂₄ = 11
```

### Feature Vector (first 6 normalized features)
```
feat[0] = 3/30    = 0.1
feat[1] = 4/30    = 0.133
feat[2] = 7/60    = 0.117
feat[3] = 11/90   = 0.122
feat[4] = 2/9     = 0.222
feat[5] = 11/24   = 0.458
```

### Feature Vector (next 18 log-scaled features)
```
feat[6]  = ln(10)     = 2.303  (B=9)
feat[7]  = ln(17)     = 2.833  (E=16)
...
feat[21] = ln(1618)   = 7.389  (h²=1617)
```

### Feature Vector (rational features)
```
feat[24] = ln(1849)   = 7.522  (L numerator = 1848)
feat[25] = ln(13)     = 2.565  (L denominator = 12)
```

---

## References
- Paper 1: QA Transition System Theory
- ChatGPT guidance: 3-bucket strategy with log-scaling
- `qa_oracle.py`: Canonical 21-element packet construction
