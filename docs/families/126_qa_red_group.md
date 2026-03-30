# Family [126] — QA_RED_GROUP_CERT.v1

**QA T-operator as red isometry; Fibonacci shift = φ-multiplication in Z[√5]/mZ[√5]**

---

## What this family certifies

The QA T-operator is the Fibonacci shift

```
F = [[0, 1],
     [1, 1]]
```

acting on state pairs (b, e) as: `(b, e) → (e, b+e mod m)`

This matrix represents **multiplication by φ = (1+√5)/2** in the split-complex ring
Z[√5]/mZ[√5], which is Wildberger's **red isometry group** for the split-complex line.

The cert verifies three key algebraic invariants of F:

| Property | Value | Meaning |
|----------|-------|---------|
| det(F) | −1 (≡ m−1 mod m) | Red norm N\_red(φ) = φ·ψ = −1 (φ is a norm−1 unit) |
| trace(F) | 1 | φ + ψ = 1 (minimal polynomial x²−x−1) |
| ord(F) in GL₂(Z/mZ) | cosmos orbit period | Orbit period = order of red rotation |

---

## Why affine orbit period = ord(F)

The QA step (1-indexed) is `(b,e) → (e, ((b+e−1) mod m)+1)`. In 0-indexed form this is the
AFFINE map `F(x) = F·x + c` with `c = (0,1)ᵀ`. The period P of this affine map satisfies:

```
F^P · x + (I + F + ... + F^{P−1}) · c ≡ x  (mod m)
```

When `F^{P/2} ≡ −I mod m` (which holds for m=9 and m=3), the translation sum vanishes:

```
(I + F + ... + F^{P−1}) = (I + ... + F^{P/2−1})(I + F^{P/2}) = S · 0 = 0
```

Therefore the affine period equals `ord(F)` exactly for these moduli.

---

## Standard orbit periods

| Modulus m | Cosmos period | F^{P/2} mod m | Verification |
|-----------|---------------|----------------|--------------|
| m = 9 | 24 | F^{12} = −I = [[8,0],[0,8]] | F^{24} = I ✓ |
| m = 3 | 8 | F^{4} = −I = [[2,0],[0,2]] | F^{8} = I ✓ |

---

## Validation checks (RG1–RG7)

| ID | Check | Fail type |
|----|-------|-----------|
| RG1 | `schema_version == 'QA_RED_GROUP_CERT.v1'` | SCHEMA\_VERSION\_WRONG |
| RG2 | `T_matrix == [[0,1],[1,1]]` (Fibonacci shift) | T\_MATRIX\_WRONG |
| RG3 | `det(T_matrix) ≡ −1 mod m` | DET\_NOT\_MINUS\_ONE |
| RG4 | `trace(T_matrix) ≡ 1 mod m` | TRACE\_NOT\_ONE |
| RG5 | `T_matrix^orbit_period ≡ I mod m` | ORBIT\_PERIOD\_WRONG |
| RG6 | `T_matrix^(P/k) ≢ I mod m` for prime k\|P (minimality) | PERIOD\_NOT\_MINIMAL |
| RG7 | orbit\_type ∈ valid set; period=1 ↔ singularity | ORBIT\_TYPE\_MISMATCH |

---

## Fixtures

| File | m | period | orbit\_type | Result | Notes |
|------|---|--------|-------------|--------|-------|
| `rg_pass_m9_cosmos.json` | 9 | 24 | cosmos | PASS | Anchor: ord(F)=24 in GL₂(Z/9Z) |
| `rg_pass_m3_cosmos.json` | 3 | 8 | cosmos | PASS | ord(F)=8 in GL₂(Z/3Z) |
| `rg_fail_wrong_period.json` | 9 | 12 (wrong) | cosmos | FAIL | ORBIT\_PERIOD\_WRONG: F^12=−I≢I; projective order 12 ≠ linear order 24 |

---

## Mathematical context

**Source**: Wildberger arXiv:math/0701338 (1D Metrical Geometry).

Three chromatic isometry groups on P¹ over a field F:
- **Blue**: [ac−bd : ad+bc] — complex multiplication (Euclidean)
- **Red**: [ac+bd : ad+bc] — split-complex multiplication (Minkowski / Z[√D])
- **Green**: [ac : bd] — multiplicative (null)

For D=5 (QA uses Q(√5)), the red isometry matrix for element u+v√5 is `[[u,5v],[v,u]]`
with norm u²−5v². The golden ratio φ = (1+√5)/2 has N\_red(φ) = −1.

The Fibonacci shift F = [[0,1],[1,1]] is the linear part of QA T applied to 0-indexed
pairs. Its eigenvalues are φ and ψ = −1/φ, satisfying the red norm identity
`φ · ψ = −1 = det(F)`. The orbit period of the QA affine map equals `ord(F)` in GL₂(Z/mZ)
whenever `F^{P/2} = −I mod m`.

**Connection to family [125]**: Family [125] certifies C=Q\_green, F=Q\_red, G=Q\_blue. This
family [126] certifies that the QA T-operator that GENERATES the (b,e) sequence is a red
isometry — the arithmetic dynamics live in the red chromatic geometry.

**Connection to family [127]**: The UHG null cert will certify that QA triples are null points
in UHG, which requires the underlying arithmetic to be over split-complex (red) geometry —
exactly what this family establishes.

---

## ok semantics

`ok=True` means the certificate is internally consistent:
detected failure types == declared `fail_ledger`, and `result` field is consistent.
