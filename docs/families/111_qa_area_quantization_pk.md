# [111] QA Inert Prime Area Quantization

**Family ID**: 111
**Schema**: `QA_AREA_QUANTIZATION_PK_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]
**Directory**: `qa_alphageometry_ptolemy/qa_area_quantization_pk/`

---

## Purpose

[111] generalizes [108] (QA Area Quantization, mod-9 specific) to arbitrary inert-prime moduli p^k. It certifies the following theorem as a machine-verifiable object:

> **Inert Prime Area Quantization Theorem**: Let p be a prime inert in Z[φ] (i.e. x²+x−1 is irreducible mod p, equivalently the Legendre symbol (5/p) = −1). For any k ≥ 1 and modulus m = p^k, the image of the norm form f(b,e) = b²+be−e² over Z/mZ is exactly:
>
> Im(f) = { r ∈ Z/p^k Z : v_p(r) ≠ 1 }
>
> The forbidden set — residues not in the image — is { r : p | r but p² ∤ r }.

Family [108] is the special case p=3, k=2. Family [111] proves the same obstruction holds for all inert primes and all exponents k ≥ 1, with the validator exhaustively recomputing the spectrum for each certified instance.

---

## Which Primes are Inert?

p is inert in Z[φ] iff the discriminant 5 is a quadratic non-residue mod p:

| Prime p | Legendre(5,p) | Status |
|---------|---------------|--------|
| 3 | −1 | **inert** |
| 7 | −1 | **inert** |
| 11 | +1 | splits |
| 13 | −1 | **inert** |
| 17 | −1 | **inert** |
| 19 | +1 | splits |
| 23 | −1 | **inert** |

Z[φ]/pZ[φ] ≅ GF(p²) for inert p; Z[φ]/p^k Z[φ] is the Galois ring GR(p^(2k), 2).

---

## Theorem Intuition

For inert p, there is no element of Z[φ] with norm exactly p (just as there is no Gaussian integer with norm 3 since 3 is inert in Z[i]). Modding out by p^k, this obstruction persists: any element x ∈ Z[φ]/p^k Z[φ] with x ≢ 0 mod p but N(x) ≡ 0 mod p must have N(x) ≡ 0 mod p². So residues of p-adic valuation exactly 1 are never norms.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected` ⊇ [0..5] | `GATE_POLICY_INCOMPATIBLE` |
| PK1 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| PK2 | `modulus` == p^k | `MODULUS_MISMATCH` |
| PK3 | Exhaustive spectrum recompute matches `theorem_claim.spectrum` | `SPECTRUM_MISMATCH` |
| PK4 | `theorem_claim.forbidden` matches `{r : v_p(r)=1}` | `FORBIDDEN_SET_MISMATCH` |

The validator is a recompute-not-trust design: it never accepts the certificate's claimed spectrum at face value. It always runs the O(m²) enumeration loop and compares.

---

## Certified Fixtures

### `pk_pass_p3_k2.json` — anchor case

p=3, k=2, modulus=9. Reduces to [108]. Spectrum={0,1,2,4,5,7,8}, forbidden={3,6}.

### `pk_pass_p3_k3.json` — first generalization

p=3, k=3, modulus=27. Spectrum has 21 elements; forbidden={3,6,12,15,21,24} (residues with v₃=1).

### `pk_pass_p7_k2.json` — second inert prime

p=7, k=2, modulus=49. Spectrum has 43 elements; forbidden={7,14,21,28,35,42} (residues with v₇=1). Proves the theorem is not an artifact of p=3.

### `pk_fail_wrong_forbidden.json`

p=3, k=2, claims forbidden={3,6,9}. But 9 mod 9 = 0, which has v₃=∞ (not 1) and IS achievable (f(0,0)=0). Fails with `FORBIDDEN_SET_MISMATCH`.

---

## Relation to [108]

[108] QA_AREA_QUANTIZATION_CERT.v1 and [111] QA_AREA_QUANTIZATION_PK_CERT.v1 are sibling families — both family_extensions of [107] QA_CORE_SPEC.v1. [108] is a self-contained cert for the specific mod-9 case with its own schema and validator. [111] generalizes the theorem with explicit p, k parameters and a PK1 inert-primality check that [108] does not include.

```
[107] QA_CORE_SPEC.v1 (kernel)
    │
    ├─── [108] QA_AREA_QUANTIZATION_CERT.v1    (specific: p=3, k=2)
    │         [certified by inherit_pass_107_to_108.json in [109]]
    │
    └─── [111] QA_AREA_QUANTIZATION_PK_CERT.v1 (general: any inert p, any k≥1)
              [certified by inherit_pass_107_to_111.json in [109]]
```

---

## Running

```bash
# Self-test (4 fixtures)
python qa_area_quantization_pk/qa_area_quantization_pk_validate.py --self-test

# Single cert
python qa_area_quantization_pk/qa_area_quantization_pk_validate.py \
  --file qa_area_quantization_pk/fixtures/pk_pass_p7_k2.json

# Full meta-validator
python qa_meta_validator.py
```
