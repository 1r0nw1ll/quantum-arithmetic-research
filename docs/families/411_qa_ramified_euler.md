<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Bushnell-Henniart (2006) doi:10.1007/978-3-540-31511-7, Arthur-Clozel (1989) ISBN 978-0-691-08517-3 -->
# [411] QA Langlands Ramified Prime p=5 — Trivial GL₄/ℚ AI Euler Factor

**Cert family**: `qa_ramified_euler_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a (GL₂/F, conductor 𝔭₅³, level 125=5³), the GL₄/ℚ
automorphic induction AI(f) has trivial local Euler factor at p=5:

```
P_5^{ram}(Y) = 1    (degree 0 — the unit polynomial)
```

## Complete Prime Classification

| Class | Condition | Euler polynomial | Degree | Cert |
|---|---|---|---|---|
| Split | p ≡ 1, 4 mod 5 | 1−TY+(N+2p)Y²−pTY³+p²Y⁴ | 4 | [404] |
| Inert | p ≡ 2, 3 mod 5 | 1 + p²Y⁴ | 4 | [409] |
| **Ramified** | **p = 5** | **1** | **0** | **[411]** |

The partition {0} ∪ {1,4} ∪ {2,3} = {0,1,2,3,4} covers all residues mod 5.
Together [404]+[409]+[411] gives the **complete** unramified+ramified GL₄/ℚ AI Euler product.

## Three-Step Derivation

**Step 1 — Ramified prime identification.**
p=5 satisfies 5%5=0. It is the unique prime that is neither split (p≡1,4 mod 5) nor inert (p≡2,3 mod 5). It is the conductor prime of f (level 125=5³) and the discriminant prime of F=ℚ(√5) (disc=5).

**Step 2 — Supercuspidal local type.**
The conductor ideal of f is 𝔭₅³, so the conductor exponent of f at 𝔭₅ is n=3. By the local theory of GL₂ (Bushnell-Henniart §14):
```
n = 0  →  unramified principal series
n = 1  →  special (Steinberg) representation
n ≥ 2  →  supercuspidal
```
Since n=3 ≥ 2, the local component π_{𝔭₅} is **supercuspidal**.

**Step 3 — Trivial Euler factor.**
Supercuspidal representations of GL₂(F_{𝔭₅}) have trivial local L-factor:
```
L(s, π_{𝔭₅}) = 1
```
Under automorphic induction, Ind_{W_{F₅}}^{W_{ℚ₅}}(ρ_{𝔭₅}) has no Frobenius-fixed vectors in the inertia coinvariants. Therefore P_5^{ram}(Y) = 1 for AI(f) at p=5.

## Conductor Exponent of AI(f) at p=5

The Artin conductor formula for an induced representation gives:

```
a₅(AI(f)) = [F₅:ℚ₅] · n + dim(ρ_f) · f(F₅/ℚ₅)
           = 2 · 3  +  2 · 1  =  8
```

Where:
- [F₅:ℚ₅] = 2 (the prime 𝔭₅² = (5) in F=ℚ(√5), giving a ramified quadratic extension at p=5)
- n = 3 (conductor exponent of GL₂/F form f at 𝔭₅)
- dim(ρ_f) = 2 (GL₂ → 2-dimensional Weil-Deligne representation)
- f(F₅/ℚ₅) = 1 (discriminant exponent of the tamely ramified extension ℚ₅(√5)/ℚ₅)

All inputs are integers. No float observer projection enters the computation (Theorem NT).

## Degree Drop at p=5

At every unramified prime (split or inert), the GL₄/ℚ AI Euler polynomial has degree 4.
At p=5 (ramified), the degree drops to 0. This degree drop is a signature of the supercuspidal
local type and the high ramification of f at its conductor prime.

## Checks

- **C1**: p=5 has p%5=0; partition {0}∪{1,4}∪{2,3} is complete and disjoint — PASS
- **C2**: Conductor exponent n=3 ≥ 2 → supercuspidal (n=1 is Steinberg, n=0 is unramified) — PASS
- **C3**: P_5^{ram}(Y) = [1] (degree 0), below split/inert degree 4 — PASS
- **C4**: Artin formula: 2·3 + 2·1 = 8 (all integer, Theorem NT satisfied) — PASS

## Chain

- Parallel to [404] (split) and [409] (inert): the three certs classify ALL primes for AI(f)
- [410] (Dedekind ζ_{ℚ(√5)}=ζ·L(s,χ₅)) shows the same {split/inert/ramified} trichotomy
  at the level of ζ_F; [411] shows it at the GL₄/ℚ AI level
- With [411], the Langlands ladder [403]→[404]→…→[411] has a complete Euler product description
