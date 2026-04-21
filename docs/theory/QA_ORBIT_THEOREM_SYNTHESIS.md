<!-- PRIMARY-SOURCE-EXEMPT: reason=one-page synthesis of QA Orbit Stratification Theorem, derived from empirical + algebraic work in QA_ORBIT_STRATIFICATION_THEOREM.md; companion summary, not a derivation from external literature -->

# QA Orbit Theorem — One-Page Synthesis

**Full derivation + proofs**: `docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md`.
**Background + audit trail**: `docs/theory/QA_GENERATOR_REACHABILITY.md`.

## Setup

`(b, e) ∈ (ℤ/mℤ)²` under the QA generator pair
```
σ(b, e) = (e, b + e)              // Fibonacci forward (qa_step)
μ(b, e) = (e, b)                  // swap
```

---

## Part I — `⟨σ, μ⟩` orbits (coarse classification)

Both σ and μ are matrix multiplications by elements of GL₂(ℤ):
```
σ = F · ,  F = [[0,1],[1,1]]       μ = S · ,  S = [[0,1],[1,0]]
SF = T = [[1,1],[0,1]],  FS = Tᵀ,  det F = det S = −1
```

`⟨F, S⟩ = GL₂(ℤ)` (T + Tᵀ generate SL₂(ℤ); det S = −1 gives the non-SL₂ coset). Reduction mod m is surjective, so **`⟨σ, μ⟩ = GL₂(ℤ/mℤ)`**.

**Theorem I.** For any modulus m, `⟨σ, μ⟩`-orbits on `(ℤ/mℤ)²` are exactly the **content-ideal classes** — determined at each prime-power factor `p^k` of m by
```
J(b, e) = min( v_p(b), v_p(e) )   ∈  {0, 1, …, k}.
```
For `m = ∏ p_i^{k_i}` (CRT), the orbit decomposition is the Cartesian product of the per-factor `(k_i + 1)`-level stratifications. Total component count `= ∏ (k_i + 1)`.

**Sizes** (on prime power `p^k`): `|L_j| = p^{2(k−j−1)}·(p²−1)` for `j < k`, `|L_k| = 1`. For composite m, sizes multiply.

**Corollary**: canonical QA `Cosmos / Satellite / Singularity` = `(L_0 / L_1 / L_k)` on the single-prime-power factor, combined via CRT.

---

## Part II — σ-only sub-stratification (fine structure)

Within each `L_j` (reduced via scaling to `L_0` of a smaller prime-power), σ-orbits are classified by how `x² − x − 1` factors mod p — equivalently by the Legendre symbol `(5 | p)`.

| Case | primes | factoring mod p | σ-orbit structure on L_0 of `p^n` |
|---|---|---|---|
| **A. Inert** | `p = 2` or `(5\|p) = −1` (p ≡ ±2 mod 5: 2, 3, 7, 13, 17, 23, …) | irreducible | uniform length π(p^n); count `|L_0|/π(p^n)` |
| **B. Split** | `(5\|p) = +1` (p ≡ ±1 mod 5: 11, 19, 29, 31, …) | `(x−φ)(x−ψ)` with φψ = −1 | two eigenspaces contribute `(p−1)p^{n−1}/ord(λ)` orbits of length `ord(λ)` each; generic orbits of length `π(p^n) = lcm(ord φ, ord ψ)` |
| **C. Ramified** | `p = 5` (discriminant of `x²−x−1`) | double root 3; Jordan block | for `n = 1`: `1×π(5) + 1×ord(3)` = `1×20 + 1×4`. For `n ≥ 2`: `5^{n−1}×π(5^n) + 5^{n−1}×π(5^{n−1})` (Jordan depth-1 lift) |

**CRT for composite m**: σ-orbit lengths combine by `lcm` across prime-power factors; orbit counts multiply.

---

## What μ does

μ is precisely the collapse operator from Part II's Fibonacci-linear refinement to Part I's GL₂-content classification:

- In Case B (Split), μ **identifies the two `F_p`-eigenspaces** of F.
- In Case C (Ramified), μ **mixes the Jordan strata**.
- In Case A (Inert), μ merges the σ-orbits directly (no eigenspaces to identify, but there are still multiple σ-orbits to collapse into one content-ideal class).

After μ acts, all refined σ-orbits within a single `L_j` merge into one `GL₂(ℤ/p^kℤ)`-orbit = `L_j` itself.

---

## QA interpretation

The canonical QA orbit partition from `qa_orbit_rules.py` is the **content-ideal / elementary-divisor** classification of pairs `(b, e)` under the full `GL₂(ℤ/mℤ)` action, CRT-factored. The Fibonacci-specific structure (σ's eigenvalues φ, ψ, Pisano periodicity, discriminant 5) lives one level below — it is what μ-less dynamics see, and what adjoining μ collapses.

**Two theorems, one sentence**:
> Part I classifies what `⟨σ, μ⟩` sees (content ideals); Part II classifies what σ alone sees before μ collapses it (Fibonacci companion-matrix orbits stratified by the mod-p factoring of `x² − x − 1`).

---

## Empirical verification footprint

| moduli tested directly | purpose |
|---|---|
| 8, 9, 15, 24, 25, 27, 45, 49, 72, 125 | Part I: level-set sizes, CRT products |
| 5, 25, 125 | Part II Case C: Jordan filtration |
| 11, 19, 29, 121 | Part II Case B: eigenspace + generic |
| 2, 3, 7, 13 | Part II Case A: uniform |

All measurements match predictions exactly. Part I propositions are proven; Part II is proven for Cases A and B (via diagonalisation) and empirically confirmed through n = 3 for Case C (Jordan structure; formal closed form follows from the `ε² = 5·unit` identity).
