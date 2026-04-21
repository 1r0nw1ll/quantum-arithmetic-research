<!-- PRIMARY-SOURCE-EXEMPT: reason=standalone theorem note extracted from empirical work documented in QA_GENERATOR_REACHABILITY.md; generators σ and μ defined in qa_orbit_rules.py and tools/qa_viz/threejs/qa_core.js -->

# QA Orbit Stratification Theorem

**Status**: all three propositions **proven**. Empirical tables retained below as corroboration.
**Scope**: `(Z/mZ)²` under the QA generator pair `{σ, μ}`, where
```
σ(b, e) = (e, qa_mod(b + e))     μ(b, e) = (e, b)
```
and `qa_mod` is the A1-compliant mod-m map `((x − 1) mod m) + 1`.
**Companion**: `docs/theory/QA_GENERATOR_REACHABILITY.md` (full audit trail, semiconjugacy checks, retractions).

---

## Proposition A — prime-power invariant

On `(Z/p^k Z)²` with `p` prime, the function
```
J(b, e) := min( v_p(b), v_p(e) )   ∈  {0, 1, …, k}
```
is invariant under both `σ` and `μ`.

**Proof.**
- `J ∘ μ = J` trivially (μ swaps).
- For `σ`, consider two cases:
  - **`v_p(b) ≠ v_p(e)`**: say `v_p(b) = j < ℓ = v_p(e)`. Then `b + e ≡ b (mod p^{j+1})`, so `v_p(b+e) = j`. Hence `J(σ(b,e)) = min(ℓ, j) = j = J(b,e)`.
  - **`v_p(b) = v_p(e) = j`**: write `b = p^j u`, `e = p^j v` with `u, v` units mod p. Then `b + e = p^j (u + v)`, so `v_p(b+e) ≥ j`. Hence `J(σ(b,e)) = min(j, v_p(b+e)) = j = J(b,e)`. ☐

---

## Proposition B — level sets are exactly the components

For `m = p^k` and `L_j := { (b,e) : J(b,e) = j }`, the `k + 1` sets `L_0, L_1, …, L_k` are exactly the orbit components of `⟨σ, μ⟩` on `(Z/mZ)²`.

### Proof

Represent `(b, e) ∈ (Z/mZ)²` as a column vector. Then:
```
σ(b, e)  =  F · (b, e)ᵀ    with F = [[0, 1], [1, 1]]       (det F = −1)
μ(b, e)  =  S · (b, e)ᵀ    with S = [[0, 1], [1, 0]]       (det S = −1)
```
Both F and S are in GL₂(ℤ). Compute:
```
S · F  =  [[1, 1], [0, 1]]  =  T      (upper-triangular elementary)
F · S  =  [[1, 0], [1, 1]]  =  Tᵀ     (lower-triangular elementary)
```
T and Tᵀ generate SL₂(ℤ) (standard; they are the Nielsen generators). Since `det S = −1`, we also get the determinant-(−1) coset, so `⟨F, S⟩ = GL₂(ℤ)`.

The reduction map `GL₂(ℤ) → GL₂(ℤ/mℤ)` is surjective (strong approximation; equivalently, both T and S lift trivially and they generate the target). Hence
```
⟨σ, μ⟩  =  GL₂(ℤ/p^k ℤ)
```
as a transformation group on `(ℤ/p^k ℤ)²`.

**Orbit structure of GL₂(R) on R² for R = ℤ/p^k ℤ** (a local principal ideal ring): the orbit of a column `(b, e)ᵀ` is characterized by its **content ideal** `(b, e) · R`, which for this local ring is `(p^j)` with `j = min(v_p(b), v_p(e))`. By row-reduction (Hermite normal form over R), any pair with content `(p^j)` can be sent by a GL₂(R) element to `(p^j, 0)ᵀ`. Hence pairs with the same content are in the same orbit; pairs with different content lie in different orbits because the content is GL₂(R)-invariant (determinants of unit matrices are units, and units preserve the ideal).

Therefore the orbits of `⟨σ, μ⟩ = GL₂(ℤ/p^k ℤ)` on `(ℤ/p^k ℤ)²` are exactly the `k + 1` level sets `L_j`. ☐

**Orbit–stabilizer corroboration** (confirms |L_j| by group theory, independent of direct enumeration):
- `|GL₂(ℤ/p^k ℤ)| = p^{4(k−1)} · (p² − 1)(p² − p)`
- Stabilizer of `(1, 0)ᵀ` has order `p^k · φ(p^k) = p^k · p^{k−1}(p − 1) = p^{2k−1}(p − 1)`
- Orbit of `(1, 0)` has size `p^{2(k−1)} · (p² − 1)` = |L_0|. ✓

Verified on (p, k) ∈ {(2, 3), (3, 2), (5, 2), (7, 2)}: orbit sizes from orbit-stabilizer counting match `|L_0|` exactly in every case.

**Closed-form level sizes** (derived from level-set definitions, verified against empirical):
```
|L_j| = p^{2(k − j − 1)} · (p² − 1)     for 0 ≤ j < k
|L_k| = 1                                (the fixed pair (p^k, p^k))
```

**Empirical verification** (direct `{σ,μ}`-closure enumeration; all match exactly):

| modulus `p^k` | predicted sizes `[L_0, L_1, …, L_k]` | measured sizes | match |
|---|---|---|---|
| `2³ = 8`    | `[48, 12, 3, 1]`              | `[48, 12, 3, 1]`              | ✓ |
| `3² = 9`    | `[72, 8, 1]`                  | `[72, 8, 1]`                  | ✓ |
| `3³ = 27`   | `[648, 72, 8, 1]`             | `[648, 72, 8, 1]`             | ✓ |
| `5² = 25`   | `[600, 24, 1]`                | `[600, 24, 1]`                | ✓ |
| `5³ = 125`  | `[15000, 600, 24, 1]`         | `[15000, 600, 24, 1]`         | ✓ |
| `7² = 49`   | `[2352, 48, 1]`               | `[2352, 48, 1]`               | ✓ |

For every component in every case, all members share the same `J` value. No violations observed.

---

## Proposition C — CRT factorization

Let `m = ∏ p_i^{k_i}`. Under the CRT ring isomorphism `ℤ/mℤ ≅ ∏ ℤ/p_i^{k_i}ℤ`, matrix multiplication commutes with the factorization:
```
(Z/mZ)²      ≅  ∏ (Z/p_i^{k_i} Z)²
GL₂(Z/mZ)    ≅  ∏ GL₂(Z/p_i^{k_i} Z)
```
σ and μ are matrix multiplications by elements of GL₂(ℤ) (Proposition B proof), which descend to each CRT factor independently. By Proposition B the orbits on each factor are the `k_i + 1` level sets; the product action has orbits = products of per-factor orbits.

Therefore the `⟨σ, μ⟩`-orbit decomposition on `(ℤ/mℤ)²` is the Cartesian product of the per-factor decompositions, with `∏ (k_i + 1)` total components and sizes = products of per-factor level sizes. ☐

**Empirical verification** (4 composite moduli; all component counts AND sizes match predicted products):

| `m` | factoring | predicted count | measured sizes (sorted) |
|---|---|---|---|
| 15  | 3·5       | 2·2 = 4   | `[192, 24, 8, 1]` |
| 24  | 8·3       | 4·2 = 8   | `[384, 96, 48, 24, 12, 8, 3, 1]` |
| 45  | 9·5       | 3·2 = 6   | `[1728, 192, 72, 24, 8, 1]` |
| 72  | 8·9       | 4·3 = 12  | `[3456, 864, 384, 216, 96, 72, 48, 24, 12, 8, 3, 1]` |

---

## Corollary — QA class partition reinterpreted

The canonical `Cosmos / Satellite / Singularity` partition from `qa_orbit_rules.py` is the prime-power valuation stratification (Prop A) applied to each CRT factor of `m`, combined via Cartesian product for composite moduli.

On a prime power `m = p^k`:
- **Cosmos** = `L_0` (at least one coordinate a unit mod `p`)
- **Satellite** = `L_1` (both divisible by `p`, at least one not by `p²`)
- **Singularity** = `L_k` = `{(p^k, p^k)}` (both divisible by `p^k`)

On composite `m = p^k · q^ℓ · …`, the canonical satellite rule `(m/3) | b ∧ (m/3) | e` from `qa_orbit_rules.py` picks out `(L_k^{(p=2 or 3)} , L_0^{(other)})` — the "deepest" stratum in the dominant 3-power factor, paired with any level in the others. For `m = 24 = 2³·3` this is `L_3^{(mod 8)} × L_0^{(mod 3)}` = 8 pairs, matching the canonical satellite count.

---

## Reproduce

```bash
node --input-type=module -e "
import('/home/player2/signal_experiments/tools/qa_viz/threejs/qa_core.js').then(m => {
  // For arbitrary modulus, re-define using qa_core's constructors or inline σ/μ
  const {sigma, mu} = m;
  // Then closure from all (b,e) to get components; count per-component
  // min(v_p(b), v_p(e)) values.
});
"
```

Full multi-modulus script is in commit message of `e17242f` and can be extracted from git history.

---

---

# Part II — σ-only sub-stratification within each `L_j`

Once μ is removed, `⟨σ⟩ = ⟨F⟩ ≤ GL₂(ℤ/mℤ)` is cyclic of order π(m) (the Pisano period). The orbits of σ alone within each level set `L_j` stratify more finely than the `{σ, μ}` orbits. The classification is governed by **how the Fibonacci characteristic polynomial `x² − x − 1` factors mod p**, equivalently by the Legendre symbol `(5 | p)`.

## Reduction to L_0

Every `(b, e) ∈ L_j` has the form `(p^j · b', p^j · e')` with `(b', e') ∈ L_0` of `(ℤ/p^{k−j} ℤ)²`, and σ is compatible with this scaling: `σ(p^j b', p^j e') = (p^j e', p^j (b' + e'))`. So the σ-orbit of `(b,e)` in `L_j^{(p^k)}` ↔ σ-orbit of `(b', e')` in `L_0^{(p^{k−j})}`. **It suffices to classify σ-orbits on L_0 of (ℤ/p^n ℤ)² for all prime powers p^n.**

## Trichotomy by `(5 | p)`

The characteristic polynomial of F = [[0,1],[1,1]] is `x² − x − 1`, with discriminant 5. Its behaviour mod p determines three cases:

### Case A — Inert (p = 2 or `(5 | p) = −1`, i.e., p ≡ ±2 mod 5)

`x² − x − 1` is irreducible over `F_p`. F has no `F_p`-eigenvectors; its eigenvalues live in `F_{p²}`. On L_0 of `(ℤ/p^n ℤ)²`, every orbit has the uniform length π(p^n), and the number of orbits is `|L_0| / π(p^n)`.

**Verified** on (p, n) ∈ {(2, 3), (3, 2), (3, 3), (7, 2), (13, 1), (13, 2)}: all orbit lengths equal π(p^n).

### Case B — Split (`(5 | p) = +1`, i.e., p ≡ ±1 mod 5)

`x² − x − 1` has two distinct roots `φ, ψ ∈ F_p*` with `φ · ψ = −1`. F is diagonalisable over `F_p`; its two 1-dimensional eigenspaces, lifted to `(ℤ/p^n ℤ)²`, contribute:
- φ-eigenspace: `(p − 1) · p^{n−1} / ord_{p^n}(φ)` orbits of length `ord_{p^n}(φ)`
- ψ-eigenspace: `(p − 1) · p^{n−1} / ord_{p^n}(ψ)` orbits of length `ord_{p^n}(ψ)`

The remaining (non-eigenspace) vectors form **generic orbits of length π(p^n) = lcm(ord φ, ord ψ)**.

**Verified on split primes** (p ≡ ±1 mod 5):

| m = p^n | φ (ord) | ψ (ord) | π(p^n) | measured L_0 orbit structure |
|---|---|---|---:|---|
| 11    | 8 (10) | 4 (5)   | 10 | 1×10 (φ) + 2×5 (ψ) + 10×10 (generic) |
| 19    | 5 (9)  | 15 (18) | 18 | 2×9 (φ) + 1×18 (ψ) + 18×18 (generic) |
| 29    | 6 (14) | 24 (7)  | 14 | 2×14 (φ) + 4×7 (ψ) + 56×14 (generic) |
| 121 = 11² | — | — | 110 | 2×55 (short eigen) + 131×110 (everything else) |

All match prediction exactly.

### Case C — Ramified (p = 5)

The discriminant of `x² − x − 1` equals 5, so mod 5 it has a double root at `x ≡ 3`. F mod 5 is a Jordan block with eigenvalue 3 and geometric multiplicity 1.

**Key identity**: let `ε := F − 3I = [[−3, 1], [1, −2]]`. By Cayley–Hamilton `F² = F + I`, so
```
ε² = F² − 6F + 9I = (F + I) − 6F + 9I = −5F + 10I = 5·(2I − F) = 5·η
```
where `η := 2I − F = [[2, −1], [−1, 1]]` has `det η = 1`, so `η ∈ GL₂(ℤ/5^k ℤ)` for all `k`. **ε behaves like an element of valuation ½ in the ramified extension**: `ε² = 5 · unit`.

**σ-orbit structure on L_0 of `(ℤ/5^k ℤ)²`** (direct enumeration, k ∈ {1, 2, 3}):

| k  | \|L_0\|  | orbit structure                              | derivation                               |
|---:|---:|----------------------------------------------|------------------------------------------|
| 1  |   24 | `1×20 + 1×4`                                | `1 × π(5)` + `1 × ord_5(3) = 4` (eigenspace mod 5) |
| 2  |  600 | `5×100 + 5×20`                              | `5 × π(25) + 5 × π(5)`                   |
| 3  | 15000 | `25×500 + 25×100`                           | `25 × π(125) + 25 × π(25)`               |

**Pattern for k ≥ 2**: exactly **two strata on L_0**, each with `5^{k−1}` orbits:

- **Generic**: `5^{k−1}` orbits of length `π(5^k) = 4·5^k`.
- **Depth-1 (Jordan-filtered)**: `5^{k−1}` orbits of length `π(5^{k−1}) = 4·5^{k−1}` — vectors reducing mod 5 into the 1-D eigenspace of `F mod 5`, lifted to `(ℤ/5^k ℤ)²`.

The depth-1 stratum has size `(p−1) · 5^{2(k−1)} = 4 · 5^{2(k−1)}` (a 1-D F_5-line lifted to `(ℤ/5^{k−1} ℤ)²` with unit-coordinate constraint). Divided by `π(5^{k−1}) = 4·5^{k−1}` gives `5^{k−1}` orbits. ✓

**No deeper strata on L_0 for k ≥ 2**: vectors in higher Jordan layers reduce mod 5 to the zero eigenspace, which puts them in `L_1` (not `L_0`). In particular, no length-4 orbits on L_0 of `(ℤ/5^k ℤ)²` for k ≥ 2 — the length-4 phenomenon only appears at k = 1 (where eigenspace ⊂ L_0 directly).

**k = 1 as exception**: when k = 1 there is no "shallow lift" above the eigenspace, so the eigenspace itself carries the short-orbit contribution (length `ord_5(3) = 4`). For k ≥ 2, the eigenspace-over-F_5 requires a 5-lift to remain in L_0, and that lift promotes the orbit length to `π(5^{k−1})`.

**L_j reduction**: `L_j` on mod `5^k` scales to `L_0` on mod `5^{k−j}` (Proposition B's scaling map intertwines σ). Applying the closed form above to the smaller modulus recovers the full `L_j` structure — e.g., `L_1` on mod 25 = `L_0` on mod 5 = `1×20 + 1×4`. Verified.

**What μ does in Case C**: it identifies the two strata. Under `{σ, μ}` (Part I), the entire `L_0` of mod `5^k` becomes a single component (size `|L_0|`), collapsing both the generic and depth-1 σ-orbits. This is exactly what Proposition B predicts from the `GL₂(ℤ/5^k ℤ)`-action classification.

## Composite m via CRT

Exactly as in Proposition C, σ-only orbits on `(ℤ/mℤ)²` factor by CRT into per-prime-power σ-only orbits. Each factor uses Case A/B/C per Legendre symbol. Orbit lengths on composite m are **lcm of per-factor orbit lengths**; orbit counts multiply.

## Upshot — what μ collapses

The `{σ, μ}` theorem (Part I) says orbits are labelled by `J = min(v_p(b), v_p(e))`. The σ-only theorem (Part II) refines each `L_j` further by the `(5 | p)` trichotomy, with eigenspace orbits carrying shorter-than-generic lengths when they exist.

μ collapses all of these substructures into a single `L_j` component per level. Concretely: **μ is the one generator that identifies the two `F_p`-eigenspaces** (in the Split case) and **mixes the Jordan strata** (in the Ramified case).

---

## Closed (previously open)

1. ~~Closed-form proof of Proposition B~~ — **done** (GL₂(ℤ/p^k ℤ) action + content-ideal orbit classification). The Pisano-periodicity path was a red herring; the proof goes through elementary matrices and local-ring linear algebra instead.
2. ~~Primes `p ≥ 11` and higher `k`~~ — Propositions A/B/C hold for **all** primes `p` and all `k ≥ 1` (the proof makes no hypothesis on `p` or `k` beyond `p` being prime). The prime-specific empirical checks for `p ∈ {2, 3, 5, 7}` are corroborations, not the base of the claim.

## Still open

- **Generator ν.** Not required for stratification; parked. Potential role: explicit bijection `L_j → L_{j-1}` via `p-1`-like scaling (but ν as defined is only a permutation of `(ℤ/mℤ)²`, so it acts within one `L_j`, not between them).
- **Algebraic interpretation of the QA Fibonacci generator σ.** The proof uses only that σ ∈ GL₂(ℤ); the Fibonacci structure (σ's specific eigenvalues φ, ψ) is not invoked. Are there finer invariants that DO distinguish subfamilies within each `L_j` under σ alone (before adjoining μ)? Yes — the σ-only orbits have length dividing π(p^k) (Pisano period), and the number of σ-orbits within `L_0` is `|L_0| / π(p^k)`. This is a richer structure that μ collapses.
