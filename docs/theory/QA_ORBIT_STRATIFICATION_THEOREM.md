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

## Closed (previously open)

1. ~~Closed-form proof of Proposition B~~ — **done** (GL₂(ℤ/p^k ℤ) action + content-ideal orbit classification). The Pisano-periodicity path was a red herring; the proof goes through elementary matrices and local-ring linear algebra instead.
2. ~~Primes `p ≥ 11` and higher `k`~~ — Propositions A/B/C hold for **all** primes `p` and all `k ≥ 1` (the proof makes no hypothesis on `p` or `k` beyond `p` being prime). The prime-specific empirical checks for `p ∈ {2, 3, 5, 7}` are corroborations, not the base of the claim.

## Still open

- **Generator ν.** Not required for stratification; parked. Potential role: explicit bijection `L_j → L_{j-1}` via `p-1`-like scaling (but ν as defined is only a permutation of `(ℤ/mℤ)²`, so it acts within one `L_j`, not between them).
- **Algebraic interpretation of the QA Fibonacci generator σ.** The proof uses only that σ ∈ GL₂(ℤ); the Fibonacci structure (σ's specific eigenvalues φ, ψ) is not invoked. Are there finer invariants that DO distinguish subfamilies within each `L_j` under σ alone (before adjoining μ)? Yes — the σ-only orbits have length dividing π(p^k) (Pisano period), and the number of σ-orbits within `L_0` is `|L_0| / π(p^k)`. This is a richer structure that μ collapses.
