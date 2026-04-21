<!-- PRIMARY-SOURCE-EXEMPT: reason=standalone theorem note extracted from empirical work documented in QA_GENERATOR_REACHABILITY.md; generators σ and μ defined in qa_orbit_rules.py and tools/qa_viz/threejs/qa_core.js -->

# QA Orbit Stratification Theorem

**Status**: Proposition A proven; Propositions B and C empirically verified on 6 prime powers + 4 composite moduli; closed-form proof of B open.
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

## Proposition B — level sets are exactly the components (empirical)

For `m = p^k` and `L_j := { (b,e) : J(b,e) = j }`, the `k + 1` sets `L_0, L_1, …, L_k` are exactly the orbit components of `⟨σ, μ⟩` on `(Z/mZ)²`.

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

**What's still open.** Proposition A shows `J` is an invariant, which gives the upper bound "at most `k + 1` components." The lower bound — transitivity of `{σ, μ}` within each `L_j` — is observed but not proven. A closed-form proof would likely go through Pisano periodicity of the Fibonacci recurrence mod `p^k`.

---

## Proposition C — CRT factorization

Let `m = ∏ p_i^{k_i}`. Under the CRT isomorphism
```
(Z/mZ)² ≅ ∏ (Z/p_i^{k_i} Z)²
```
both `σ` and `μ` act componentwise on the RHS:
- `σ` is additive in the ambient ring (`(b+e) mod p_i^{k_i}` agrees with `σ` on the i-th factor).
- `μ` permutes the ordered pair, independently of coordinate value.

Therefore the `⟨σ, μ⟩`-orbit decomposition on `(Z/mZ)²` is the **Cartesian product** of the per-factor Proposition-B decompositions. The total component count is `∏ (k_i + 1)`.

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

## Open questions

1. **Closed-form proof of Proposition B** — transitivity of `⟨σ, μ⟩` within each level `L_j`. Likely proof path: reduce to the Fibonacci recurrence mod `p^k` and invoke Pisano-period transitivity on `(Z/p^{k−j} Z)*`.
2. **Primes `p ≥ 11`** — the pattern held on `p ∈ {2, 3, 5, 7}`. Extending to larger primes is a mechanical check but worth doing once (mod 121 = 11², mod 169 = 13²).
3. **Higher `k`** — `p = 2` tested at `k = 3`; other primes tested at `k = 2, 3`. Does `p = 2, k = 4` (mod 16) still give 5 components? Expected yes.
4. **Generator ν** — not required for stratification; parked. Its potential role in proving Proposition B (as a scaling that ties `L_j` to `L_{j-1}` via explicit bijection) remains speculative.
