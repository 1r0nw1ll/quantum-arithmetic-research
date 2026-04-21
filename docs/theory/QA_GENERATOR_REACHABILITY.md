<!-- PRIMARY-SOURCE-EXEMPT: reason=empirical reachability analysis over canonical QA generators {σ, μ, ν} defined in tools/qa_viz/threejs/qa_core.js mirroring qa_orbit_rules.py; derived by direct enumeration in Node, not from external literature -->

# QA Generator Reachability on D₉²

**Scope**: empirical reachability of states in D₉² under various subsets of the canonical generator set `{σ, μ, ν}` defined in `tools/qa_viz/threejs/qa_core.js`. Motivated by a claim in the DRTH review session (2026-04-19) that **ν "collapses the torus to full connectivity"**. Verified against data.

**Generators (JS form):**
```js
sigma(b, e) = [e, qa_mod(b + e)]          // Fibonacci forward — the QA σ / qa_step
mu(b, e)    = [e, b]                       // swap
nu(b, e)    = [qa_mod(5·b), qa_mod(5·e)]   // componentwise ×5 = ×2⁻¹ mod 9 (halving)
```

## Reachability table

Reachability set = closure of `{seed}` under the listed generators. States counted until no new state appears.

| Seed    | Class       | `{σ}` | `{σ,μ}` | `{σ,ν}` | `{σ,μ,ν}` | `{μ,ν}` |
|---------|-------------|------:|--------:|--------:|----------:|--------:|
| (1,1)   | cosmos      |    24 |      72 |      72 |        72 |       6 |
| (1,2)   | cosmos      |    24 |      72 |      72 |        72 |      12 |
| (1,3)   | cosmos      |    24 |      72 |      72 |        72 |      12 |
| (1,4)   | cosmos      |    24 |      72 |      72 |        72 |      12 |
| (2,2)   | cosmos      |    24 |      72 |      72 |        72 |       6 |
| (3,3)   | satellite   |     8 |       8 |       8 |         8 |       2 |
| (9,9)   | singularity |     1 |       1 |       1 |         1 |       1 |

## Findings

1. **`{σ}` alone** gives 24 on Cosmos (one of the 3 period-24 orbits), 8 on Satellite (the full Satellite set is a single σ-orbit), 1 on Singularity. This is the canonical dynamics.

2. **Adding either μ or ν to σ bridges all 3 Cosmos orbits into one reachable set of 72.** Both `{σ,μ}` and `{σ,ν}` achieve full within-Cosmos connectivity. **Neither is uniquely the "bridge" — μ alone suffices.**

3. **`{σ,μ,ν}` does not reach more than `{σ,μ}`** — once you have either μ or ν, adding the other is redundant for reachability purposes on D₉². (ν may still have algebraic meaning distinct from μ, but not reachability-wise.)

4. **All three generators preserve the canonical 3-class partition.** Cosmos→Cosmos, Satellite→Satellite, Singularity fixed. "Full connectivity" from the DRTH-review discussion means *within-class*, not across. There is no generator in `{σ, μ, ν}` that bridges Cosmos to Satellite (to exit Cosmos you'd need to land on a state with `3∣b ∧ 3∣e`, which σ/μ/ν do not force).

5. **`{μ, ν}` without σ is weak**: 6–12 states from a cosmos seed (ν has period 6, μ is an involution, their joint closure is a dihedral-like small group). σ is essential.

## Upshot (on mod 9)

ChatGPT's framing "ν collapses torus → full connectivity" is accurate *in effect but not in uniqueness*: ν is one of two sufficient bridges alongside σ (μ is the other, and simpler). The correct minimal statement is:

> On mod 9: `{σ, μ}` or `{σ, ν}` → full within-class reachability on D₉².
> The 3-class partition (Cosmos / Satellite / Singularity) is invariant under the full generator group `⟨σ, μ, ν⟩`.

## Beyond mod 9: full structure theorem via p-adic valuation + CRT

Earlier drafts framed the mod-24 result as a counterexample to the mod-9 finding. **Retracted.** The right question is not "does mod 24 match mod 9?" but "what is the invariant that distinguishes `{σ,μ}`-components on any modulus?" The answer turns out to be clean: p-adic valuation stratification on each prime-power CRT factor.

### Proposition A — prime-power invariant

On `(Z/p^k Z)²` with `p` prime, the function
```
J(b, e) := min(v_p(b), v_p(e))      (taking values in {0, 1, ..., k})
```
is invariant under `σ(b,e) = (e, qa_mod(b+e))` and `μ(b,e) = (e, b)`.

**Proof.** μ swaps coordinates, so `J ∘ μ = J` trivially. For σ:
- If `v_p(b) ≠ v_p(e)`, say `v_p(b) = j < ℓ = v_p(e)`, then `v_p(b+e) = j`. So `J(σ(b,e)) = min(ℓ, j) = j = J(b,e)`.
- If `v_p(b) = v_p(e) = j`, write `b = p^j u, e = p^j v` with `u, v` units mod p. Then `b+e = p^j (u+v)`, so `v_p(b+e) ≥ j`. Hence `J(σ(b,e)) = min(j, v_p(b+e)) = j = J(b,e)`. ☐

### Proposition B — level sets are exactly the components

On `(Z/p^k Z)²` for `p ∈ {2, 3}`, the `k+1` level sets `L_j := {(b,e) : J(b,e) = j}` for `j ∈ {0, 1, ..., k}` are exactly the `{σ,μ}`-orbit components. **Empirically verified** on (p, k) ∈ {(2, 3), (3, 2), (3, 3)}. No closed-form proof of transitivity within a level yet — left as open.

**Closed-form level sizes**: For `j < k`, `|L_j| = p^{2(k−j−1)} · (p² − 1)`. For `j = k`, `|L_k| = 1`.

| (p, k) | level 0 | level 1 | level 2 | level 3 | sum = p^{2k} |
|---|---:|---:|---:|---:|---:|
| (2, 3) = mod 8  | 48  | 12  | 3   | 1 | 64 ✓ |
| (3, 2) = mod 9  | 72  | 8   | 1   | — | 81 ✓ |
| (3, 3) = mod 27 | 648 | 72  | 8   | 1 | 729 ✓ |

### Proposition C — CRT factorization for composite m

Let `m = Π p_i^{k_i}`. Under the CRT isomorphism
```
(Z/mZ)² ≅ Π (Z/p_i^{k_i} Z)²
```
σ and μ act componentwise (σ is additive in the ambient ring, μ permutes the ordered pair; both structures are preserved by CRT). Therefore the `{σ,μ}`-orbit decomposition on `(Z/mZ)²` is the **Cartesian product** of the per-factor level-set decompositions from Proposition B. The total component count is `Π (k_i + 1)`.

**Empirical verification** (all match exactly):

| m   | factoring | expected count | found count | found sizes (sorted) |
|---|---|---:|---:|---|
| 15  | 3·5       | 2·2 = 4 | 4  | 192, 24, 8, 1 |
| 24  | 8·3       | 4·2 = 8 | 8  | 384, 96, 48, 24, 12, 8, 3, 1 |
| 45  | 9·5       | 3·2 = 6 | 6  | 1728, 192, 72, 24, 8, 1 |
| 72  | 8·9       | 4·3 = 12 | 12 | 3456, 864, 384, 216, 96, 72, 48, 24, 12, 8, 3, 1 |

Each mod-m component factors as (level in factor 1) × (level in factor 2) × … with the size equal to the product of level sizes. The QA class partition inherits: `Cosmos / Satellite / Singularity` on each factor = `(level = 0) / (level = 1) / (level = k_i)`, and on composite m the classes combine via CRT.

### "Class shift" retracted

In an earlier δ_A (mod-9 reduction) fiber analysis, some mod-24 Cosmos components projected to mod-9 Satellite/Singularity. I called this "class shift" and asked whether it was signal or artifact. **Confirmed artifact**: δ_A discards the mod-8 factor of the CRT decomposition, collapsing states with different `(level mod 8, level mod 3)` pairs onto the same mod-9 level. The resulting class mismatch reflects what δ_A throws away, not a QA invariant.

### Semiconjugacy check (for the record)

| candidate δ : (Z/24Z)² → · | σ semiconjugacy | μ semiconjugacy |
|---|:---:|:---:|
| componentwise qa_mod_9 (δ_A) | 276/576 ✗ | 576/576 ✓ |
| componentwise qa_mod_3 (common quotient) | 576/576 ✓ | ✓ |

δ_A fails σ because it is not a ring-homomorphism from Z/24 to Z/9 (9 ∤ 24). Only the common mod-3 quotient semiconjugates, but it projects to a modulus where class structure is degenerate.

### ν is mod-9 specific (unchanged)

ν was `(b,e) → (qa_mod(5·b), qa_mod(5·e))` because `5 = 2⁻¹ (mod 9)`. On mod 24, `gcd(2, 24) = 2` so 2 has no inverse; "halving" does not transfer. A mod-24 analogue would require a different unit in `(Z/24Z)*`, with its own domain-specific interpretation.

## Corollary — QA class partition reinterpreted

The canonical Cosmos / Satellite / Singularity partition from `qa_orbit_rules.py` is the **prime-power valuation stratification** applied to each CRT factor of `m`, combined via Cartesian product for composite moduli. On a pure prime power `m = p^k`:
- Cosmos ≡ level 0 (at least one coordinate a unit)
- Satellite ≡ level 1 (both divisible by p, at least one not by p²)
- Singularity ≡ level k (both divisible by p^k — the single fixed state `(p^k, p^k)`)

On composite m the class partition is the componentwise combination. For m = 24 = 2³·3: level in mod-8 is 2-adic, level in mod-3 is 3-adic, and the canonical Satellite rule `(m/3)|b ∧ (m/3)|e = 8|b ∧ 8|e` picks out states at **level-3-in-mod-8** (paired with any mod-3 level) — a specific corner of the full stratification.

## Open questions

1. **Formal proof of Proposition B** (transitivity of `{σ,μ}` within each level). Empirically holds for (p,k) ∈ {(2,3), (3,2), (3,3)}; needs a closed-form argument (likely using Pisano periodicity of Fibonacci mod p^k).
2. **Primes other than 2 and 3** — verify on mod 25 = 5², mod 49 = 7², etc. The invariant and CRT structure should generalize, but empirical check would close the gap.
3. **Does ν have a universal analogue across moduli?** For odd m, `ν = ×(m+1)/2` is always defined (since `2·((m+1)/2) = m+1 ≡ 1 mod m`). For even m, 2 is not invertible and a different "scale" unit is needed.

## Reproduce

```bash
node --input-type=module -e "
import('/home/player2/signal_experiments/tools/qa_viz/threejs/qa_core.js').then(m => {
  const {sigma, mu, nu} = m;
  const reach = (seed, gens) => {
    const V = new Set([seed[0]*10+seed[1]]), F = [seed];
    while (F.length) {
      const [b,e] = F.pop();
      for (const g of gens) {
        const [nb,ne] = g(b,e), k = nb*10+ne;
        if (!V.has(k)) { V.add(k); F.push([nb,ne]); }
      }
    }
    return V.size;
  };
  console.log('{σ,μ} from (1,1):', reach([1,1], [sigma, mu]));
});
"
```
