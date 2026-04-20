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

## On mod 24: CRT factorization, not a refutation

Earlier revisions of this doc framed the mod-24 result as "theorem doesn't generalize." **Retracted.** That framing was methodologically wrong: mod-24 component counts are not comparable to mod-9 component counts without a specified morphism, and when we looked for one, the structure turned out to be CRT.

### Semiconjugacy check (why `δ_A = qa_mod_9` isn't the right demodulation)

| candidate δ : D_24² → · | σ semiconjugacy | μ semiconjugacy |
|---|:---:|:---:|
| componentwise qa_mod_9 (δ_A) | 276/576 ✗ | 576/576 ✓ |
| componentwise qa_mod_3 (δ_B, common quotient) | 576/576 ✓ | ✓ |

`δ_A` fails on σ because it discards the mod-8 half of the CRT decomposition. Only `δ_B` (reducing all the way down to the common mod-3 quotient) semiconjugates both generators — but mod 3 has degenerate class structure (sat_div = 1 makes every non-(3,3) state formally "Satellite"), so it says nothing useful about Cosmos connectivity.

### The real structure: CRT factorization

**Proposition (CRT factorization of `{σ, μ}` orbits — empirically verified for m ∈ {9, 24}).**

Let `m = a · b` with `gcd(a, b) = 1`. Under the CRT isomorphism
`(Z/mZ)² ≅ (Z/aZ)² × (Z/bZ)²`,
the generators `σ(b,e) = (e, qa_mod(b+e))` and `μ(b,e) = (e, b)` are **coordinate-wise in the state pair**, and `σ` is **additive in the ambient ring**. Both act componentwise under CRT. Therefore the `⟨σ, μ⟩`-orbit decomposition on `(Z/mZ)²` is the Cartesian product of the decompositions on `(Z/aZ)²` and `(Z/bZ)²`.

**Induced on Cosmos**: the product restricts to the Cosmos sub-fibers of each factor (since class membership is also componentwise under CRT).

### Empirical verification table (m = 24 = 8 · 3)

Direct enumeration: `{σ,μ}` on (Z/24Z)² yields **8 components**, each of which factors exactly as (mod-3 component) × (mod-8 component). No exceptions.

| mod-24 comp | size | (mod-3 comp, mod-8 comp) | factored size | class on mod 24 |
|---:|---:|:---:|:---|:---|
| 0 | 384 | (c3=0, c8=0) | 8 × 48 | Cosmos |
| 1 |  96 | (c3=0, c8=1) | 8 × 12 | Cosmos |
| 2 |  48 | (c3=1, c8=0) | 1 × 48 | Cosmos |
| 3 |  24 | (c3=0, c8=2) | 8 × 3  | Cosmos |
| 4 |  12 | (c3=1, c8=1) | 1 × 12 | Cosmos |
| 5 |   8 | (c3=0, c8=3) | 8 × 1  | Satellite |
| 6 |   3 | (c3=1, c8=2) | 1 × 3  | Cosmos |
| 7 |   1 | (c3=1, c8=3) | 1 × 1  | Singularity |
| **sum** | **576** = 24² ✓ | | | |

mod 3 has 2 `{σ,μ}`-components (size 8, size 1). mod 8 has 4 (sizes 48, 12, 3, 1). Product of counts: 2 × 4 = 8, matching the mod-24 count. Product of sizes pairwise: matches the mod-24 sizes exactly.

### Contrast: why mod 9 has a single Cosmos component

- **m = 9 = 3²** — a single prime power, so CRT gives no non-trivial factorization. `{σ,μ}` on (Z/9Z)² decomposes into the 2 components measured earlier: one size-72 Cosmos + one size-8 Satellite + singleton (9,9), giving 3 classes total (not components in the CRT sense, but the full decomposition).
- **m = 24 = 8 · 3** — CRT splits. `{σ,μ}` on (Z/24Z)² gives 8 components = 2 (mod-3 factor) × 4 (mod-8 factor).
- **No contradiction**: the mod-9 result does not "fail to generalize"; it is the m = 9 instance of the same CRT proposition. 24 and 9 are simply moduli with different prime-power factorizations.

### "Class shift" retracted

In the δ_A fiber analysis earlier, some mod-24 Cosmos components projected to mod-9 Satellite/Singularity. The previous draft called this "class shift" and asked whether it was signal or artifact. **Confirmed artifact**: δ_A discards mod-8 information, collapsing arithmetically distinct mod-24 states into the same mod-9 residue. The resulting class mismatch reflects what δ_A throws away, not a QA invariant.

### ν is mod-9 specific (unchanged)

ν was `(b,e) → (qa_mod(5·b), qa_mod(5·e))` because `5 = 2⁻¹ (mod 9)`. On mod 24, `gcd(2, 24) = 2`, so 2 has no inverse — the "halving" interpretation does not transfer. Any mod-24 analogue of ν would require choosing a different unit in `(Z/24Z)*`. Not measured.

## Open questions

1. **Prove the CRT proposition in closed form.** The empirical verification on m ∈ {9, 24} is suggestive; the proof is straightforward (CRT iso is a ring isomorphism, σ and μ are ring-polynomial maps in the state coordinates) but should be written out.
2. **Confirm on further moduli** — e.g., m = 45 = 9 · 5, m = 72 = 8 · 9, m = 15 = 3 · 5. Each should give components = (mod-a decomp) × (mod-b decomp).
3. **Does ν have a universal definition across all m?** Requires a choice of unit in `(Z/mZ)*` whose interpretation is domain-meaningful. Open.

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
