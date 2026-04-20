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

## Upshot

ChatGPT's framing "ν collapses torus → full connectivity" is accurate *in effect but not in uniqueness*: ν is one of two sufficient bridges alongside σ (μ is the other, and simpler). The correct minimal statement is:

> `{σ, μ}` or `{σ, ν}` → full within-class reachability on D₉².
> The 3-class partition (Cosmos / Satellite / Singularity) is invariant under the full generator group `⟨σ, μ, ν⟩`.

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
