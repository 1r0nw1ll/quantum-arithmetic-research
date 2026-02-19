# QA Theorem: QARM Generator Interaction Curvature

**Theorem ID**: QA_THEOREM_QARM_GENERATOR_INTERACTION_CURVATURE.v1
**Status**: Active | **Date**: 2026-02-19
**Sibling theorems**:
- `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md` — gradient/CD-1 regime
- `docs/theory/QA_THEOREM_MONOIDAL_CURVATURE_FUNCTOR.md` — monoidal composition regime
**Authored by**: ChatGPT architectural review (2026-02-19)

---

## Notation
- Let \((\mathcal S, \mathcal F, \mathbb P)\) be a filtered probability space and \((\mathcal F_t)_{t\ge 0}\) the natural filtration of the run log.
- Let \(G\) be a **generator set** acting on states: each \(g\in G\) is a (partial) map \(g:\mathcal S \to \mathcal S\cup\{\bot\}\) where \(\bot\) denotes a failed move.
- A run is a (possibly stochastic) sequence \(g_t \in G\) and states \(s_{t+1} = g_t(s_t)\) when defined; otherwise \(s_{t+1}=s_t\) and the run logs a failure mode.

---

## Definitions (D1′–D6′)

**D1′. (Deviation object and norm)**
Fix a *target structure* \(\mathcal M \subseteq \mathcal S\) (e.g., a certified component, a fixed‑\(q\) leaf, or a manifold signature class).
Let \(\Delta:\mathcal S \to \mathbb R^m\) be a **deviation map** with deviation norm
\[
V(s)\;\stackrel{\mathrm{def}}{=}\;\|\Delta(s)\|^2,
\]
where \(\|\cdot\|\) is any norm for which Gate‑3 recomputation is deterministic (e.g., \(\ell_2\) over a fixed coordinate embedding or a canonical feature vector).

**D2′. (Generator‑driven evolution with split into restorative vs dispersive parts)**
Assume the run's generator at time \(t\) factors conceptually as
\[
g_t \equiv r_t \circ n_t,
\]
where:
- \(r_t\) is **restorative** relative to \(\mathcal M\) (attempts to decrease \(V\)),
- \(n_t\) is **dispersive** (injects variability / exploration / stochasticity),
and the certification log records enough to deterministically reconstruct (or upper‑bound) their combined effect on \(V\).

**D3′. (Basin‑local mean contraction / restorative dominance)**
There exists a **basin** \(\mathcal B \subseteq \mathcal S\) and constant \(\lambda_b>0\) such that for all \(s\in\mathcal B\),
\[
\mathbb E\!\left[V(s_{t+1}) \mid \mathcal F_t\right]
\;\le\;
\left(1-2\eta_t\lambda_b\right)\,V(s_t)\;+\;\eta_t^2\,\mathbb E\!\left[\|w_t\|^2\mid \mathcal F_t\right],
\]
where \(\eta_t\in(0,\eta_{\max}]\) is a logged step‑size (or step‑scale) and \(w_t\) is an \(\mathcal F_t\)-measurable noise proxy capturing the dispersive generator effect after the restorative correction.

> Interpretation: D3′ is the QARM analogue of restricted strong convexity: the *expected* deviation energy decreases linearly in‑basin, up to a dispersion term.

**D4′. (Dispersion bound / bounded conditional second moment)**
Assume \(w_t\) is conditionally mean‑zero and has bounded conditional energy:
\[
\mathbb E[w_t \mid \mathcal F_t]=0,
\qquad
\mathbb E[\|w_t\|^2 \mid \mathcal F_t]\;\le\;\sigma^2_{\mathrm{dev}}(t),
\]
where \(\sigma^2_{\mathrm{dev}}(t)\) is either (i) logged, or (ii) upper‑bounded by a logged quantity, or (iii) replaced by a closed‑form bound in a special family.

**D5′. (QARM generator interaction curvature)**
Define the **QARM interaction curvature**
\[
\kappa_{\mathrm{QARM}}(t)\;\stackrel{\mathrm{def}}{=}\;
\lambda_b - \tfrac12\,\eta_t\,\sigma^2_{\mathrm{dev}}(t).
\]
As in the gradient theorem, \(\kappa_{\mathrm{QARM}}(t)>0\) is a **noise‑dominance certificate**; it is not, by itself, the full stability condition (see D6′ / Theorem \(T_{\mathrm{QARM}}\)).

**D6′. (Basin smoothness / affine Lipschitz bound for the deviation update)**
There exists \(L_b\ge 0\) such that for all \(s\in\mathcal B\) and all admissible generators \(g\) used in the run,
\[
\|\Delta(g(s))-\Delta(g(s'))\|\;\le\;L_b\,\|\Delta(s)-\Delta(s')\|,
\quad \forall s,s'\in\mathcal B.
\]
This is a discrete‑time analogue of Lipschitz smoothness of the projected drift: within the basin, generator action does not amplify deviation differences faster than \(L_b\).

---

## Theorem \(T_{\mathrm{QARM}}\) (Basin‑Local Stability under Discrete Restorative Generators)

Assume D1′–D6′. Fix a constant step scale \(\eta_t\equiv \eta\) and a time‑uniform dispersion bound \(\sigma^2_{\mathrm{dev}}(t)\le\sigma^2_{\mathrm{dev}}\).
If
\[
0<\eta<\frac{2\lambda_b}{L_b^2},
\]
then for all \(t\ge 0\) while \(s_t\in\mathcal B\),
\[
\mathbb E[V(s_t)]
\;\le\;
(1-2\eta\lambda_b+\eta^2 L_b^2)^t\,V(s_0)
\;+\;
\frac{\eta\,\sigma^2_{\mathrm{dev}}}{2\,(2\lambda_b-\eta L_b^2)}\,
\Bigl(1-(1-2\eta\lambda_b+\eta^2 L_b^2)^t\Bigr).
\]

### Proof sketch (Gate‑friendly)
1. Start from D3′ and apply D4′ to upper‑bound the conditional dispersion term by \(\eta^2\sigma^2_{\mathrm{dev}}\).
2. Use D6′ to bound cross terms in the one‑step drift expansion, yielding the quadratic contraction factor
   \((1-2\eta\lambda_b+\eta^2 L_b^2)\).
3. Unroll the resulting linear recursion for \(\mathbb E[V(s_t)]\) (standard geometric series).

---

## Corollaries

**C1. (Noise floor inside the basin)**
Under \(T_{\mathrm{QARM}}\), as \(t\to\infty\),
\[
\limsup_{t\to\infty}\mathbb E[V(s_t)]
\;\le\;
\frac{\eta\,\sigma^2_{\mathrm{dev}}}{2\lambda_b-\eta L_b^2}.
\]

**C2. ("Escape destroys guarantees")**
The bounds in \(T_{\mathrm{QARM}}\) and C1 hold **only while \(s_t\in\mathcal B\)**.
If the run exits the basin (i.e., \(s_t\notin\mathcal B\) for some \(t\)), the constants \((\lambda_b,L_b,\sigma^2_{\mathrm{dev}})\) are no longer certified to apply and **no contraction/noise‑floor guarantee is implied**.

**C3. (Curvature as a certified bottleneck)**
If a run logs per‑step \(\eta_t\) and a certified upper bound \(\sigma^2_{\mathrm{dev}}(t)\), then
\[
\kappa_{\mathrm{QARM}}^{\min}\;\stackrel{\mathrm{def}}{=}\;\min_t \kappa_{\mathrm{QARM}}(t)
\]
is a Gate‑3 recomputable scalar anchor.
\(\kappa_{\mathrm{QARM}}^{\min}\le 0\) is a principled **obstruction signal**: dispersion dominates restorative drift at the certified scale.

---

## QA Certification Mapping (Gate 3/4 anchors)

### Required log fields (minimum viable)
- `deviation_norm_per_step[t]`: \( \|\Delta(s_t)\| \) or \(V(s_t)\)
- `eta_per_step[t]`: \(\eta_t\)
- `basin_id` or `basin_predicate`: how the run asserts \(s_t\in\mathcal B\) (can be conservative)
- either:
  - `sigma2_dev_per_step[t]` (preferred), or
  - `dispersion_upper_bound[t]` from which \(\sigma^2_{\mathrm{dev}}(t)\) is derived.

### Gate 3 rules
- Recompute `kappa_qarm_per_step[t] = λ_b - 0.5*eta_t*sigma2_dev_t`.
- Check:
  - hashes over the per‑step lists (tamper evidence),
  - min/argmin attestations (`min_kappa`, `min_kappa_step`),
  - optional max deviation spike attestations (`max_dev`, `max_dev_step`).
- Emit obstructions below.

### Gate 4 rules
- Deterministic replay (when applicable) must reproduce:
  - the deviation trace hash,
  - the curvature hash,
  - any declared basin predicate hash (if derived from replay).

---

## Minimal obstruction list (QARM sibling set)

| Obstruction code | Trigger | Meaning |
|---|---|---|
| `BASIN_ESCAPE` | certified predicate shows \(s_t\notin\mathcal B\) for some \(t\) | guarantees cease beyond escape |
| `DRIFT_CONTRACTION_VIOLATION` | empirical one‑step inequality in D3′ fails beyond tolerance | restorative dominance not observed/certified |
| `DISPERSION_BOUND_MISSING` | \(\sigma^2_{\mathrm{dev}}(t)\) absent and no derivation available | curvature cannot be certified |
| `NEGATIVE_GENERATOR_CURVATURE` | recomputed \(\kappa_{\mathrm{QARM}}(t)<0\) for some \(t\) | dispersion dominates |
| `CURVATURE_RECOMPUTE_MISMATCH` | curvature list/hash/min attestations mismatch recomputation | tamper or nondeterminism |
| `DEVIATION_TRACE_HASH_MISMATCH` | replay produces different deviation trace hash | nondeterminism / bad logging |
| `MOVE_FAIL_TYPE_MISMATCH` | logged failure algebra differs from replay / step logs | governance / trace integrity breach |

---

## Notes for fitting to existing QARM algebra
- If QARM moves are *purely deterministic* given the logged action, then `DISPERSION_BOUND_MISSING` can be avoided by defining \(\sigma^2_{\mathrm{dev}}(t)\equiv 0\) and treating randomness as occurring only through action selection (governed by a higher‑level policy cert).
- If dispersive action selection is stochastic, the policy layer should log sufficient statistics to upper‑bound dispersion in the induced deviation process.

---

## References

- `docs/QA_DYNAMICS_SPINE.md` — opt-in standard for Dynamics-Compatible families
- `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md` — gradient-regime sibling
- `docs/theory/QA_THEOREM_MONOIDAL_CURVATURE_FUNCTOR.md` — monoidal composition sibling
- `docs/theory/QA_CURVATURE_UNIFICATION_APPENDIX.md` — unified arXiv appendix merging all three
- ChatGPT architectural review (2026-02-19) — source of QARM theorem object
