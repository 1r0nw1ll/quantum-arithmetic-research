# QA Theorem: Locality Boundary as Variance-Bias Decomposition

**Status:** Lemma-level; empirically certified by Families [77]–[78]
**Date:** 2026-02-24

---

## 1. Setup

Let $x_i \in \mathbb{R}^d$ be the per-pixel spectrum at pixel $i$, and let $c_i \in \{1,\ldots,C\}$
be its class label.  A **locality generator** of radius $r$ maps each pixel to a summary statistic
over its $r$-neighborhood $\mathcal{N}_r(i)$:

$$
g_r(i) = \phi\!\left(\{x_j : j \in \mathcal{N}_r(i)\}\right)
$$

where $\phi$ computes mean, standard deviation, gradient statistics, etc.
A **spectral generator** maps $x_i \mapsto P^\top x_i$ where $P \in \mathbb{R}^{d \times k}$
is a PCA projection.

---

## 2. Error Decomposition

The classification error at radius $r$ decomposes as:

$$
\mathrm{err}(r) \approx \underbrace{V(r)}_{\text{within-class variance}} + \underbrace{B(r)}_{\text{boundary bias}}
$$

**Variance term** $V(r)$: Within a spatially homogeneous class region, the neighborhood average
reduces per-pixel noise.  Under the model $x_j = \mu_c + \varepsilon_j$ with i.i.d. noise:

$$
V(r) \propto \frac{\sigma^2_\varepsilon}{|\mathcal{N}_r(i)|} = \frac{\sigma^2_\varepsilon}{(2r+1)^2}
$$

$V(r)$ is **strictly decreasing** in $r$.

**Boundary bias** $B(r)$: When the neighborhood $\mathcal{N}_r(i)$ straddles a class boundary,
the summary statistic mixes class means:

$$
B(r) \propto \rho(r) \cdot \|\mu_c - \mu_{c'}\|^2
$$

where $\rho(r)$ is the **cross-class contamination rate** — the fraction of edges in $\mathcal{N}_r(i)$
whose endpoints belong to different classes.  $\rho(r)$ is related to the 4-neighbor adjacency
rate $\text{adj}_4$ measured by the Family [78] cert:

$$
\rho(r) \approx \text{adj}_4 \cdot r \qquad \text{(grows with } r \text{ and adjacency rate)}
$$

$B(r)$ is **increasing** in $r$ when $\text{adj}_4 > 0$.

---

## 3. Sign of $\Delta\mathrm{OA}(r)$

Define $\Delta\mathrm{OA}(r) = \mathrm{OA}(g_r) - \mathrm{OA}(g_\mathrm{spec})$.  The dominant regime and
failure regime are separated by the sign of the derivative of $\mathrm{err}(r)$:

$$
\frac{d\,\mathrm{err}(r)}{dr} \approx -\frac{d V(r)}{dr} + \frac{d B(r)}{dr}
$$

### Dominant Regime (certified by Family [77] `DOMINANT`)

$$
\frac{d V}{dr} > \frac{d B}{dr} \quad \Leftrightarrow \quad \text{adj}_4 \text{ is small relative to } \sigma^2_\varepsilon
$$

In this regime, variance reduction wins at all radii up to $r^*$, so
$\Delta\mathrm{OA}(r) > 0$ for $r \leq r^*$ and plateaus thereafter.

**Certificate condition:** $\text{patch}[r^*] > \text{spec}$ (Gate 3 of Family [77]).

### Failure Regime (certified by Family [78])

$$
\frac{d B}{dr} > \frac{d V}{dr} \quad \text{for all } r \leq r^* \quad \Leftrightarrow \quad \text{adj}_4 > \text{adj}_4^{\mathrm{crit}}(\sigma^2_\varepsilon)
$$

In this regime, boundary contamination dominates at every tested radius:
$\Delta\mathrm{OA}(r) \leq 0$ for all $r$.

**Certificate condition:** `all_deltas_nonpositive = true` (Gate 3 of Family [78]);
`fragmentation_scale_lt_r_star = true` (Gate 5); `adj_rate_4` verified from embedded
label grid (Gate 6).

---

## 4. Critical Adjacency Threshold

The crossover between dominant and failure regimes occurs at:

$$
\text{adj}_4^{\mathrm{crit}} \approx \frac{\sigma^2_\varepsilon}{r \cdot \|\mu_c - \mu_{c'}\|^2}
$$

Scenes with high inter-class contrast (large $\|\mu_c - \mu_{c'}\|$) or low noise
(small $\sigma^2_\varepsilon$) have low thresholds — they fail earlier as fragmentation
increases.  Scenes with high noise (MNIST-like cluttered imagery) tolerate more
fragmentation.

---

## 5. Bridge to Cert Families

| Quantity | Family [77] | Family [78] |
|----------|-------------|-------------|
| $\Delta\mathrm{OA}(r^*) > 0$ | Gate 3 `DOMINANT` | — |
| $\Delta\mathrm{OA}(r) \leq 0 \;\forall r$ | Gate 3 `FAILS_BOUNDARY_CONTAMINATION` | Gate 3 failure curve |
| $\text{adj}_4$ measured | boundary_metrics (proxy) | Gate 6 adjacency_witness (deterministic) |
| $\text{fragmentation\_scale} < r^*$ | boundary_metrics.fragmentation_proxy | Gate 5 fragmentation_explanation |

The adjacency witness in Family [78] v1.1 provides a **direct measurement** of $\text{adj}_4$,
converting the informal inequality above into a machine-verifiable gate.

---

## 6. Formal Claim (Conjectural)

**Lemma (Locality Boundary Condition):**
*For a scene with 4-neighbor cross-class adjacency rate $\text{adj}_4$,
the patch generator $g_r$ fails to dominate the spectral generator $g_\mathrm{spec}$
at all radii $r \leq r_{\max}$ if and only if
$\text{adj}_4 > \text{adj}_4^{\mathrm{crit}}(\sigma^2_\varepsilon, \Delta\mu)$.*

**Status:** Verified empirically on KSC (adj_4 ≈ 0.5 in proxy sense); formal proof
requires bounding the distribution of $\varepsilon$ and the geometry of class boundaries.
This is the target for a future Family [79] cert: `QA_LOCALITY_BOUNDARY_THEORY_CERT`.

---

## References

- Family [77]: `qa_neighborhood_sufficiency_cert_v1/` — DOMINANT outcome certification
- Family [78]: `qa_locality_boundary_cert_v1/` — boundary condition certification with adjacency witness
- Paper: `papers/in-progress/locality-dominance/locality_dominance_paper.tex`
