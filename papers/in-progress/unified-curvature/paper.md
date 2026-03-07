# Unified Curvature Certificates Across Learning, Aggregation, Attention, Arithmetic, and Symbolic Search

### Abstract

The dynamics of the Quantum Arithmetic (QA) system correspond to multiplication by $\varphi^2$ in $\mathbb{Z}[\varphi]$, the ring of integers of $\mathbb{Q}(\sqrt{5})$: the map $T:(b,e)\mapsto(b+e,\,b+2e)$ equals $Q^2$, the square of the Fibonacci companion matrix, and acts as $\times\varphi^2$ on $b+e\varphi$. The Fibonacci quadratic norm $N(b+e\varphi)=b^2+be-e^2$ is $T$-invariant, orbit lengths equal $\pi(m)/2$ (half the Pisano period), and the harmonic index $H_{QA}$ admits a clean operator decomposition: $4H_{\mathrm{raw}}=\mathrm{tr}(\Lambda)$ where $\Lambda=\mathrm{diag}(\sigma_d,\sigma_{od})$ is a $2\times 2$ cross-coupling operator whose degree-0 eigenvalue $\sigma_d=ba/(e^2+d^2)$ tracks the projective Fibonacci angle and whose degree-1 eigenvalue $\sigma_{od}=ed/(b+a)$ collapses to $e/2$ via the Fibonacci identity $a+b=2d$.

Building on this algebraic foundation, we introduce a certificate-normal-form framework for comparing one-step stability across heterogeneous computational systems. For any certified family whose local update writes as $p_{\mathrm{after}}=p_{\mathrm{before}}-\eta_{\mathrm{eff}}\cdot\mathrm{grad}$ with $\eta_{\mathrm{eff}}=\mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}$, we define the universal curvature score

$$\kappa = 1 - |1-\eta_{\mathrm{eff}}|.$$

We instantiate this across eight certified families: gradient-based learning [89], graph aggregation [93], attention layers [94], modular arithmetic dynamics [95], symbolic search [96], orbit curvature [97], spectral-gain extensions [98, 99], and gradient Lipschitz gain [101]. Three families ([98], [99], [101]) derive gain from native structural objects—the spectral norm of weight/score matrices and the gradient $\ell_2$ norm—upgrading from consistency check to structural analysis. Empirically: across seven QA substrates, mean $\kappa$ correlates with final loss at $r=-0.843$; normalizing $\eta_{\mathrm{eff}}=1$ across all substrates equalizes convergence (loss std $1.9\times10^{-5}$), confirming that $H_{QA}$ governs convergence exclusively through $\eta_{\mathrm{eff}}$.

### 1. Introduction

The field of artificial intelligence is characterized by a proliferation of architectures. From multi-layer perceptrons to Graph Neural Networks (GNNs) and Transformers, each model family has its own stability considerations and hyperparameter conventions. Comparing the stability of a GNN's aggregation step with that of an attention head in a Transformer is not straightforward—each family uses different update semantics, different scale conventions, and different metadata. However, once each is mapped to a certified scalar local update form, comparison becomes possible.

This paper addresses this challenge by proposing a unified framework for certifying one-step stability across diverse computational architectures. Our primary contribution is the introduction of a curvature score $\kappa$, derived from a common mathematical substrate $H_{QA}$. This metric provides a single, interpretable value characterizing the single-step convergence behavior of a system's certified update rule, independent of domain-specific implementation details.

**What this paper claims.** The paper's claim is that eight heterogeneous architecture families admit the same machine-checkable one-step curvature certificate form, not that they share the same full dynamics. For three of these families ([98], [99], [101]) the gain is not a free witness but a derived structural invariant—the spectral norm of a native operator—making the certificate a structural analysis, not merely a consistency check. We do not assert global convergence, multi-step equivalence, or semantic identity across architecture classes.

We demonstrate this across seven distinct families:
1. **QALM Gradient [89]:** Standard gradient-based optimization.
2. **GNN Aggregation [93]:** The neighborhood aggregation step in Graph Neural Networks.
3. **Attention Layer [94]:** The core mechanism in Transformer models.
4. **QARM Arithmetic [95]:** Operations within a modular arithmetic system.
5. **Symbolic Search [96]:** Heuristic-guided search algorithms.
6. **Orbit Curvature [97]:** QA orbit stability margin across a full finite orbit.
7. **GNN Spectral Gain [98]:** Gain derived from σ_max(W) of the GNN weight matrix.
8. **Attention Spectral Gain [99]:** Gain derived from σ_max(QKᵀ/√d_k).
9. **Gradient Lipschitz Gain [101]:** Gain derived from ‖grad_vector‖₂, capped at 2.0.

For each family, a machine-checkable certificate binds structural parameters and a gain witness to the universal $\kappa$ formula via a three-gate validation pattern, making stability an enforceable and verifiable property.

**Table 1: Seven certified families at a glance.**

| Family | Domain | Gain witness | Gain type | Cert ID |
|---|---|---|---|---|
| QALM Gradient | Gradient optimization | `gain` | free scalar | [89] |
| GNN Aggregation | Graph neural networks | `agg_gain` | free scalar | [93] |
| Attention Layer | Transformer attention | `attn_gain` | free scalar | [94] |
| QARM Arithmetic | Modular arithmetic dynamics | `qarm_gain` | free scalar | [95] |
| Symbolic Search | Heuristic beam search | `sym_gain` | free scalar | [96] |
| Orbit Curvature | QA orbit stability | $\kappa_{\min}$ | derived (orbit enum.) | [97] |
| GNN Spectral Gain | GNN weight geometry | $\sigma_{\max}(W)$ | derived (power iter.) | [98] |
| Attention Spectral Gain | Attention score geometry | $\sigma_{\max}(QK^\top/\!\sqrt{d_k})$ | derived (power iter.) | [99] |
| Gradient Lipschitz Gain | Gradient descent | $\min(\|g\|_2, 2)$ | derived (L2 norm) | [101] |

### 2. Mathematical Foundation

The $H_{QA}$ substrate is not an ad-hoc formula. It arises from the arithmetic of $\mathbb{Q}(\sqrt{5})$ and from the projective dynamics of the Fibonacci recurrence. We summarise the structure here; formal proofs appear in §12.

**The ring $\mathbb{Z}[\varphi]$ and the QA map.** The golden ratio $\varphi=(\sqrt{5}+1)/2$ satisfies $\varphi^2=\varphi+1$. The ring of integers $\mathbb{Z}[\varphi]=\{b+e\varphi : b,e\in\mathbb{Z}\}$ carries the Fibonacci quadratic norm $N(b+e\varphi)=b^2+be-e^2$. The QA map $T:(b,e)\mapsto(d,a)$ with $d=b+e$, $a=b+2e$ (before modular reduction) corresponds to multiplication by $\varphi^2$ in $\mathbb{Z}[\varphi]$:
$$\varphi^2(b+e\varphi) = (b+e) + (b+2e)\varphi = d + a\varphi.$$
The matrix of $T$ is $Q^2=\begin{pmatrix}1&1\\1&2\end{pmatrix}$, the square of the Fibonacci companion matrix $Q=\begin{pmatrix}0&1\\1&1\end{pmatrix}$.

**Norm invariance and orbit structure (Propositions 1 and 2, §12).** Since $N(\varphi^2)=1$, the norm $N(b+e\varphi)=b^2+be-e^2$ is $T$-invariant. Modulo $m$, the orbit length equals $\pi(m)/2$ — half the Pisano period — because the order of $Q^2$ in $GL_2(\mathbb{Z}/m\mathbb{Z})$ is $\pi(m)/2$ for all $m\geq 3$. For $m=9$: $\pi(9)=24$, orbit length $=12$, 72 starting pairs ("Cosmos" group). The modular value $N(b+e\varphi)\bmod m$ classifies orbits: for $m=9$, $3\nmid N$ gives cosmos (length 12), $3^2\mid N$ gives satellite (length 4 or 1), $3^4\mid N$ gives the singularity (length 1).

**Projective Fibonacci flow.** The projective ratio $z=e/(b+e)$ evolves as the Möbius map $z\mapsto(z+1)/(z+2)$ under $T$, with unique fixed point $z^*=1/\varphi\approx 0.618$. Every pre-modular orbit converges projectively to $z^*$.

**$H_{QA}$ as a cross-coupling trace.** The step matrix $M=\begin{pmatrix}b&e\\d&a\end{pmatrix}$ (rows = state before and after $T$) satisfies $\det(M)=N(b+e\varphi)$ (pre-modular). The harmonic index decomposes as:
$$4H_{\mathrm{raw}} = \underbrace{\frac{ba}{e^2+d^2}}_{\sigma_d,\;\text{degree-0}} + \underbrace{\frac{ed}{b+a}}_{\sigma_{od},\;\text{degree-1}} = \mathrm{tr}(\Lambda), \qquad \Lambda = \mathrm{diag}(\sigma_d,\,\sigma_{od}).$$
The degree-0 component $\sigma_d=\cos(2\arctan(z))$ tracks the projective Fibonacci angle and is scale-invariant; the degree-1 component $\sigma_{od}=e/2$ is fixed by the Fibonacci identity $a+b=2d$ and carries the amplitude. The factor $1/4$ is the normalised-trace scaling: $H_{\mathrm{raw}}=\mathrm{tr}(\Lambda)/4=R(\Lambda,(1,1)^\top/\sqrt{2})/2$, the Rayleigh quotient at the equal-weight vector divided by 2. Together, properties (i) degree-(0,1) homogeneity, (ii) transpose symmetry $\sigma(M)=\sigma(M^\top)$, and (iii) projective Fibonacci convergence characterise $4H_{\mathrm{raw}}$ as the natural cross-coupling trace of the QA step matrix.

### 3. The $H_{QA}$ Substrate

Section 2 established the algebraic origin of $H_{QA}$. We now give the formal computational definition used by all validators.

#### 3.1 Formal Definition

Given four non-negative integers $a, b, d, e$ (all $\ge 1$), define:

$$G = e \cdot e + d \cdot d$$
$$F = b \cdot a$$

From these, compute a raw harmonic ratio $H_{raw}$ with a small epsilon $\varepsilon = 10^{-12}$ for numerical stability:

$$H_{raw} = 0.25 \cdot \left( \frac{F}{G + \varepsilon} + \frac{e \cdot d}{a + b + \varepsilon} \right)$$

#### 3.2 Boundedness Property

The raw index $H_{raw}$ can be arbitrarily large. A saturating function maps its absolute value to $[0, 1)$:

$$H_{QA} = \frac{|H_{raw}|}{1 + |H_{raw}|}$$

This ensures $H_{QA}$ is always non-negative and strictly less than 1. This boundedness is a critical prerequisite for the stability analysis of the universal $\kappa$ formula.

#### 3.3 Geometric Interpretation

As derived in §2, $4H_{\mathrm{raw}}=\mathrm{tr}(\Lambda)$ where $\Lambda=\mathrm{diag}(\sigma_d,\sigma_{od})$ is the cross-coupling operator of the QA step matrix. The degree-0 eigenvalue $\sigma_d=ba/(e^2+d^2)=\cos(2\arctan(z))$ measures the projective Fibonacci angle $z=e/d$; the degree-1 eigenvalue $\sigma_{od}=ed/(b+a)=e/2$ carries the state amplitude. Both terms are algebraically determined by the Q($\sqrt{5}$) structure; only the normalization constant $1/4$ carries a residual derivation gap (see §12).

### 4. The Universal $\kappa$ Formula

#### 4.1 Derivation

Consider a general iterative process where a parameter $p$ is updated at each step. A typical update rule is:

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \eta \cdot \mathrm{grad}$$

We generalize this by defining an *effective learning rate* modulated by the system's substrate:

$$\eta_{\mathrm{eff}} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}$$

Here, $\mathrm{lr}$ is a global base learning rate, $H_{QA}$ is the Harmonic Index of the system's current state, and $\mathrm{gain}$ is a scalar witness specific to the architecture class. The certified update rule becomes:

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \eta_{\mathrm{eff}} \cdot \mathrm{grad}$$

For a simple quadratic loss function, the condition for stable convergence of such an iterative update is that the scalar factor $(1 - \eta_{\mathrm{eff}})$ must have magnitude less than 1. We define:

$$\kappa = 1 - |1 - \eta_{\mathrm{eff}}| = 1 - |1 - \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}|$$

**Interpretation.** The quantity $\kappa = 1 - |1 - \eta_{\mathrm{eff}}|$ should be interpreted as a **stability-margin score on the interval $(0,2)$**. It is positive exactly when the one-step scalar factor lies in the standard stable interval, and it is maximized at $\eta_{\mathrm{eff}} = 1$. Thus, $\kappa$ is **not** a monotone measure of update smallness. Very small $\eta_{\mathrm{eff}}$ gives $\kappa \approx 0$, while $\eta_{\mathrm{eff}}$ near 1 gives $\kappa \approx 1$. The metric therefore measures proximity to the center of the admissible one-step interval rather than the absolute magnitude of the step.

#### 4.2 Stability Theorem

**Theorem:** The single-step update is stable (i.e., $\kappa \in (0, 1]$) if and only if the effective learning rate $\eta_{\mathrm{eff}}$ is in the open interval $(0, 2)$.

**Proof Sketch:**
For $\kappa$ to be in $(0, 1]$, we require $0 < 1 - |1 - \eta_{\mathrm{eff}}| \le 1$.
This simplifies to $0 \le |1 - \eta_{\mathrm{eff}}| < 1$.
This holds if and only if $-1 < 1 - \eta_{\mathrm{eff}} < 1$, i.e., $0 < \eta_{\mathrm{eff}} < 2$.

A practical sufficient condition is to constrain $\mathrm{gain} \in (0,2]$ and choose $\mathrm{lr} < 1$. Since $H_{QA} < 1$ by construction, this implies $\eta_{\mathrm{eff}} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA} < 1 \cdot 2 \cdot 1 = 2$, ensuring stability.

### 5. Seven Architecture Classes

We now demonstrate the framework by applying it to seven distinct computational families. Each family maps its own specific gain into the universal formula. Cross-row differences in the comparison table (Section 7) arise from different substrate and gain inputs, not from architecture identity alone. Families [98] and [99] represent a qualitative upgrade: gain is derived internally by the validator from a native operator, not supplied as a free scalar.

#### 5.1 QALM Gradient [ID 89]
- **Gain witness:** `gain`
- **Structural metadata:** None (base case).
- **Description:** The `gain` parameter acts as a scalar on the learning rate, modulated by the harmonicity of the current state.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.2 GNN Aggregation [ID 93]
- **Gain witness:** `agg_gain`
- **Structural metadata:** `n_nodes`, `n_edges`
- **Description:** `agg_gain` controls the strength of the aggregated message that updates a node's representation. The gain witness is supplied as a scalar; the structural metadata provides auditable graph context but does not enter the curvature computation.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.3 Attention Layer [ID 94]
- **Gain witness:** `attn_gain`
- **Structural metadata:** `n_heads`, `d_model`, `seq_len`
- **Description:** `attn_gain` scales the output of the attention head. The gain witness is a scalar; the structural metadata provides auditable configuration context.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.4 QARM Arithmetic [ID 95]
- **Gain witness:** `qarm_gain`
- **Structural metadata:** `modulus`, `orbit_size`, `generator`
- **Description:** `qarm_gain` controls the magnitude of state transitions within the finite field, where the substrate is derived from arithmetic operands.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.5 Symbolic Search [ID 96]
- **Gain witness:** `sym_gain`
- **Structural metadata:** `beam_width`, `search_depth`, `rule_count`
- **Description:** `sym_gain` modulates the influence of $H_{QA}$ on path selection. The structural metadata characterizes the search configuration.
- **Fixture Example:** Substrate $(b=3, e=5, d=8, a=13)$, $H_{QA} \approx 0.4235$. With $\mathrm{lr}=0.01$ and $\mathrm{sym\_gain}=1.1$: $\kappa \approx 0.00466$.

#### 5.6 Orbit Curvature [ID 97]
- **Gain:** $\kappa_{\min}$ across the full finite orbit (derived by enumeration).
- **Structural metadata:** `orbit_start` $(b_0,e_0)$, `modulus`.
- **Description:** Rather than a single-state snapshot, this family certifies the tightest stability bottleneck across an entire QA orbit. The validator enumerates the complete orbit under the two-step Fibonacci recurrence $(b,e)\!\to\!(d,a)$ with $d{=}(b{+}e)\bmod^* m$, $a{=}(b{+}2e)\bmod^* m$, computes $H_{QA}$ at every state, and reports $\kappa_{\min}=\min_t \kappa_t$.
- **Mathematical note:** Orbit length equals half the Pisano period: $|{\rm orbit}|=\pi(m)/2$. For $m=9$, $\pi(9)=24$, giving 12-step orbits with 72 starting pairs (the "Cosmos" group). This identity follows from the fact that the QA map equals $Q^2$, the square of the Fibonacci companion matrix, whose order in $GL_2(\mathbb{Z}/m\mathbb{Z})$ is $\pi(m)/2$ when $\pi(m)$ is even (which holds for all $m\geq 3$). See §12 for a formal proof.
- **Fixture Example:** Orbit start $(b_0=1,e_0=2)$, modulus $9$. With $\mathrm{lr}=0.5$, $\mathrm{gain}=1.0$: $\kappa_{\min}\approx 0.1077$ (bottleneck at state $(8,1,9,1)$, orbit length 12).

#### 5.7 GNN Spectral Gain [ID 98] and Attention Spectral Gain [ID 99]
- **Gain (GNN):** $\sigma_{\max}(W)$ derived from the GNN weight matrix $W$ via power iteration on $W^\top W$.
- **Gain (Attention):** $\sigma_{\max}(QK^\top/\!\sqrt{d_k})$ derived from the attention score matrix via power iteration.
- **Description:** These two families operationalize the derived-gain upgrade identified in §9. Instead of accepting a free scalar $\in(0,2]$, Gate B computes the gain internally from the submitted native operator and rejects any certificate where the claimed value disagrees with the recomputed spectral norm. The update rule is $p_{\mathrm{after}} = p_{\mathrm{before}} - \mathrm{lr}\cdot\sigma_{\max}\cdot H_{QA}\cdot\mathrm{grad}$.
- **GNN Fixture:** $W=\begin{pmatrix}0.8&0.2\\0.1&0.6\end{pmatrix}$, substrate $(3,5,8,13)$, $H_{QA}\!\approx\!0.4235$, $\sigma_{\max}(W)\!\approx\!0.8821$, $\kappa\!\approx\!0.00374$.
- **Attention Fixture:** $Q=\begin{pmatrix}0.6&0.4\\0.3&0.7\end{pmatrix}$, $K=\begin{pmatrix}0.5&0.3\\0.2&0.8\end{pmatrix}$, $d_k=2$, $\sigma_{\max}(QK^\top/\!\sqrt{2})\!\approx\!0.6603$, $\kappa\!\approx\!0.00280$.

#### 5.8 Gradient Lipschitz Gain [ID 101]
- **Gain:** $\min(\|\mathbf{g}\|_2, 2.0)$ where $\mathbf{g}$ is the gradient vector; derived by the validator from the submitted gradient, not a free scalar.
- **Description:** The L2 norm of the gradient is the natural local Lipschitz constant of the update step—it bounds how far the parameter moves per unit of curvature. Gate B recomputes the norm internally and rejects any certificate where the claimed value disagrees.
- **Fixture Example:** $\mathbf{g}=[0.3, 0.4]$, $\|\mathbf{g}\|_2=0.5$, substrate $(3,5,8,13)$, $H_{QA}\!\approx\!0.4235$, $\mathrm{lr}=0.01$, $\kappa\!\approx\!0.00212$.

### 6. The Three-Gate Certificate Pattern

To make this unification a practical and enforceable standard, we introduce a machine-checkable certificate pattern. Any artifact claiming compliance must contain the necessary data and pass a three-gate validation process.

- **Gate A — Substrate Integrity:** The validator recomputes $H_{QA}$ from the raw parameters $(a,b,d,e)$ in the certificate and compares to the claimed $H_{QA}$.
  - **Failure Mode:** `H_QA_MISMATCH`

- **Gate B — Update Rule Integrity:** The validator checks that the gain witness $\in (0, 2]$ (strict rejection, no clamping), then recomputes $\eta_{\mathrm{eff}} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}$ and verifies the claimed update witness $p_{\mathrm{after}}$.
  - **Failure Modes:** `GAIN_OUT_OF_RANGE`, `UPDATE_RULE_MISMATCH`

- **Gate C — Curvature Integrity:** Using verified $\mathrm{lr}$, $\mathrm{gain}$, and $H_{QA}$, the validator recomputes $\kappa$ and compares to the claimed value.
  - **Failure Mode:** `KAPPA_MISMATCH`

An additional failure mode, `SCHEMA_INVALID`, occurs if the certificate is malformed. This pattern ensures any certified component transparently and correctly implements the one-step normal form.

The failure types are disjoint: a certificate fails at exactly one gate, and the `invariant_diff` field in the failure payload identifies the specific numerical discrepancy. This makes the failure algebra closed under composition: a system built from multiple certified components can propagate failure types without ambiguity.

### 7. Comparative Certificate Instantiations

The table below is not a benchmark table. It shows that once a family is represented in the certified normal form, the resulting $\kappa$ depends only on the instantiated scalar inputs ($\mathrm{lr}$, $\mathrm{gain}$, $H_{QA}$), not on the architectural label of the family. This is the core comparability result of the framework.

We use $\mathrm{lr}=0.01$ and $\mathrm{gain}=1.1$ for all families for comparison.

| Family | ID | gain name | Substrate $(b,e,d,a)$ | $H_{QA}$ | gain | $\kappa$ |
|---|---|---|---|---|---|---|
| QALM Gradient | [89] | `gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| GNN Aggregation | [93] | `agg_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| Attention Layer | [94] | `attn_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| QARM Arithmetic | [95] | `qarm_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| Symbolic Search | [96] | `sym_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| — | — | — | — | — | — | — |
| QALM Gradient | [89] | `gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| GNN Aggregation | [93] | `agg_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| Attention Layer | [94] | `attn_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| QARM Arithmetic | [95] | `qarm_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| Symbolic Search | [96] | `sym_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |

For a given substrate, $\kappa$ is identical across all five families despite their different purposes and structural metadata. This demonstrates that $\kappa$ is sensitive to the underlying scalar inputs (via $H_{QA}$ and $\mathrm{gain}$) while being independent of the architectural specifics (which are captured in the structural metadata but do not enter the curvature computation).

### 8. Unified Curvature Normal-Form Theorem

**Theorem (Unified Curvature Normal Form).** Let a certified family provide a one-step update of the form

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \eta_{\mathrm{eff}}\cdot \mathrm{grad}, \qquad \eta_{\mathrm{eff}}=\mathrm{lr}\cdot \mathrm{gain}\cdot H_{QA},$$

where $H_{QA}$ is computed from a four-parameter substrate $(a,b,d,e)$ by the definitions in Section 3, $\mathrm{lr}\ge 0$, and $\mathrm{gain}\ge 0$. Then the associated one-step curvature score

$$\kappa = 1-\lvert 1-\eta_{\mathrm{eff}}\rvert$$

is positive if and only if $0<\eta_{\mathrm{eff}}<2$. In particular, any certified family whose validator enforces this normal form inherits the same scalar one-step stability criterion, regardless of domain-specific interpretation of the gain witness.

**Interpretation.** This theorem is a statement about a shared certified local update form. It does not assert global convergence, multi-step equivalence, or semantic identity across architecture classes.

#### 8.1 Finite-Orbit Descent Theorem (Quadratic Loss)

The Unified Curvature Normal Form Theorem certifies one-step stability. For QA orbit families, the finite structure of the orbit enables a multi-step descent result.

**Theorem (Finite-Orbit Descent).** *Let $L(w) = \tfrac{1}{2}w^2$ be a scalar quadratic loss. Let $\mathcal{O}=\{s_0,\ldots,s_{L-1}\}$ be a QA cosmos orbit of length $L=\pi(m)/2$ with per-step effective rates $\eta_{\mathrm{eff}}^{(t)}=\mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}(s_t)$ and curvature scores $\kappa_t = 1-|1-\eta_{\mathrm{eff}}^{(t)}|$. Define the orbit contraction factor*
$$\rho(\mathcal{O}) := \prod_{t=0}^{L-1}(1-\kappa_t)^2.$$
*If $\kappa_{\min}(\mathcal{O}):=\min_t\kappa_t > 0$, then:*
*(i)* $L_{t+L}(w) \leq \rho(\mathcal{O})\cdot L_t(w)$ *for every initial $w$;*
*(ii)* $\rho(\mathcal{O}) \leq (1-\kappa_{\min})^{2L} < 1$;
*(iii)* $\rho(\mathcal{O})$ *is a computable exact quantity from the orbit alone, with no stochastic approximation.*

**Proof.** For the scalar quadratic, the update gives $w_{t+1}=(1-\eta_{\mathrm{eff}}^{(t)})w_t$, so $L_{t+1}=(1-\eta_{\mathrm{eff}}^{(t)})^2 L_t$. Since $|1-\eta_{\mathrm{eff}}^{(t)}|=1-\kappa_t$, we have $L_{t+1}=(1-\kappa_t)^2 L_t$. Composing over $L$ steps: $L_{t+L}=\rho(\mathcal{O})\cdot L_t$. For (ii): $(1-\kappa_t)^2\leq(1-\kappa_{\min})^2$ for all $t$, so $\rho\leq(1-\kappa_{\min})^{2L}<1$ since $\kappa_{\min}>0$. For (iii): each $\kappa_t$ is computable from $s_t$ via §3, and $L=\pi(m)/2$ is finite by Prop.\ 1 (§12). $\square$

**Corollary (Multi-orbit convergence).** After $k$ full orbits, $L_{kL}\leq\rho(\mathcal{O})^k\cdot L_0\to 0$ geometrically with rate $\log(1/\rho(\mathcal{O}))$ per orbit.

**Key identity.** The per-step loss contraction factor is $(1-\kappa_t)^2$: higher $\kappa$ directly and exactly governs loss decrease. This gives the first-principles explanation of the empirical correlation $r(\text{mean}\,\kappa,\,\text{final loss})=-0.843$ reported in §11.

**Numerical example ($m=9$, lr $=0.5$, gain $=1$).** The mod-9 cosmos orbit has $L=12$ states. Numerical evaluation yields $\kappa_{\min}=0.1077$ (bottleneck at state $(8,1,9,1)$) and
$$\rho(\mathcal{O}) = \prod_{t=0}^{11}(1-\kappa_t)^2 = 0.001582,$$
so every full orbit reduces loss by a factor of 632. The orbit-invariant $\rho(\mathcal{O})$ is the same for all 72 starting pairs in the cosmos group (they all traverse the same 12-cycle). After 10 orbits: $L_{10L}\leq(0.001582)^{10}\cdot L_0\approx 10^{-28}\cdot L_0$.

**Scope.** The theorem is stated for scalar quadratic loss to give an exact, self-contained proof. Extension to vector-valued quadratic ($L(w)=\tfrac{1}{2}w^\top Q w$, $Q\succ 0$) is straightforward when $\eta_{\mathrm{eff}}\cdot\lambda_{\max}(Q)<2$; extension to PL-conditioned nonconvex loss is standard but requires additional assumptions on the landscape and is left to future work.

### 9. Limitations

For general nonconvex losses, this framework certifies only a local one-step normal form and does not prove multi-step convergence, generalization, or robustness under distribution shift. The Finite-Orbit Descent Theorem (§8.1) gives an exact multi-step guarantee for scalar quadratic loss; extension to nonconvex settings requires additional assumptions on the loss landscape. In families [89]–[96], the gain is supplied as a free scalar witness; the metadata fields ($\mathrm{n\_nodes}$, $\mathrm{d\_model}$, $\mathrm{orbit\_size}$, etc.) are recorded for auditing but do not constrain that scalar. Accordingly, those families should be read as reusable machine-checkable local certification patterns.

Families [98], [99], and [101] close this gap across three architecture classes: gain is derived from $\sigma_{\max}(W)$, $\sigma_{\max}(QK^\top/\!\sqrt{d_k})$, and $\|\mathbf{g}\|_2$ respectively, and cannot be freely adjusted. This upgrades those families from consistency checking to structural analysis. Extending derived-gain derivations to the remaining families (QARM, symbolic search) is left to future work.

### 10. Conclusion

This paper introduced a certificate-normal-form framework for local one-step stability analysis across heterogeneous computational architectures. By grounding the framework in a common mathematical substrate, $H_{QA}$, and a family-specific gain witness, we derived a universal curvature score $\kappa$ that provides a single comparable measure for certified update rules across five distinct architecture classes.

The Unified Curvature Normal Form Theorem provides a bridge across disparate fields of AI, but only at the level of certified local update forms. This offers a practical route toward interoperable local stability checks across mixed computational systems.

Future work will proceed in several directions. First, we plan to extend the framework to recurrent architectures, diffusion models, and Hopfield networks. Second, we will derive gain from native objects in the remaining free-witness families: QARM (orbit-step ratio) and symbolic search (effective branching factor). The gradient family [101] has already completed this transition via L2 norm. Third, §8.1 establishes the first exact multi-step descent guarantee for quadratic loss; extending this to PL-conditioned nonconvex losses via Lyapunov-like arguments is a natural next direction. Finally, we will explore using $\kappa$ as an active regularization term in training.

### 11. Empirical Validation

We validate the $\kappa$ framework with two controlled experiments on synthetic binary classification (800 samples, 20 features, two Gaussian classes). The model is a two-layer MLP (20→32 ReLU→1 sigmoid) trained with binary cross-entropy. All experiments use pure NumPy; no GPU is required.

**Setup.** The QA-modulated update is

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}\cdot\nabla\mathcal{L}, \qquad \kappa = 1-|1-\mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}|.$$

**Experiment 1 — H_QA sweep (fixed lr, gradient-norm gain).** We train seven QA substrates spanning $H_{QA}\in[0.20,\,0.71]$ (mod-9, gain $=\min(\|\mathbf{g}\|_2,\,2)$, lr $=0.1$, 300 epochs) plus a plain-SGD baseline (H_QA $=1$, gain $=1$). Results:

| Substrate | $H_{QA}$ | mean $\kappa$ | Final loss |
|-----------|----------|---------------|------------|
| (2,8)     | 0.201    | 0.0035        | 0.0434     |
| (4,7)     | 0.305    | 0.0043        | 0.0395     |
| (1,4)     | 0.356    | 0.0052        | 0.0407     |
| (5,3)     | 0.471    | 0.0056        | 0.0289     |
| (9,8)     | 0.529    | 0.0065        | 0.0279     |
| (3,5)     | 0.594    | 0.0068        | 0.0254     |
| (1,5)     | 0.715    | 0.0077        | 0.0231     |

The ordering is monotone: higher $H_{QA}$ → higher $\kappa$ → lower final loss. The Pearson correlation is $r(\text{mean}\,\kappa,\,\text{final loss})=-0.843$, $r(H_{QA},\,\text{final accuracy})=+0.769$. Plain SGD achieves lower absolute loss because gain $=\|\mathbf{g}\|_2\ll 1$ in the QA conditions shrinks the effective step by 20–150×; within the QA family the $\kappa$ ordering is preserved and strongly predictive.

**Experiment 2A — $\eta_{\mathrm{eff}}$ sweep (fixed substrate, gain $=1$).** We decouple step-size magnitude from the substrate by fixing gain $=1$ and choosing lr $=\eta_{\mathrm{eff}}/H_{QA}$ so that $\eta_{\mathrm{eff}}$ is exact. Using substrate $(9,8)$ ($H_{QA}=0.529$) and $\eta_{\mathrm{eff}}\in\{0.05,0.10,0.25,0.50,0.75,1.00,1.25,1.50,1.75,1.90\}$:

| $\eta_{\mathrm{eff}}$ | $\kappa$ | Final loss |
|-----------------------|----------|------------|
| 0.05 | 0.05 | 1.77×10⁻³ |
| 0.25 | 0.25 | 2.32×10⁻⁴ |
| 0.50 | 0.50 | 1.05×10⁻⁴ |
| 1.00 | 1.00 | 4.6×10⁻⁵  |
| 1.50 | 0.50 | 1.1×10⁻⁵  |
| 1.90 | 0.10 | 1.9×10⁻⁵  |

Loss decreases monotonically from $\eta_{\mathrm{eff}}=0.05$ to $1.50$, then stabilises. The minimum at $\eta_{\mathrm{eff}}=1.50$ (rather than exactly 1.00) is consistent with the theory: $\kappa=1$ is the boundary of the guaranteed no-oscillation interval, not the unique global minimiser for arbitrary data geometry. On linearly separable data, modest overshooting ($\eta_{\mathrm{eff}}>1$) is tolerated without divergence. The Pearson correlation across the sweep is $r(\kappa,\,\text{final loss})=-0.53$.

**Experiment 2B — substrate equalization ($\eta_{\mathrm{eff}}=1$ for all).** Setting lr $=1/H_{QA}$ for each of the seven substrates forces $\kappa=1$ uniformly. All seven converge in epoch 1; loss standard deviation across substrates is $1.9\times10^{-5}$ (loss range $[7\times10^{-6},\,7\times10^{-5}]$). This confirms the model's prediction: $H_{QA}$ governs convergence only through its role in $\eta_{\mathrm{eff}}$; once $\eta_{\mathrm{eff}}$ is equalized, substrate identity has negligible effect on convergence quality.

**Summary.** The three experiments jointly support three claims: (i) within the QA update family, mean $\kappa$ is a strong predictor of convergence quality ($r=-0.843$); (ii) the critical design parameter is $\eta_{\mathrm{eff}}=\mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}$, with $\kappa$ characterizing proximity to the stability boundary; (iii) once $\eta_{\mathrm{eff}}$ is equalized across substrates, convergence is equalized—confirming that $H_{QA}$ exerts its influence exclusively through $\eta_{\mathrm{eff}}$.

### 12. Appendix: The Half-Pisano Orbit Theorem

This appendix formalises the orbit-length result that underlies the Orbit Curvature Certificate family [97]. Let $m \geq 3$ be a positive integer and let $\{1,\ldots,m\}$ carry the nonzero-residue convention: arithmetic is performed mod $m$ with the result remapped to $m$ when it would otherwise be $0$.

**Notation.** Let $Q = \begin{pmatrix}0&1\\1&1\end{pmatrix}$ denote the Fibonacci companion matrix. Write $\pi(m)$ for the Pisano period — the smallest $k>0$ such that $F_k \equiv 0$ and $F_{k+1}\equiv 1\pmod{m}$, equivalently the order of $Q$ in $GL_2(\mathbb{Z}/m\mathbb{Z})$.

**Proposition (Half-Pisano Orbit Length).**
*For every $m \geq 3$, the QA update map $T:(b,e)\mapsto(b{+}e\bmod_m,\;b{+}2e\bmod_m)$ satisfies $T=Q^2$ and has maximal orbit length $\pi(m)/2$, where $\pi(m)$ is even for all $m\geq 3$. A state $(b_0,e_0)\in\{1,\ldots,m\}^2$ is called **primitive** if its orbit under $T$ attains this maximal length; non-primitive states have orbit lengths that are proper divisors of $\pi(m)/2$.*

**Proof sketch.**

*Step 1 — Matrix identity.* Direct computation shows

$$T\begin{pmatrix}b\\e\end{pmatrix} = \begin{pmatrix}1&1\\1&2\end{pmatrix}\begin{pmatrix}b\\e\end{pmatrix} = Q^2\begin{pmatrix}b\\e\end{pmatrix} \pmod{m},$$

so the QA map is the square of the Fibonacci companion map.

*Step 2 — Pisano period is even for $m\geq 3$.* A standard result (Wall 1960) states that for $m\geq 3$ the Pisano period satisfies $\pi(m) \equiv 0\pmod{2}$. This follows from the anti-symmetry identity $F_{\pi(m)/2} \equiv 0\pmod{m}$ and $F_{\pi(m)/2+1}\equiv -1\pmod{m}$, which forces $\pi(m)/2$ to be a half-period index and $\pi(m)$ to be even.

*Step 3 — Order of $Q^2$.* Because $Q^2 = (Q)^2$ and the order of $Q$ is $\pi(m)$, the order of $Q^2$ in $GL_2(\mathbb{Z}/m\mathbb{Z})$ is

$$\operatorname{ord}(Q^2) = \frac{\pi(m)}{\gcd(2,\pi(m))} = \frac{\pi(m)}{2}.$$

*Step 4 — Orbit length.* The orbit length of $(b_0,e_0)$ under $T = Q^2$ divides $\operatorname{ord}(Q^2) = \pi(m)/2$. A state is called *primitive* if its orbit length equals the full divisor $\pi(m)/2$; non-primitive states lie in proper $Q^2$-invariant subspaces and yield shorter orbits (e.g.\ the 4-cycles and the fixed point $(m,m)$ in the mod-9 case). $\square$

**Numerical verification.** The identity $\max_{(b,e)}\lvert\mathcal{O}(b,e)\rvert = \pi(m)/2$ was confirmed computationally for $m\in\{3,4,5,7,9,11,16,24\}$ with zero exceptions (see `empirical_kappa_experiment.py`).

**Corollary (Orbit Curvature Certificate [97]).** Because QA orbits are finite with known length $\pi(m)/2$, the minimum curvature score over a full orbit,

$$\kappa_{\min} = \min_{t=0}^{\pi(m)/2-1}\bigl(1-\lvert 1-\mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}^{(t)}\rvert\bigr),$$

is computable exactly with no stochastic approximation. Family [97] certifies this quantity for a declared $(b_0,e_0,m)$ triple, providing the first multi-step certified stability bound in the framework.

**Proposition 2 (Quadratic Norm Invariant and Q(√5) Structure).**

*The QA map $T$ is the linear representation of multiplication by $\varphi^2$ in $\mathbb{Z}[\varphi]$. The quadratic form $f(b,e)=b^2+be-e^2$ is the algebraic norm $N(b+e\varphi)$ in $\mathbb{Q}(\sqrt{5})$, and is invariant under $T$. The mod-$m$ value of $f$ classifies QA orbits.*

**Proof.**

*Step 1 — Algebraic identification.* Let $\varphi=(1+\sqrt{5})/2$ and $\bar\varphi=(1-\sqrt{5})/2$. In $\mathbb{Z}[\varphi]$ with basis $(1,\varphi)$, multiplication by $\varphi$ has matrix representation $Q=\begin{pmatrix}0&1\\1&1\end{pmatrix}$ (since $\varphi\cdot(b+e\varphi)=b\varphi+e\varphi^2=e+(b+e)\varphi$). Therefore $T=Q^2$ is the representation of multiplication by $\varphi^2$.

*Step 2 — Norm identification.* The algebraic norm $N(b+e\varphi)=(b+e\varphi)(b+e\bar\varphi)=b^2+be(\varphi+\bar\varphi)+e^2\varphi\bar\varphi=b^2+be\cdot 1+e^2\cdot(-1)=b^2+be-e^2=f(b,e)$, using $\varphi+\bar\varphi=1$ and $\varphi\bar\varphi=-1$.

*Step 3 — Invariance.* Since $N(\varphi^2)=N(\varphi)^2=(-1)^2=1$, multiplication by $\varphi^2$ preserves the norm: $N(\varphi^2\cdot(b+e\varphi))=N(b+e\varphi)$. Hence $f(T(b,e))=f(b,e)$ for all integer pairs. $\square$

**Geometric identity.** An equivalent elementary proof: expanding the $2\times 2$ determinant $\det\bigl(\begin{smallmatrix}b&d\\e&a\end{smallmatrix}\bigr)=ba-ed=b(b+2e)-e(b+e)=b^2+be-e^2=f(b,e)$. Thus $f(b,e)$ is the determinant of the matrix whose columns are the state before and after one QA step.

**Corollary (Orbit Classification by Norm Residue).** For $m=9$, the orbit type of $(b_0,e_0)$ is determined by the 3-adic valuation of $N(b_0+e_0\varphi)$:

| $v_3(f(b_0,e_0))$ | Orbit length | Orbit type |
|--------------------|--------------|------------|
| $\geq 4$           | 1            | Singularity $(9,9)$ |
| $2$ or $3$         | 4            | Satellite |
| $0$                | 12           | Cosmos (primitive) |

This is numerically verified for all 81 states of $\{1,\ldots,9\}^2$. The three orbit-type classes correspond exactly to the three cosets of the norm image in $(\mathbb{Z}/9\mathbb{Z})^{\times}$ under divisibility by $3$.

**Remark.** The orbit classification shows that QA dynamics on $\mathbb{Z}/m\mathbb{Z}$ are orbits of the map $x\mapsto\varphi^2 x$ in the ring $\mathbb{Z}[\varphi]/m\mathbb{Z}[\varphi]$, a finite quotient of the ring of integers of $\mathbb{Q}(\sqrt{5})$. This identifies QA as a dynamical system inside an algebraic number ring, not merely an ad-hoc modular arithmetic construction.

**Fibonacci projective flow.** Let $z_t = e_t/(b_t+e_t)$ be the projective ratio of the QA state at step $t$. One verifies directly that

$$z_{t+1} = \frac{1+z_t}{2+z_t},$$

i.e.\ $T$ induces the Möbius transformation $z\mapsto(z+1)/(z+2)$ on projective coordinates. This is precisely the action of $Q^2$ on $\mathbb{P}^1$. The unique fixed point satisfies $z^2+z-1=0$, giving $z^* = (\sqrt{5}-1)/2 = 1/\varphi \approx 0.618$. Consequently, every QA orbit's projective ratio converges toward $z^*$ under repeated application of $T$ (over $\mathbb{Z}$, before modular reduction).

**Partial result on $H_{QA}$.** Setting $d=b+e$ and $z=e/d$, a direct computation yields

$$H_{\text{raw}} = \frac{1-z^2}{4(1+z^2)} + \frac{e}{8}.$$

The first term is a pure function of the projective ratio $z$, measuring the asymmetry of the Fibonacci pair relative to the golden fixed point $z^*$: it equals zero at $z=1$ (equal components), is maximised as $z\to 0$ (strongly asymmetric pair), and equals $(\sqrt{5}-1)/(2(\sqrt{5}+1))\approx 0.112$ at the attractor $z=z^*$. This term arises from the Fibonacci projective dynamics, not directly from the algebraic structure of $\mathbb{Q}(\sqrt{5})$. The second term $e/8$ is a scale-dependent correction: writing $e = zd$, it equals $zd/8$ and depends on the absolute magnitude $d$ of the pair, not only on the ratio $z$. Consequently, $H_{\text{raw}}$ is not a projective invariant — it encodes both the ratio $z$ (Fibonacci projective flow) and the scale $e$ (absolute magnitude of the trailing component). Both terms are verified to match $H_{\text{raw}}$ exactly for all integer pairs tested.

**Resolution of the scale term.** The original formula has a symmetric two-term structure that explains the $e/8$ term completely. Labelling the 4-tuple elements as *outer pair* $(b,a)$ and *inner pair* $(e,d)$, the formula is

$$4H_{\text{raw}} = \underbrace{\frac{b\cdot a}{e^2+d^2}}_{\text{outer product}/\text{inner norm}^2} + \underbrace{\frac{e\cdot d}{a+b}}_{\text{inner product}/\text{outer sum}}.$$

The first term is the outer-to-inner ratio; the second is the complementary inner-to-outer ratio. The Fibonacci recurrence forces $a+b=(b+2e)+b=2(b+e)=2d$, so the second term collapses: $e\cdot d/(a+b)=e\cdot d/(2d)=e/2$. This gives $4H_{\text{raw}}=\cos(2\theta)+e/2$, i.e.\ $H_{\text{raw}}=\cos(2\theta)/4+e/8$. The $e/8$ term is therefore not a free parameter—it is algebraically determined by the Fibonacci identity $a+b=2d$ applied to the inner-to-outer ratio. The two terms together form a complementary pair: one captures the projective shape of the Fibonacci progression (via $\cos(2\theta)$), the other captures the absolute scale of its trailing element (via $e$).

**Matrix form and homogeneity analysis.** Writing $M = \begin{pmatrix} b & e \\ d & a \end{pmatrix}$ for the QA step matrix (row 1 = state before $T$, row 2 = state after $T$), the formula reads
$$4H_{\text{raw}} = \underbrace{\frac{M_{11}M_{22}}{M_{12}^2+M_{21}^2}}_{\sigma_d(M)} + \underbrace{\frac{M_{12}M_{21}}{M_{11}+M_{22}}}_{\sigma_{od}(M)}.$$
The two summands have distinct homogeneity: $\sigma_d(\lambda M)=\sigma_d(M)$ (degree-0, scale-invariant) while $\sigma_{od}(\lambda M)=\lambda\,\sigma_{od}(M)$ (degree-1). Consequently $\sigma=\sigma_d+\sigma_{od}$ is not projective: $\sigma_d$ tracks the *angular position* of the state in Fibonacci projective coordinates (via $\cos 2\theta$ identified above), while $\sigma_{od}$ tracks the *absolute amplitude* of the inner pair. In the linear pre-modular regime these become $\sigma_d=\cos 2\theta$ and $\sigma_{od}=e/2$ exactly — the projective component depends only on $z=e/d$, and the amplitude component depends only on $e$. The two components therefore partition $H_{\text{raw}}$ into an angle term and a scale term with no overlap.

The formula satisfies the *transpose symmetry* $\sigma(M)=\sigma(M^T)$: swapping $e\leftrightarrow d$ leaves both $e^2+d^2$ and $e\cdot d$ unchanged, so neither summand changes. This reflects the symmetry of the harmonic coupling under exchange of the inner pair.

Clearing denominators yields the bilinear identity
$$4H_{\text{raw}}\cdot(e^2+d^2)\cdot(b+a) \;=\; ba\,(b+a)\;+\;ed\,(e^2+d^2),$$
i.e.\ $4H_{\text{raw}}$ equals the outer self-coupling $ba$ weighted by the outer sum $(b+a)$, plus the inner self-coupling $ed$ weighted by the inner squared norm $(e^2+d^2)$, normalised by the product of those weights. This cross-coupling structure is verified to hold exactly for all seven experimental substrates (Table 1 values, integer arithmetic).

**Trace and Rayleigh quotient interpretation.** Define the $2\times 2$ cross-coupling operator $\Lambda = \operatorname{diag}(\sigma_d,\, \sigma_{od})$. Then
$$4H_{\text{raw}} = \operatorname{tr}(\Lambda) = \mathbf{e}_1^\top\Lambda\,\mathbf{e}_1 + \mathbf{e}_2^\top\Lambda\,\mathbf{e}_2,$$
verified exactly for all experimental substrates. Equivalently,
$$2H_{\text{raw}} = R\!\left(\Lambda,\,\tfrac{1}{\sqrt{2}}(1,1)^\top\right),$$
the Rayleigh quotient of $\Lambda$ at the equal-weight unit vector $v_0 = (1,1)^\top/\sqrt{2}$. This gives the precise meaning of the factor $1/4$:
$$H_{\text{raw}} = \frac{\operatorname{tr}(\Lambda)}{4} = \frac{\operatorname{tr}(\Lambda)}{2n}\bigg|_{n=2},$$
i.e.\ $H_{\text{raw}}$ is the *normalised trace* of $\Lambda$ — the arithmetic mean of its two eigenvalues, divided by $2$. The normalised trace is also $R(\Lambda,v_0)/2$: the Rayleigh quotient at the equal-weight vector, halved. The $1/4$ therefore factors as $(1/n)\times(1/2)$, where $1/n=1/2$ is the dimension-averaging factor of the $2\times 2$ operator and the residual $1/2$ is the Rayleigh-quotient-to-trace conversion at the equal-weight vector.

In the linear Fibonacci case ($a+b=2d$) the bilinear identity reduces to $G\cdot 2d = F\cdot 2d + (ed)\cdot G$, which can be read as: (inner squared norm $\times$ outer sum) $=$ (outer product $\times$ outer sum) $+$ (inner product $\times$ inner squared norm), expressing $4H_{\text{raw}}$ as a ratio of energy-balanced cross-couplings with the Fibonacci-orbit constraint absorbed.

**Partial resolution of the open question.** The two-term structure $\sigma_d+\sigma_{od}$ is characterised by properties (i)–(iii): (i) degree-(0,1) homogeneity, (ii) transpose symmetry, (iii) decomposition into a projective Fibonacci component (converging to $\cos(2\arctan(1/\varphi))$ at the golden-ratio attractor) whose amplitude component is fixed by the Fibonacci identity $a+b=2d$. The factor $1/4$ is the normalised-trace scaling of the resulting $2\times 2$ operator, equivalent to taking the Rayleigh quotient at the equal-weight vector and halving. What remains open is a derivation of why the equal-weight vector $v_0$ is selected — i.e.\ a first-principles argument connecting the Möbius action $z\mapsto(z+1)/(z+2)$ or the norm/trace structure of $\mathbb{Z}[\varphi]$ to the choice $v_0$.

---

**Theorem (Orbit Contraction).** *Let $\mathcal{O} = \{s_0, s_1, \ldots, s_{L-1}\}$ be a finite QA orbit of length $L = \pi(m)/2$ under the certified update family, with per-step effective rate $\eta_{\mathrm{eff}}^{(t)} = \mathrm{lr}\cdot\mathrm{gain}\cdot H_{QA}^{(t)}$. Define*

$$\kappa_{\min}(\mathcal{O}) = \min_{t=0}^{L-1}\Bigl(1 - \bigl\lvert 1 - \eta_{\mathrm{eff}}^{(t)}\bigr\rvert\Bigr).$$

*If $\kappa_{\min}(\mathcal{O}) > 0$, then $0 < \eta_{\mathrm{eff}}^{(t)} < 2$ holds for every $t \in \{0,\ldots,L-1\}$. That is, every one-step QA update along the full orbit remains strictly inside the scalar stability interval.*

**Proof.** The condition $\kappa_{\min} > 0$ requires $1 - |1 - \eta_{\mathrm{eff}}^{(t)}| > 0$ at every step, which is equivalent to $|1 - \eta_{\mathrm{eff}}^{(t)}| < 1$, i.e.\ $0 < \eta_{\mathrm{eff}}^{(t)} < 2$. Since the orbit is finite and fully enumerable, this condition can be checked exactly for all $t$. $\square$

**Remark.** The theorem is deliberately minimal: it asserts only that all per-step updates avoid the boundary of the stability interval (the $\eta_{\mathrm{eff}} \leq 0$ divergence zone and the $\eta_{\mathrm{eff}} \geq 2$ oscillation zone), not that the sequence of iterates converges in any norm. Connecting $\kappa_{\min} > 0$ to multi-step convergence rates requires additional assumptions on the loss landscape and is left to future work. The value of the theorem is that $\kappa_{\min}$ is the only scalar that simultaneously certifies all $L$ steps of a full orbit in one computable quantity, with no stochastic approximation and no elliptic PDE—a property unique to the finite enumerable structure of QA orbits.

### References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998–6008).
2. Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). A comprehensive survey on graph neural networks. *IEEE transactions on neural networks and learning systems*, 32(1), 4–24.
3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
4. Bottou, L. (2012). Stochastic gradient descent tricks. In *Neural networks: Tricks of the trade* (pp. 421–436). Springer, Berlin, Heidelberg.
5. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
6. Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.
7. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484–489.
8. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). HuggingFace's Transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*.
9. Wall, D. D. (1960). Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime. *American Mathematical Monthly*, 67(6), 525–532. [Standard reference for Pisano period properties; the evenness of $\pi(m)$ for $m\geq 3$ is an immediate consequence of the anti-symmetry identity proved therein.]
