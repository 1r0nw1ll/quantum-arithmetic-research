# Unified Curvature Certificates Across Learning, Aggregation, Attention, Arithmetic, and Symbolic Search

### Abstract

We introduce a certificate-normal-form framework for comparing one-step stability behavior across heterogeneous computational systems. The framework is built around a shared scalar substrate, $H_{QA}$, computed from a four-parameter tuple $(a,b,d,e)$, and a family-specific gain witness. For any certified family whose local update can be written in the form

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \eta_{\mathrm{eff}}\cdot \mathrm{grad}, \qquad \eta_{\mathrm{eff}}=\mathrm{lr}\cdot \mathrm{gain}\cdot H_{QA},$$

we define a common curvature score

$$\kappa = 1 - \lvert 1-\eta_{\mathrm{eff}}\rvert.$$

We instantiate this template across five certified families: gradient-based learning [89], graph aggregation [93], attention layers [94], modular arithmetic dynamics [95], and symbolic search [96]. Our contribution is not a claim that these systems are globally equivalent, but that they admit a common machine-checkable one-step certification pattern. We formalize this pattern through a three-gate validation scheme—substrate recomputation, update-rule verification, and curvature verification—and show how it yields a single comparable scalar across multiple architecture classes. This provides a reproducible basis for cross-family local stability analysis and for future work connecting certified one-step curvature to richer multi-step behavior.

### 1. Introduction

The field of artificial intelligence is characterized by a proliferation of architectures. From multi-layer perceptrons to Graph Neural Networks (GNNs) and Transformers, each model family has its own stability considerations and hyperparameter conventions. Comparing the stability of a GNN's aggregation step with that of an attention head in a Transformer is not straightforward—each family uses different update semantics, different scale conventions, and different metadata. However, once each is mapped to a certified scalar local update form, comparison becomes possible.

This paper addresses this challenge by proposing a unified framework for certifying one-step stability across diverse computational architectures. Our primary contribution is the introduction of a curvature score $\kappa$, derived from a common mathematical substrate $H_{QA}$. This metric provides a single, interpretable value characterizing the single-step convergence behavior of a system's certified update rule, independent of domain-specific implementation details.

**What this paper claims.** The paper's claim is that five heterogeneous architecture families admit the same machine-checkable one-step curvature certificate form, not that they share the same full dynamics. We do not assert global convergence, multi-step equivalence, or semantic identity across architecture classes.

We demonstrate this across five distinct families:
1. **QALM Gradient [89]:** Standard gradient-based optimization.
2. **GNN Aggregation [93]:** The neighborhood aggregation step in Graph Neural Networks.
3. **Attention Layer [94]:** The core mechanism in Transformer models.
4. **QARM Arithmetic [95]:** Operations within a modular arithmetic system.
5. **Symbolic Search [96]:** Heuristic-guided search algorithms.

For each family, a machine-checkable certificate binds structural parameters and a gain witness to the universal $\kappa$ formula via a three-gate validation pattern, making stability an enforceable and verifiable property.

**Table 1: Five certified families at a glance.**

| Family | Domain | Gain witness | Structural metadata | Cert ID |
|---|---|---|---|---|
| QALM Gradient | Gradient optimization | `gain` | — | [89] |
| GNN Aggregation | Graph neural networks | `agg_gain` | `n_nodes`, `n_edges` | [93] |
| Attention Layer | Transformer attention | `attn_gain` | `n_heads`, `d_model`, `seq_len` | [94] |
| QARM Arithmetic | Modular arithmetic dynamics | `qarm_gain` | `modulus`, `orbit_size`, `generator` | [95] |
| Symbolic Search | Heuristic beam search | `sym_gain` | `beam_width`, `search_depth`, `rule_count` | [96] |

### 2. Background

Our framework is built upon a mathematical structure we call the Quantum Arithmetic (QA) system. The core concept is the representation of relationships as harmonic structures. The central metric derived from this system is the Harmonic Index, $H_{QA}$.

The QA system represents any state using four fundamental components labeled $a, b, d, e$. The Harmonic Index $H_{QA}$ is a function of these components producing a normalized value in $[0, 1)$. It measures the balance of the relationship between two conceptual pairs, $(b, a)$ and $(e, d)$. In the context of learning systems, $H_{QA}$ can be interpreted as a measure of the alignment between components of a system's internal state, which in turn influences the one-step stability of an update rule.

### 3. The $H_{QA}$ Substrate

The foundation of our framework is the $H_{QA}$ substrate, a set of equations that map four input parameters to the bounded Harmonic Index.

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

The parameters $(a, b, d, e)$ can be conceptualized as defining a geometric configuration. One can imagine two vectors $\vec{v}_1 = (a, d)$ and $\vec{v}_2 = (b, e)$, with $H_{QA}$ measuring aspects of their relative lengths and orientations. This interpretation is heuristic and is not used in the validator proofs; it motivates the choice of substrate structure but does not enter any formal argument.

### 4. The Universal $\kappa$ Formula

#### 4.1 Derivation

Consider a general iterative process where a parameter $p$ is updated at each step. A typical update rule is:

$$p_{\text{after}} = p_{\text{before}} - \eta \cdot \text{grad}$$

We generalize this by defining an *effective learning rate* modulated by the system's substrate:

$$\eta_{eff} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}$$

Here, $\mathrm{lr}$ is a global base learning rate, $H_{QA}$ is the Harmonic Index of the system's current state, and $\mathrm{gain}$ is a scalar witness specific to the architecture class. The certified update rule becomes:

$$p_{\text{after}} = p_{\text{before}} - \eta_{eff} \cdot \text{grad}$$

For a simple quadratic loss function, the condition for stable convergence of such an iterative update is that the scalar factor $(1 - \eta_{eff})$ must have magnitude less than 1. We define:

$$\kappa = 1 - |1 - \eta_{eff}| = 1 - |1 - \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}|$$

**Interpretation.** The quantity $\kappa = 1 - |1 - \eta_{eff}|$ should be interpreted as a **stability-margin score on the interval $(0,2)$**. It is positive exactly when the one-step scalar factor lies in the standard stable interval, and it is maximized at $\eta_{eff} = 1$. Thus, $\kappa$ is **not** a monotone measure of update smallness. Very small $\eta_{eff}$ gives $\kappa \approx 0$, while $\eta_{eff}$ near 1 gives $\kappa \approx 1$. The metric therefore measures proximity to the center of the admissible one-step interval rather than the absolute magnitude of the step.

#### 4.2 Stability Theorem

**Theorem:** The single-step update is stable (i.e., $\kappa \in (0, 1]$) if and only if the effective learning rate $\eta_{eff}$ is in the open interval $(0, 2)$.

**Proof Sketch:**
For $\kappa$ to be in $(0, 1]$, we require $0 < 1 - |1 - \eta_{eff}| \le 1$.
This simplifies to $0 \le |1 - \eta_{eff}| < 1$.
This holds if and only if $-1 < 1 - \eta_{eff} < 1$, i.e., $0 < \eta_{eff} < 2$.

A practical sufficient condition is to constrain $\mathrm{gain} \in (0,2]$ and choose $\mathrm{lr} < 1$. Since $H_{QA} < 1$ by construction, this implies $\eta_{\mathrm{eff}} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA} < 1 \cdot 2 \cdot 1 = 2$, ensuring stability.

### 5. Five Architecture Classes

We now demonstrate the framework by applying it to five distinct computational families. Each family maps its own specific gain witness into the universal formula. Cross-row differences in the comparison table (Section 7) arise from different substrate and gain inputs, not from architecture identity alone.

#### 5.1 QALM Gradient [ID 89]
- **gain name:** `gain`
- **Structural metadata:** None (base case).
- **Description:** The `gain` parameter acts as a scalar on the learning rate, modulated by the harmonicity of the current state.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.2 GNN Aggregation [ID 93]
- **gain name:** `agg_gain`
- **Structural metadata:** `n_nodes`, `n_edges`
- **Description:** `agg_gain` controls the strength of the aggregated message that updates a node's representation. The gain witness is supplied as a scalar; the structural metadata provides auditable graph context but does not enter the curvature computation.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.3 Attention Layer [ID 94]
- **gain name:** `attn_gain`
- **Structural metadata:** `n_heads`, `d_model`, `seq_len`
- **Description:** `attn_gain` scales the output of the attention head. The gain witness is a scalar; the structural metadata provides auditable configuration context.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.4 QARM Arithmetic [ID 95]
- **gain name:** `qarm_gain`
- **Structural metadata:** `modulus`, `orbit_size`, `generator`
- **Description:** `qarm_gain` controls the magnitude of state transitions within the finite field, where the substrate is derived from arithmetic operands.
- **Fixture Example:** Substrate $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.5 Symbolic Search [ID 96]
- **gain name:** `sym_gain`
- **Structural metadata:** `beam_width`, `search_depth`, `rule_count`
- **Description:** `sym_gain` modulates the influence of $H_{QA}$ on path selection. The structural metadata characterizes the search configuration.
- **Fixture Example:** Substrate $(b=3, e=5, d=8, a=13)$, $H_{QA} \approx 0.4235$. With $\mathrm{lr}=0.01$ and $\mathrm{sym\_gain}=1.1$: $\kappa \approx 0.00466$.

### 6. The Three-Gate Certificate Pattern

To make this unification a practical and enforceable standard, we introduce a machine-checkable certificate pattern. Any artifact claiming compliance must contain the necessary data and pass a three-gate validation process.

- **Gate A — Substrate Integrity:** The validator recomputes $H_{QA}$ from the raw parameters $(a,b,d,e)$ in the certificate and compares to the claimed $H_{QA}$.
  - **Failure Mode:** `H_QA_MISMATCH`

- **Gate B — Update Rule Integrity:** The validator checks that `gain` $\in (0, 2]$ (strict rejection, no clamping), then recomputes $\eta_{eff} = \mathrm{lr} \cdot \mathrm{gain} \cdot H_{QA}$ and verifies the claimed update witness $p_{\text{after}}$.
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

### 8. Unified Curvature Theorem

**Theorem (Unified Curvature Normal Form).** Let a certified family provide a one-step update of the form

$$p_{\mathrm{after}} = p_{\mathrm{before}} - \eta_{\mathrm{eff}}\cdot \mathrm{grad}, \qquad \eta_{\mathrm{eff}}=\mathrm{lr}\cdot \mathrm{gain}\cdot H_{QA},$$

where $H_{QA}$ is computed from a four-parameter substrate $(a,b,d,e)$ by the definitions in Section 3, $\mathrm{lr}\ge 0$, and $\mathrm{gain}\ge 0$. Then the associated one-step curvature score

$$\kappa = 1-\lvert 1-\eta_{\mathrm{eff}}\rvert$$

is positive if and only if $0<\eta_{\mathrm{eff}}<2$. In particular, any certified family whose validator enforces this normal form inherits the same scalar one-step stability criterion, regardless of domain-specific interpretation of the gain witness.

**Interpretation.** This theorem is a statement about a shared certified local update form. It does not assert global convergence, multi-step equivalence, or semantic identity across architecture classes.

### 9. Limitations

This framework currently certifies only a local one-step normal form. It does not, by itself, prove multi-step convergence, generalization, or robustness under distribution shift. In several families, the gain term is supplied as a scalar witness rather than recomputed from the full native object (for example, from an actual graph operator, attention tensor, arithmetic orbit structure, or symbolic search tree). Accordingly, the present result should be read as a reusable machine-checkable local certification pattern, not yet as a full dynamical theory of heterogeneous architectures.

A further limitation is that the structural metadata fields ($\mathrm{n\_nodes}$, $\mathrm{d\_model}$, $\mathrm{orbit\_size}$, etc.) are recorded in the certificate for auditing and provenance but do not presently constrain the gain witness. Future work should derive the gain witness from these richer native objects rather than accepting an arbitrary scalar.

### 10. Conclusion

This paper introduced a certificate-normal-form framework for local one-step stability analysis across heterogeneous computational architectures. By grounding the framework in a common mathematical substrate, $H_{QA}$, and a family-specific gain witness, we derived a universal curvature score $\kappa$ that provides a single comparable measure for certified update rules across five distinct architecture classes.

The Unified Curvature Normal Form Theorem provides a bridge across disparate fields of AI, but only at the level of certified local update forms. This offers a practical route toward interoperable local stability checks across mixed computational systems.

Future work will proceed in several directions. First, we plan to extend the framework to recurrent architectures, diffusion models, and Hopfield networks. Second, we will investigate deriving the gain witness from richer native objects (full graph operators, attention tensors, orbit structures) rather than scalar witnesses, which would strengthen the claim from a local certification pattern to a fuller dynamical result. Finally, we will explore using $\kappa$ as an active regularization term in training, and connecting certified one-step curvature to multi-step convergence guarantees via Lyapunov-like arguments.

### References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998–6008).
2. Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). A comprehensive survey on graph neural networks. *IEEE transactions on neural networks and learning systems*, 32(1), 4–24.
3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
4. Bottou, L. (2012). Stochastic gradient descent tricks. In *Neural networks: Tricks of the trade* (pp. 421–436). Springer, Berlin, Heidelberg.
5. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
6. Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.
7. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484–489.
8. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). HuggingFace's Transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*.
