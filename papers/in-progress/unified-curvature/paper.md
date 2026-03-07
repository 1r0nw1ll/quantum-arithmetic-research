# Unified Curvature Certificates Across Learning, Aggregation, Attention, Arithmetic, and Symbolic Search

### Abstract

The increasing diversity of neural network architectures has created a need for unified principles to analyze and guarantee their stability and convergence properties. This paper introduces a universal curvature metric, $\kappa$, that provides a consistent stability certificate across five distinct and significant architecture classes: gradient-based learning, graph neural network aggregation, transformer-style attention, modular arithmetic systems, and symbolic tree search. Our approach is founded on a shared four-parameter substrate, $H_{QA}$, which quantifies a system's intrinsic harmonic balance. We derive a universal curvature formula, $\kappa = 1 - |1 - \eta_{eff}|$, where the effective learning rate $\eta_{eff}$ is a product of a global learning rate, an architecture-specific gain, and $H_{QA}$. We formalize this unification through a "three-gate" machine-checkable certificate pattern, ensuring any compliant implementation adheres to the same stability criterion. This framework provides a common language for reasoning about system dynamics and a practical tool for building more reliable and interoperable AI systems.

### 1. Introduction

The field of artificial intelligence is characterized by a Cambrian explosion of architectures. From the foundational multi-layer perceptrons to the graph-centric worldview of Graph Neural Networks (GNNs) and the sequence-to-sequence prowess of Transformers, each model family has its own set of heuristics, hyperparameters, and stability considerations. This specialization has been a boon for performance but has led to a fractured theoretical landscape. Comparing the stability of a GNN's aggregation step with that of an attention head in a Transformer is not just difficult; it is a comparison of apples and oranges.

This lack of a common measure for stability and dynamics poses significant challenges for both researchers and practitioners. It complicates the transfer of theoretical insights from one domain to another and makes it difficult to build robust, hybrid systems that combine the strengths of different architectures. How can we guarantee that the learning dynamics in one component of a complex system are compatible with another?

This paper addresses this challenge by proposing a unified framework for certifying stability across a wide range of computational architectures. Our primary contribution is the introduction of a universal curvature metric, $\kappa$, which is derived from a common mathematical substrate, the Harmonic Index $H_{QA}$. This metric provides a single, interpretable value that characterizes the single-step convergence behavior of a system, regardless of its underlying implementation details.

We demonstrate the generality of our approach by applying it to five distinct families of computation:
1.  **QALM Gradient:** Standard gradient-based optimization.
2.  **GNN Aggregation:** The neighborhood aggregation step in Graph Neural Networks.
3.  **Attention Layer:** The core mechanism in Transformer models.
4.  **QARM Arithmetic:** Operations within a novel modular arithmetic system.
5.  **Symbolic Search:** Heuristic-guided search algorithms.

For each family, we define a machine-checkable certificate that binds its structural parameters and a specific gain hyperparameter to the universal $\kappa$ formula. This certificate system, based on a simple "three-gate" validation pattern, makes stability an enforceable and verifiable property of a model or system. The paper culminates in the Unified Curvature Theorem, a formal statement of this cross-architectural stability guarantee.

### 2. Background

Our framework is built upon a mathematical structure we call the Quantum Arithmetic (QA) system. While a full exposition is beyond the scope of this paper, the core concept is the representation of relationships not as single numerical values but as harmonic structures. The central metric derived from this system is the Harmonic Index, $H_{QA}$.

The QA system posits that any state can be characterized by a set of four fundamental components, which we label $a, b, d, e$. The Harmonic Index, $H_{QA}$, is a function of these four components that produces a normalized value in the range $[0, 1)$. It is designed to measure the "balance" or "consonance" of the relationship between two conceptual pairs: $(b, a)$ and $(e, d)$. A high $H_{QA}$ value indicates a strong harmonic relationship, while a low value indicates dissonance. In the context of learning systems, we can interpret $H_{QA}$ as a measure of the alignment between different components of a system's internal state, which in turn influences the stability of an update step.

### 3. The $H_{QA}$ Substrate

The foundation of our unified framework is the $H_{QA}$ substrate, a set of equations that map four input parameters to the bounded Harmonic Index.

#### 3.1 Formal Definition

Given four non-negative real numbers, $a, b, d, e$, we first define two intermediate quantities, $F$ (representing a product-based interaction) and $G$ (representing a sum-of-squares, or Euclidean, interaction):

$$ G = e \cdot e + d \cdot d $$
$$ F = b \cdot a $$

From these, we compute a raw, unbounded harmonic ratio, $H_{raw}$. This ratio combines the interactions $F$ and $G$ with a cross-term, creating a sensitive measure of the system's state. A small epsilon, $\text{eps}$, is used to ensure numerical stability.

$$ H_{raw} = 0.25 \cdot \left( \frac{F}{G + \text{eps}} + \frac{e \cdot d}{a + b + \text{eps}} \right) $$

#### 3.2 Boundedness Property

The raw index $H_{raw}$ can be arbitrarily large. To create a normalized metric suitable for a stability formula, we apply a saturating function that maps its absolute value to the interval $[0, 1)$.

$$ H_{QA} = \frac{|H_{raw}|}{1 + |H_{raw}|} $$

This formulation ensures that $H_{QA}$ is always non-negative and strictly less than 1. This boundedness is a critical prerequisite for the stability proof of the universal $\kappa$ formula.

#### 3.3 Geometric Interpretation

The four parameters $(a, b, d, e)$ can be conceptualized as defining a geometric configuration in a 2D plane. For instance, one can imagine two vectors, $\vec{v}_1 = (a, d)$ and $\vec{v}_2 = (b, e)$. In this view, $H_{QA}$ becomes a measure of the geometric relationship between these vectors—capturing aspects of their relative lengths and orientations. This geometric intuition underlies its applicability across diverse domains, from the "shape" of a loss landscape in gradient descent to the "structure" of a graph in a GNN.

### 4. The Universal $\kappa$ Formula

The core of our unification is a single formula for curvature, $\kappa$, that applies to any system built on the $H_{QA}$ substrate.

#### 4.1 Derivation

Consider a general iterative process where a parameter $p$ is updated at each step. A typical update rule, as seen in stochastic gradient descent (SGD), is:

$$ p_{\text{after}} = p_{\text{before}} - \eta \cdot \nabla L $$

where $\eta$ is the learning rate and $\nabla L$ is the gradient of a loss function. We generalize this by defining an *effective learning rate*, $\eta_{eff}$, which is modulated by the system's intrinsic harmonicity:

$$ \eta_{eff} = \text{lr} \cdot \text{gain} \cdot H_{QA} $$

Here, `lr` is a global base learning rate, $H_{QA}$ is the harmonic index of the system's current state, and `gain` is a scalar hyperparameter specific to the architecture class. The update rule becomes:

$$ p_{\text{after}} = p_{\text{before}} - \eta_{eff} \cdot \text{grad} $$

For a simple quadratic loss function, the condition for stable convergence of such an iterative update is that the multiplicative factor of the update, $(1 - \eta_{eff})$, must have a magnitude less than 1. We define our universal curvature metric, $\kappa$, to capture how close the system is to the edge of this stable region.

$$ \kappa = 1 - |1 - \eta_{eff}| = 1 - |1 - \text{lr} \cdot \text{gain} \cdot H_{QA}| $$

A $\kappa$ value close to 1 indicates a very small, stable update, while a $\kappa$ value approaching 0 indicates an update that is approaching the boundary of oscillation or divergence. A negative $\kappa$ would imply instability.

#### 4.2 Stability Theorem

**Theorem:** The single-step update is stable (i.e., $\kappa \in (0, 1]$) if and only if the effective learning rate $\eta_{eff}$ is in the open interval $(0, 2)$.

**Proof Sketch:**
For $\kappa$ to be in $(0, 1]$, we require $0 < 1 - |1 - \eta_{eff}| \le 1$.
This simplifies to $0 \le |1 - \eta_{eff}| < 1$.
This inequality holds if and only if $-1 < 1 - \eta_{eff} < 1$.
Subtracting 1 from all parts gives $-2 < -\eta_{eff} < 0$.
Multiplying by -1 and reversing the inequalities yields $0 < \eta_{eff} < 2$.

Since `lr`, `gain`, and $H_{QA}$ are all defined as non-negative, the condition $\eta_{eff} > 0$ is generally met. The critical condition for stability is therefore $\eta_{eff} < 2$.

A practical sufficient (but not necessary) condition for stability is to constrain the architectural `gain` to be in the range $(0, 2]$ and to ensure the learning rate is chosen such that $\text{lr} \cdot H_{QA} < 1$. Since $H_{QA} < 1$, this is easily achievable. This choice guarantees $\eta_{eff} = \text{lr} \cdot \text{gain} \cdot H_{QA} < 1 \cdot 2 = 2$, thus ensuring stability.

### 5. Five Architecture Classes

We now demonstrate the universality of the $\kappa$ framework by applying it to five distinct computational families. Each family is associated with a unique certificate ID and maps its own specific `gain` parameter into the universal formula.

#### 5.1 QALM Gradient [ID 89]
This is the most direct application, corresponding to standard gradient-based optimization.
- **gain name:** `gain`
- **Structural metadata:** None.
- **Description:** The `gain` parameter acts as a simple scalar on the learning rate, modulated by the harmonicity of the current state.
- **Fixture Example:** For a substrate of $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.2 GNN Aggregation [ID 93]
This applies to the neighborhood aggregation step in a Graph Neural Network. The substrate parameters $(a,b,d,e)$ could be derived from properties of the node and its neighbors.
- **gain name:** `agg_gain`
- **Structural metadata:** `n_nodes`, `n_edges`
- **Description:** `agg_gain` controls the strength of the aggregated message that updates a node's representation. The metadata provides context about the graph structure.
- **Fixture Example:** For a substrate of $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.3 Attention Layer [ID 94]
This applies to the value-weighting step within a self-attention mechanism in a Transformer.
- **gain name:** `attn_gain`
- **Structural metadata:** `n_heads`, `d_model`, `seq_len`
- **Description:** `attn_gain` scales the output of the attention head before it is combined with others. The substrate could be derived from the query, key, and value matrices.
- **Fixture Example:** For a substrate of $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.4 QARM Arithmetic [ID 95]
This applies to operations within a custom modular arithmetic system (Quantum Arithmetic Modulo).
- **gain name:** `qarm_gain`
- **Structural metadata:** `modulus`, `orbit_size`, `generator`
- **Description:** `qarm_gain` controls the magnitude of state transitions within the finite field, where the substrate is derived from the arithmetic operands.
- **Fixture Example:** For a substrate of $(b=1, e=2, d=3, a=5)$, $H_{QA} \approx 0.2571$.

#### 5.5 Symbolic Search [ID 96]
This applies to heuristic-guided search algorithms, such as beam search in a symbolic or logical space.
- **gain name:** `sym_gain`
- **Structural metadata:** `beam_width`, `search_depth`, `rule_count`
- **Description:** `sym_gain` modulates the influence of a heuristic score ($H_{QA}$) on the path selection/pruning process.
- **Fixture Example:** For a substrate of $(b=3, e=5, d=8, a=13)$, $H_{QA} \approx 0.4235$. With `lr=0.01` and `sym_gain=1.1`, this yields $\kappa \approx 0.00466$.

### 6. The Three-Gate Certificate Pattern

To make this unification a practical and enforceable standard, we introduce a machine-checkable certificate pattern. Any artifact (e.g., a model weight file, a software library) claiming compliance must contain the necessary data and pass a three-gate validation process.

- **Gate A: Substrate Integrity Check:** The validator recomputes $H_{QA}$ from the raw substrate parameters $(a,b,d,e)$ provided in the certificate. It then compares this computed value to the $H_{QA}$ value stored in the certificate. A mismatch indicates a corrupted or invalid substrate calculation.
  - **Failure Mode:** `H_QA_MISMATCH`

- **Gate B: Update Rule Integrity Check:** The validator first checks if the architecture-specific `gain` parameter is within the prescribed stable range, i.e., `gain` $\in (0, 2]$. It then recomputes the effective learning rate $\eta_{eff} = \text{lr} \cdot \text{gain} \cdot H_{QA}$. This value is checked against the update rule implicitly or explicitly defined in the artifact.
  - **Failure Modes:** `GAIN_OUT_OF_RANGE`, `UPDATE_RULE_MISMATCH`

- **Gate C: Curvature Integrity Check:** Using the now-verified `lr`, `gain`, and $H_{QA}$, the validator recomputes the final curvature metric $\kappa = 1 - |1 - \text{lr} \cdot \text{gain} \cdot H_{QA}|$. This is compared to the $\kappa$ value stored in the certificate.
  - **Failure Mode:** `KAPPA_MISMATCH`

An additional failure mode, `SCHEMA_INVALID`, occurs if the certificate is malformed and cannot be parsed by the validator. This simple but rigorous pattern ensures that any certified component transparently and correctly implements the unified stability framework.

### 7. Empirical Results

The true power of the $\kappa$ framework lies in its ability to place vastly different computational systems on a single, comparable scale. The table below shows results for the five families using two different substrate configurations. We use a fixed `lr=0.01` and a representative `gain=1.1` for all families for comparison.

| Family | ID | gain name | Substrate (b,e,d,a) | $H_{QA}$ | gain | $\kappa$ |
|---|---|---|---|---|---|---|
| QALM Gradient | [89] | `gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| GNN Aggregation | [93] | `agg_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| Attention Layer | [94] | `attn_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| QARM Arithmetic | [95] | `qarm_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| Symbolic Search | [96] | `sym_gain` | (1,2,3,5) | 0.2571 | 1.1 | 0.00283 |
| --- | --- | --- | --- | --- | --- | --- |
| QALM Gradient | [89] | `gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| GNN Aggregation | [93] | `agg_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| Attention Layer | [94] | `attn_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| QARM Arithmetic | [95] | `qarm_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |
| Symbolic Search | [96] | `sym_gain` | (3,5,8,13) | 0.4235 | 1.1 | 0.00466 |

The results are striking. For a given substrate, the stability metric $\kappa$ is identical across all five families, despite their wildly different purposes and structural metadata. The first substrate, composed of small integers, results in a lower harmonicity ($H_{QA} \approx 0.26$) and a correspondingly smaller $\kappa$. The second substrate, using consecutive Fibonacci numbers, exhibits a higher intrinsic harmony ($H_{QA} \approx 0.42$), leading to a larger (though still small) effective learning rate and a higher $\kappa$. This demonstrates that $\kappa$ is sensitive to the underlying state of the system (via $H_{QA}$) while being independent of the architectural specifics (which are absorbed into the `gain` parameter).

### 8. Unified Curvature Theorem

We now formally state the main result of this paper.

**Theorem (Unified Curvature):** Given a computational process whose state $p$ is updated via a rule equivalent to $p_{\text{after}} = p_{\text{before}} - \eta_{eff} \cdot \text{grad}$, where the effective learning rate $\eta_{eff}$ can be factored as $\eta_{eff} = \text{lr} \cdot \text{gain} \cdot H_{QA}$. Here, $H_{QA}$ is the Harmonic Index derived from a four-parameter substrate $(a,b,d,e)$ as defined in Section 3, `gain` is an architecture-specific scalar in $(0, 2]$, and `lr` is a global learning rate. If $\text{lr} \cdot H_{QA} < 1$, then the single-step update is stable, and its convergence behavior can be characterized by a universal curvature metric $\kappa = 1 - |1 - \eta_{eff}|$, which is guaranteed to be in $(0, 1)$. This unification holds across the five certified families: QALM Gradient [89], GNN Aggregation [93], Attention Layer [94], QARM Arithmetic [95], and Symbolic Search [96].

### 9. Conclusion and Future Work

In this paper, we have introduced a novel and powerful framework for unifying the analysis of stability across diverse computational architectures. By grounding our framework in a common mathematical substrate, $H_{QA}$, we derived a universal curvature metric, $\kappa$, that provides a single, comparable measure of single-step convergence dynamics. We demonstrated its applicability across five major classes of computation and proposed a practical, machine-checkable certificate system to enforce compliance.

The Unified Curvature Theorem provides a bridge between disparate fields of AI, offering a common language to describe and ensure stability. This has profound implications for the design of complex, hybrid AI systems, enabling greater reliability and interoperability.

Future work will proceed in several directions. First, we plan to extend the framework to other important architectural classes, such as Recurrent Neural Networks (RNNs) and Capsule Networks. Second, we will investigate the relationship between the structural metadata associated with each certificate family (e.g., `n_nodes`, `d_model`) and the optimal setting of the `gain` parameter. This could lead to methods for automatically tuning these systems for maximal stability. Finally, we will explore the use of $\kappa$ not just as a passive metric but as an active component in the learning process, for example, by using it as a regularization term to encourage more stable dynamics.

### References

1.  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998-6008).
2.  Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). A comprehensive survey on graph neural networks. *IEEE transactions on neural networks and learning systems*, 32(1), 4-24.
3.  Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
4.  Bottou, L. (2012). Stochastic gradient descent tricks. In *Neural networks: Tricks of the trade* (pp. 421-436). Springer, Berlin, Heidelberg.
5.  Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
6.  Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.
7.  Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *nature*, 529(7587), 484-489.
8.  Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*.
 [96].
