# 5. Discussion

## 5.1 Why Local Neighborhood Statistics Outperform Spectral Transforms

The central empirical finding across all three datasets is that low-dimensional neighborhood statistics outperform substantially higher-dimensional per-pixel spectral representations. This behavior can be understood through variance reduction and spatial stationarity.

Let a pixel spectrum be modeled as:

    x_i = μ_c + ε_i

where μ_c is the class-conditional mean and ε_i captures illumination variability, sensor noise, subpixel mixing, and fine-scale texture. Pixel-wise classifiers must separate classes under high within-class variance.

By contrast, the neighborhood operator computes summary statistics over an r×r window:

    x̄_i = (1/|N_r|) Σ_{j ∈ N_r(i)} x_j

Under spatial coherence assumptions, variance shrinks approximately as 1/|N_r|, while the class mean remains stable. The representation therefore increases signal-to-noise ratio without increasing feature dimensionality.

The PCA-derived patch statistics and local variance features additionally encode second-order texture structure, capturing discriminative information that spectral basis transforms (e.g., DFT-derived quadrances) do not uniquely provide once spatial context is incorporated.

## 5.2 The Neighborhood Sufficiency Principle (Empirical)

Across datasets, performance improves monotonically with window size up to a scene-dependent threshold r*, beyond which accuracy plateaus or slightly regresses.

This behavior is consistent with two competing effects:

1. **Variance reduction** within homogeneous class regions.
2. **Boundary contamination** when windows span multiple classes.

We therefore observe a minimal sufficient neighborhood scale:

    r* ≈ class spatial coherence radius

Empirically:
- Houston multimodal: r* ≈ 5×5
- Indian Pines: r* ≈ 7×7
- PaviaU: r* ≈ 7×7

These differences align with known scene structure: agricultural fields and urban blocks exhibit larger spatial homogeneity than dense mixed scenes.

**Neighborhood Sufficiency Principle (empirical):**

> A compact local-statistics representation computed over a small, scene-dependent neighborhood achieves performance comparable to or exceeding high-dimensional per-pixel spectral representations, with performance saturating beyond a bounded spatial scale r*.

## 5.3 Minority-Class Rescue and Class Imbalance

On Indian Pines, the improvement in average accuracy (AA) substantially exceeds the improvement in overall accuracy (OA). This indicates that gains are not limited to dominant classes but instead reflect broad improvements across rare and spectrally ambiguous categories.

Pixel-only spectral models exhibit majority-class bias: OA increases while AA remains suppressed. Neighborhood statistics mitigate this effect by stabilizing minority-class signatures through local averaging and texture encoding.

This suggests that spatial context functions not only as a noise-reduction operator but also as a class-balance equalizer.

## 5.4 Why Uncertainty Gating Fails

Uncertainty-based gating assumes that per-modality classifier confidence is aligned with classification correctness. However, in the Houston multimodal dataset, gate confidence exhibits a slight negative correlation with correctness (corr ≈ −0.08). This indicates that ambiguity is intrinsic to the class manifold rather than a reliable fusion signal.

Area-based gating succeeds modestly because amplitude proxies local signal stability. However, even this approach is dominated by explicit spatial encoding.

Thus, uncertainty is not a reliable invariant for multimodal fusion in these scenes; spatial locality is.

## 5.5 Relation to Deep Learning and Texture Theory

The empirical dominance of small neighborhood statistics aligns with classical texture descriptors (e.g., second-order statistics) and with convolutional neural network receptive field theory.

CNNs implicitly learn spatial filters that capture local mean, variance, and gradient information over small receptive fields. The present results show that much of this benefit can be achieved explicitly using handcrafted low-dimensional statistics, without deep architectures.

This suggests that the primary structural advantage of CNN-based hyperspectral models may lie in their implicit spatial aggregation rather than in high-capacity spectral transforms.

## 5.6 Limitations and Scope

The findings are empirical and evaluated on three benchmarks. While locality dominance is consistent across datasets tested, the optimal window size varies with scene structure. Further evaluation on additional multimodal benchmarks and scenes with highly fragmented spatial patterns would strengthen generality claims.

Additionally, the current work uses fixed window shapes and summary statistics; adaptive or anisotropic neighborhoods may further improve performance.

## 5.7 Implications

These results suggest a design principle for hyperspectral and multimodal classification:

1. Encode spatial context first.
2. Keep representation compact.
3. Introduce spectral transforms only if they add orthogonal information beyond local statistics.

In the tested benchmarks, the neighborhood operator subsumes spectral transform generators in class-separating power.
