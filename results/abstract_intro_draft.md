# Abstract and Introduction Draft

---

## Abstract

Hyperspectral and multimodal land-cover classification methods typically rely on high-dimensional per-pixel spectral representations and increasingly complex fusion strategies. In this work, we demonstrate that a compact neighborhood-statistic representation—computed entirely from local spatial context—outperforms substantially higher-dimensional spectral baselines across three benchmarks. Using stratified 10% training splits, a 10–14 dimensional patch-statistics feature vector (mean, standard deviation, gradient magnitude, and PCA-derived neighborhood descriptors) achieves +3.0–11.1 percentage point (pp) improvements in overall accuracy (OA) and +6.3–16.1pp in average accuracy (AA) relative to 30–200 dimensional spectral baselines on Houston (multimodal), Indian Pines, and PaviaU datasets. Notably, gains in AA exceed gains in OA, indicating improved minority-class performance. Accuracy saturates at a small, scene-dependent neighborhood scale (5×5–7×7), beyond which larger windows offer no consistent benefit. In contrast, uncertainty-driven gating and spectral transform augmentations provide negligible or negative impact once spatial context is encoded. These results suggest that local second-order spatial statistics constitute a sufficient and dominant representation for class separation in standard hyperspectral benchmarks, substantially reducing feature dimensionality while improving performance.

---

## 1. Introduction

Hyperspectral image (HSI) and multimodal remote sensing classification remain central tasks in land-cover analysis. Traditional approaches emphasize high-dimensional spectral representations, often employing dimensionality reduction (e.g., PCA), spectral transforms, or increasingly complex multimodal fusion strategies. More recently, deep convolutional architectures have demonstrated strong performance by implicitly exploiting spatial context through learned receptive fields. Despite these advances, the relative importance of spectral transforms versus explicit spatial aggregation remains underexplored.

Most per-pixel spectral classifiers operate on high-dimensional feature vectors (100–200 bands), implicitly assuming that class separability is encoded primarily in spectral signatures. However, real-world scenes exhibit substantial intra-class spectral variability arising from illumination changes, mixed pixels, and local texture effects. Consequently, pixel-wise models frequently require either high model capacity or complex fusion mechanisms to stabilize predictions.

In this work, we revisit a fundamental question:
**Is high-dimensional spectral encoding necessary once local spatial context is explicitly represented?**

We evaluate a compact neighborhood-statistics representation constructed entirely from local spatial summary features—mean, standard deviation, gradient magnitude, and PCA-derived descriptors computed within small spatial windows. The resulting feature vector is 10–14 dimensions, independent of the number of spectral bands.

Across three benchmarks—Houston (multimodal HSI+MS+LiDAR), Indian Pines, and PaviaU—using stratified 10% training splits, we observe that this neighborhood-based representation consistently outperforms 30–200 dimensional spectral baselines. Improvements range from +3.0 to +11.1pp in overall accuracy and +6.3 to +16.1pp in average accuracy, with substantial gains in minority-class performance. Accuracy saturates at a small, scene-dependent window size (5×5–7×7), indicating the existence of a minimal sufficient spatial scale.

We further show that uncertainty-driven gating strategies do not improve performance in the multimodal setting and that spectral transform augmentations become redundant once spatial context is encoded.

The primary contributions of this work are:

1. **Neighborhood Sufficiency Principle (empirical):** A compact local-statistics representation achieves performance comparable to or exceeding high-dimensional spectral encodings across datasets.
2. **Minimal Spatial Scale Identification:** Classification accuracy saturates at a small, scene-dependent neighborhood radius r*, ranging from 5×5 to 7×7 across tested benchmarks.
3. **Minority-Class Stabilization:** Neighborhood statistics disproportionately improve average accuracy (ΔAA > ΔOA), mitigating majority-class bias inherent in high-dimensional spectral models.
4. **Negative Fusion Result:** Uncertainty-based gating (entropy, margin, Gini) does not confer benefit in the evaluated multimodal benchmark; gate confidence exhibits slight negative correlation with classification correctness.

These findings suggest that local second-order spatial statistics constitute a dominant representational mechanism for standard hyperspectral benchmarks, with significant dimensionality and computational advantages.
