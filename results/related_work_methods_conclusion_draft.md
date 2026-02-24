# Related Work, Methods, and Conclusion Drafts

---

## 2. Related Work

### 2.1 Pixel-wise Spectral Classification

Classical hyperspectral image (HSI) classification methods operate on per-pixel spectral signatures, often comprising 100–200 bands. Dimensionality reduction techniques such as principal component analysis (PCA), linear discriminant analysis (LDA), and band selection methods are commonly applied prior to classification using support vector machines (SVM), random forests (RF), or k-nearest neighbors. While effective in controlled settings, pixel-wise approaches are sensitive to intra-class spectral variability and mixed pixels, often requiring high model capacity to compensate.

Our spectral baselines (PCA-30 and raw spectral RF) follow this standard protocol to provide a direct comparison against neighborhood-based representations.

### 2.2 Spectral–Spatial Feature Extraction

To address the limitations of pixel-only models, numerous works incorporate spatial context through handcrafted texture descriptors, morphological profiles, and spatial filtering techniques. Extended morphological profiles (EMP), gray-level co-occurrence matrix (GLCM) features, and local binary patterns (LBP) have demonstrated that second-order spatial statistics improve classification stability.

More recently, patch-based feature extraction and joint spectral–spatial descriptors have become standard, particularly in conjunction with deep architectures. However, many methods either dramatically increase feature dimensionality or rely on learned convolutional filters.

In contrast, the present work isolates a minimal set of low-dimensional neighborhood summary statistics and evaluates their sufficiency without deep models or high-capacity transforms.

### 2.3 Convolutional Neural Networks for HSI

Convolutional neural networks (CNNs) have achieved state-of-the-art performance in hyperspectral classification by learning hierarchical spectral–spatial filters. These architectures implicitly exploit local receptive fields, progressively aggregating spatial information across layers.

While CNNs demonstrate strong performance, they introduce significant computational overhead and model complexity. The present study investigates whether the performance gains attributed to deep models may arise primarily from local spatial aggregation rather than from deep spectral transformation capacity.

Our results suggest that small fixed neighborhoods (5×5–7×7) with simple second-order statistics capture much of the discriminative power typically associated with convolutional architectures on standard benchmarks.

### 2.4 Multimodal Fusion and Gating

Multimodal remote sensing benchmarks (e.g., HSI + LiDAR + multispectral) have motivated diverse fusion strategies, including feature concatenation, late fusion, attention mechanisms, and uncertainty-driven gating. Entropy-based or margin-based gating schemes aim to weight modalities according to classifier confidence.

However, uncertainty estimates may not align with true class-separating invariants in complex scenes. Our results provide empirical evidence that uncertainty-driven gating can degrade performance and that amplitude-based heuristics provide only modest improvement relative to explicit spatial encoding.

---

## 3. Methods

### 3.1 Datasets

We evaluate on three benchmarks:

- **Houston Multimodal:** Hyperspectral (144 bands), multispectral (8 bands), and LiDAR elevation data with 15 classes. Pre-extracted 11×11 patches are provided; 2832 labeled samples total.
- **Indian Pines:** 145×145 HSI with 200 spectral bands and 16 classes (10249 labeled pixels).
- **PaviaU:** 610×340 HSI with 103 spectral bands and 9 classes (42776 labeled pixels).

For Indian Pines and PaviaU, patches are extracted dynamically from the full image cube using reflect padding at image borders.

### 3.2 Train–Test Protocol

For Indian Pines and PaviaU, we use stratified random splits with 10% of labeled pixels per class allocated to training and 90% to testing, following standard literature protocol. For Houston multimodal, we use a 70%/30% stratified split. All reported results use identical splits across methods for fair comparison. For Houston and Indian Pines, experiments are repeated across three random seeds (42, 1, 7) to assess stability.

PCA transformations are fitted exclusively on the training set and applied to test data to prevent leakage. Patch-level PCA statistics are likewise fitted on training patch pixels only.

### 3.3 Spectral Baselines

Two spectral baselines are evaluated:

1. **PCA-k spectral baseline:** PCA reduced to k components (k=30 for HSI-only datasets, k=50 for Houston), classified using Random Forest (200 estimators).
2. **Raw spectral baseline:** All bands (103–200) used directly with Random Forest.

For Houston multimodal, a 59-dimensional concatenation baseline (HSI PCA-50 + LiDAR + MS) is used as the primary multimodal reference, along with area-weighted gating variants.

### 3.4 Neighborhood-Statistic Representation

For each labeled pixel, a square window of size r×r centered at the pixel is extracted. The following features are computed over the neighborhood:

- Mean over all spectral bands (flattened patch)
- Standard deviation over all spectral bands
- Gradient magnitude (finite differences on mean-pooled spectral intensity at patch center)
- Mean and standard deviation of the first three PCA components computed over the r×r patch pixels (6 features)

For the multimodal Houston dataset, additional modality-specific statistics are included: LiDAR center-pixel residual (center minus patch mean), MS gradient magnitude, and per-modality mean/std. This yields a 14-dimensional feature vector for Houston and a 10-dimensional vector for HSI-only datasets.

Window sizes r ∈ {3, 5, 7, 11} are evaluated to identify the minimal sufficient spatial scale. The chosen center for all crops is pixel position (5, 5) within the 11×11 pre-extracted patch for Houston, and the labeled pixel center for dynamically extracted patches.

### 3.5 Uncertainty Gating (Houston Only)

For the Houston multimodal dataset, per-modality Random Forest classifiers are trained and used to compute uncertainty signals per sample: entropy (H = −Σ p log p), margin (top-1 minus top-2 probability), and Gini impurity (1 − Σ p²). Gate weights are computed as softmax(u/T) where u is the uncertainty signal and T is a temperature parameter. Sweeps over T ∈ {0.1, 0.5, 1.0, 2.0, 5.0} are conducted.

Gate diagnostics reported include: per-modality mean gate weight, saturation rate (fraction of samples where any gate weight > 0.9 or < 0.1), and Pearson correlation between maximum gate weight and per-sample classification correctness.

### 3.6 Classification Model

Random Forest classifiers (200 estimators, fixed random seed) are used for all primary comparisons to isolate representational effects from model capacity. Experiments with gradient-boosted decision trees (HistGradientBoostingClassifier) confirm that the ranking between representations is stable across classifier choices (difference ≤ 0.24pp).

### 3.7 Evaluation Metrics

Performance is measured using:

- **Overall Accuracy (OA):** Fraction of correctly classified test pixels.
- **Average Accuracy (AA):** Mean of per-class accuracies; emphasizes minority-class performance under class imbalance.
- **Cohen's kappa coefficient (κ):** Agreement corrected for chance.

Average Accuracy is emphasized throughout to assess whether gains are broad-based or driven by dominant classes.

---

## 6. Conclusion

We investigated whether high-dimensional spectral representations are necessary for hyperspectral and multimodal land-cover classification once local spatial context is explicitly encoded. Across three benchmarks (Houston multimodal, Indian Pines, and PaviaU) using stratified 10% training splits, we found that a compact 10–14 dimensional neighborhood-statistics representation consistently outperforms 30–200 dimensional spectral baselines. Improvements range from +3.0 to +11.1 percentage points in overall accuracy and +6.3 to +16.1 points in average accuracy, with particularly strong gains for minority classes.

Performance saturates at a small, scene-dependent neighborhood scale (5×5–7×7), suggesting the existence of a minimal sufficient spatial radius beyond which larger windows provide no consistent benefit. In contrast, uncertainty-driven multimodal gating strategies fail to improve performance and, in some cases, substantially degrade it, with gate confidence exhibiting slight negative correlation with classification correctness.

These results indicate that local second-order spatial statistics constitute a dominant and efficient representational mechanism for standard hyperspectral benchmarks. Importantly, this representation achieves superior performance with substantial dimensionality reduction (76–95% fewer features) and minimal computational overhead.

Future work should evaluate this principle on additional multimodal datasets, explore adaptive or anisotropic neighborhood selection, and analyze the interaction between explicit neighborhood statistics and learned convolutional representations.
