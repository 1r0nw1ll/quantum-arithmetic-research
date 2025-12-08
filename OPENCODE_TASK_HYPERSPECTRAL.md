# OpenCode Agent Task Specification: Hyperspectral Pipeline Testing

**Task ID:** HYPERSPECTRAL_VALIDATION_001
**Priority:** 2 (from project roadmap)
**Estimated Duration:** 4-6 hours
**Assigned Date:** 2025-10-31

---

## Objective

Test the QA hyperspectral imaging pipeline (`qa_hyperspectral_pipeline.py`) on real-world hyperspectral datasets to validate:
1. Phase-aware DFT encoding functionality
2. QA chromatic field generation (Eb, Er, Eg)
3. Harmonic-aware clustering performance
4. Classification accuracy compared to baselines

---

## Context & Background

**What is the QA Hyperspectral Pipeline?**
- Novel approach using Quantum Arithmetic (QA) for hyperspectral image analysis
- Converts spectral signatures to (b,e) QA tuples via phase-aware DFT
- Generates chromatic fields: Eb (electric), Er (magnetic), Eg (scalar)
- Uses harmonic-aware K-means and DBSCAN for clustering
- Previously tested on synthetic 50×50×100 cube (successful)

**Current Status:**
- Implementation complete: `qa_hyperspectral_pipeline.py` (650 lines)
- Synthetic tests passed ✓
- Real dataset testing: NOT YET DONE
- Datasets available in `archive.zip` ✓

**Why This Matters:**
- Validates QA framework on real scientific data
- Complements Bell inequality validation (Priority 1, completed)
- Part of unified QA framework demonstration
- Potential publication material

---

## Input Resources

### 1. Code Implementation
**File:** `/home/player2/signal_experiments/qa_hyperspectral_pipeline.py`
**Size:** 650 lines
**Status:** Ready to run
**Key Functions:**
- `spectrum_to_be_phase_multi()` - DFT encoding
- `qa_chromatic_fields()` - Generate Eb, Er, Eg fields
- `kmeans_from_scratch()` - QA-aware clustering
- `dbscan_from_scratch()` - Density-based clustering

### 2. Datasets
**Location:** `/home/player2/signal_experiments/archive.zip`
**Contents:**
- `Indian_pines_corrected.mat` (5.95 MB) + `Indian_pines_gt.mat` (1.1 KB)
- `PaviaU.mat` (34.8 MB) + `PaviaU_gt.mat` (11 KB)
- `KSC.mat` (56.8 MB) + `KSC_gt.mat` (3.2 KB)
- `Salinas_corrected.mat` (26.6 MB) + `Salinas_gt.mat` (4.3 KB)

**Format:** MATLAB .mat files (readable via scipy.io.loadmat)

### 3. Reference Documentation
- `qa_hyperspectral_pipeline.py` - Contains full docstrings
- `CLAUDE.md` - Project overview and QA system description
- `BELL_TESTS_FINAL_SUMMARY.md` - Related validation work

---

## Task Steps (Detailed)

### Step 1: Environment Setup (30 min)

**Actions:**
1. Extract datasets from archive.zip:
   ```bash
   cd /home/player2/signal_experiments
   unzip -j archive.zip "*.mat" -d hyperspectral_data/
   ```

2. Verify Python dependencies:
   ```python
   import numpy
   import matplotlib.pyplot
   import scipy.io  # For loading .mat files
   from sklearn.decomposition import PCA
   from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
   ```

3. Install missing dependencies if needed:
   ```bash
   pip install scipy scikit-learn matplotlib
   ```

**Success Criteria:**
- All 8 .mat files extracted to `hyperspectral_data/`
- All Python imports successful
- No version conflicts

**Deliverable:**
- `setup_log.txt` documenting environment state

---

### Step 2: Dataset Loading & Inspection (45 min)

**Actions:**

1. Create `load_hyperspectral_dataset.py` helper script:
   ```python
   import scipy.io as sio
   import numpy as np

   def load_dataset(name):
       """Load hyperspectral dataset and ground truth."""
       data_file = f'hyperspectral_data/{name}.mat'
       gt_file = f'hyperspectral_data/{name}_gt.mat'

       # Load data (handle different key names)
       data_mat = sio.loadmat(data_file)
       gt_mat = sio.loadmat(gt_file)

       # Extract actual data (skip MATLAB metadata)
       data_keys = [k for k in data_mat.keys() if not k.startswith('__')]
       gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]

       data = data_mat[data_keys[0]]
       gt = gt_mat[gt_keys[0]]

       return data, gt

   def dataset_summary(name, data, gt):
       """Print dataset statistics."""
       print(f"\n{name} Dataset:")
       print(f"  Shape: {data.shape} (H×W×Bands)")
       print(f"  Ground truth shape: {gt.shape}")
       print(f"  Number of classes: {len(np.unique(gt)) - 1}")  # -1 for background
       print(f"  Labeled pixels: {np.sum(gt > 0)}")
       print(f"  Total pixels: {data.shape[0] * data.shape[1]}")
       print(f"  Spectral range: [{data.min():.2f}, {data.max():.2f}]")
   ```

2. Load and inspect all 4 datasets:
   ```python
   datasets = ['Indian_pines_corrected', 'PaviaU', 'KSC', 'Salinas_corrected']
   for name in datasets:
       data, gt = load_dataset(name)
       dataset_summary(name, data, gt)
   ```

**Success Criteria:**
- All 4 datasets load without errors
- Dataset dimensions printed correctly
- Ground truth classes identified

**Deliverable:**
- `dataset_inspection_report.txt` with statistics for each dataset

---

### Step 3: Indian Pines Testing (2 hours)

**Dataset:** Indian_pines_corrected.mat (smallest - start here)

**Actions:**

1. **Adapt pipeline for real data:**
   - Modify `qa_hyperspectral_pipeline.py` to accept .mat files
   - Add command-line arguments for dataset selection
   - Add ground truth evaluation metrics

2. **Run pipeline:**
   ```bash
   python qa_hyperspectral_pipeline.py \
       --dataset hyperspectral_data/Indian_pines_corrected.mat \
       --ground-truth hyperspectral_data/Indian_pines_gt.mat \
       --output-dir results/indian_pines/ \
       --bins 24 \
       --k-clusters auto
   ```

3. **Generate outputs:**
   - Phase-aware (b,e) parameter maps
   - Chromatic field visualizations (Eb, Er, Eg maps)
   - K-means clustering results
   - DBSCAN clustering results
   - Sector-masked classifications
   - Comparison with ground truth

4. **Compute metrics:**
   ```python
   from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
   from sklearn.metrics import classification_report

   metrics = {
       'ARI': adjusted_rand_score(gt_flat, predicted_flat),
       'NMI': normalized_mutual_info_score(gt_flat, predicted_flat),
       'Accuracy': accuracy_score(gt_flat, predicted_flat)
   }
   ```

**Success Criteria:**
- Pipeline runs to completion
- All visualizations generated
- Metrics computed and reasonable (ARI > 0.3 acceptable for first pass)
- No crashes or memory errors

**Deliverables:**
- `results/indian_pines/phase_map.png`
- `results/indian_pines/chromatic_fields.png`
- `results/indian_pines/clustering_comparison.png`
- `results/indian_pines/metrics.json`

---

### Step 4: PaviaU Testing (1.5 hours)

**Dataset:** PaviaU.mat (larger dataset)

**Actions:**

1. Run same pipeline on PaviaU:
   ```bash
   python qa_hyperspectral_pipeline.py \
       --dataset hyperspectral_data/PaviaU.mat \
       --ground-truth hyperspectral_data/PaviaU_gt.mat \
       --output-dir results/pavia_u/ \
       --bins 24 \
       --k-clusters auto
   ```

2. Compare performance vs Indian Pines
3. Note any computational challenges (memory, runtime)

**Success Criteria:**
- Same outputs as Indian Pines
- Runtime documented
- Memory usage acceptable

**Deliverables:**
- Same file structure as Indian Pines in `results/pavia_u/`
- `results/pavia_u/metrics.json`

---

### Step 5: Baseline Comparison (1.5 hours)

**Objective:** Show QA pipeline performance vs standard methods

**Actions:**

1. **Implement baseline methods:**
   ```python
   # Baseline 1: K-means on raw spectra
   from sklearn.cluster import KMeans
   kmeans_baseline = KMeans(n_clusters=n_classes)
   labels_baseline = kmeans_baseline.fit_predict(spectra_flat)

   # Baseline 2: K-means on PCA-reduced spectra
   from sklearn.decomposition import PCA
   pca = PCA(n_components=10)
   spectra_pca = pca.fit_transform(spectra_flat)
   kmeans_pca = KMeans(n_clusters=n_classes)
   labels_pca = kmeans_pca.fit_predict(spectra_pca)
   ```

2. **Compute metrics for all methods:**
   - QA pipeline (phase-aware)
   - K-means raw
   - K-means PCA
   - (Optional) Spectral clustering

3. **Generate comparison table:**
   ```python
   results_table = pd.DataFrame({
       'Method': ['QA Phase-Aware', 'K-Means Raw', 'K-Means PCA'],
       'ARI': [ari_qa, ari_raw, ari_pca],
       'NMI': [nmi_qa, nmi_raw, nmi_pca],
       'Runtime': [t_qa, t_raw, t_pca]
   })
   ```

**Success Criteria:**
- Baselines run successfully on both datasets
- Fair comparison (same number of clusters, same preprocessing)
- QA pipeline shows competitive or better performance

**Deliverable:**
- `results/comparison_table.csv`
- `results/comparison_visualization.png` (bar chart of metrics)

---

### Step 6: Optional - KSC and Salinas (if time permits)

Only if Steps 1-5 complete ahead of schedule.

**Actions:**
- Run pipeline on KSC.mat
- Run pipeline on Salinas_corrected.mat
- Add to comparison table

**Deliverables:**
- `results/ksc/` directory with outputs
- `results/salinas/` directory with outputs

---

### Step 7: Documentation & Summary (30 min)

**Actions:**

1. Create comprehensive summary document:
   ```markdown
   # QA Hyperspectral Pipeline Validation Report

   ## Executive Summary
   [Key findings, performance metrics, computational cost]

   ## Dataset Results
   [Detailed results for each dataset]

   ## Comparison Analysis
   [QA vs baselines]

   ## Visualizations
   [Embedded images or references]

   ## Computational Performance
   [Runtime, memory usage]

   ## Conclusions & Next Steps
   [Interpretation, limitations, future work]
   ```

2. File organization:
   ```
   results/
   ├── indian_pines/
   ├── pavia_u/
   ├── comparison_table.csv
   ├── comparison_visualization.png
   └── HYPERSPECTRAL_VALIDATION_REPORT.md
   ```

**Deliverable:**
- `results/HYPERSPECTRAL_VALIDATION_REPORT.md` (main document)

---

## Expected Outputs (Complete List)

### Generated Files
```
results/
├── indian_pines/
│   ├── phase_map.png
│   ├── chromatic_fields.png
│   ├── clustering_comparison.png
│   ├── metrics.json
│   └── qa_parameters.npy
├── pavia_u/
│   ├── phase_map.png
│   ├── chromatic_fields.png
│   ├── clustering_comparison.png
│   ├── metrics.json
│   └── qa_parameters.npy
├── comparison_table.csv
├── comparison_visualization.png
├── setup_log.txt
├── dataset_inspection_report.txt
└── HYPERSPECTRAL_VALIDATION_REPORT.md
```

### Key Metrics to Report
- **Adjusted Rand Index (ARI):** Clustering agreement with ground truth
- **Normalized Mutual Information (NMI):** Information-theoretic similarity
- **Runtime:** Seconds per dataset
- **Memory Usage:** Peak RAM consumption
- **Classification Accuracy:** If applicable

---

## Error Handling & Contingencies

### Potential Issues & Solutions

**Issue 1: Memory errors on large datasets**
**Solution:** Implement chunked processing or downsample spatially

**Issue 2: .mat file format incompatibility**
**Solution:** Try different scipy.io.loadmat parameters or use h5py for v7.3 files

**Issue 3: Pipeline crashes on real data**
**Solution:** Add try-except blocks, log errors, continue with remaining datasets

**Issue 4: Poor clustering performance (ARI < 0.2)**
**Solution:** Document honestly - not all methods work on all data. Try parameter tuning (bins=12 or bins=36).

**Issue 5: Missing dependencies**
**Solution:** Install as needed, document in setup_log.txt

---

## Success Criteria (Summary)

**Minimum Acceptable:**
- ✓ Pipeline runs successfully on Indian Pines
- ✓ Visualizations generated
- ✓ Metrics computed and documented
- ✓ Baseline comparison completed
- ✓ Summary report written

**Target (Full Success):**
- ✓ All of above PLUS:
- ✓ Pipeline runs on PaviaU
- ✓ QA shows competitive performance vs baselines
- ✓ Clean, reproducible code
- ✓ Professional documentation

**Stretch Goals:**
- ✓ All 4 datasets tested
- ✓ QA outperforms baselines
- ✓ Parameter sensitivity analysis

---

## Time Budget

| Task | Estimated Time |
|------|----------------|
| Step 1: Setup | 30 min |
| Step 2: Data Loading | 45 min |
| Step 3: Indian Pines | 2 hours |
| Step 4: PaviaU | 1.5 hours |
| Step 5: Baseline Comparison | 1.5 hours |
| Step 6: KSC/Salinas (optional) | +2 hours |
| Step 7: Documentation | 30 min |
| **Total (core)** | **6 hours 45 min** |

---

## Communication & Checkpoints

**Checkpoint 1 (After Step 2):**
- Report dataset loading status
- Confirm all data is accessible
- Flag any format issues

**Checkpoint 2 (After Step 3):**
- Share Indian Pines results
- Report initial metrics
- Request feedback if results unexpected

**Checkpoint 3 (After Step 5):**
- Share comparison results
- Highlight key findings
- Confirm whether to proceed to Step 6

**Final Report:**
- Complete summary document
- All outputs organized in results/
- Ready for review and potential publication

---

## References

**Project Documentation:**
- `/home/player2/signal_experiments/CLAUDE.md` - Project overview
- `/home/player2/signal_experiments/qa_hyperspectral_pipeline.py` - Implementation
- `/home/player2/signal_experiments/BELL_TESTS_FINAL_SUMMARY.md` - Related work

**Literature:**
- Standard hyperspectral classification benchmarks
- QA framework theoretical foundations (see vault)

---

## Notes for Agent

**Important Considerations:**

1. **Honesty in Reporting:** If QA pipeline performs poorly, document it. Science requires honest negative results.

2. **Computational Limits:** If datasets are too large, downsample rather than crash. Document the choice.

3. **Parameter Tuning:** The `bins=24` default comes from QA theory. If results are poor, try bins ∈ {12, 18, 24, 36} and report which works best.

4. **Code Modifications:** You may need to add a CLI interface to `qa_hyperspectral_pipeline.py`. Preserve the core functions; add wrapper code as needed.

5. **Visualization Quality:** Make figures publication-ready (high DPI, clear labels, color-blind friendly if possible).

6. **Git Tracking:** Consider committing intermediate results to git with clear commit messages.

---

## Approval & Launch

**Prepared by:** Claude (session 2025-10-31)
**Status:** DRAFT - Awaiting user approval
**Ready to launch:** YES (pending review)

**To approve and launch:**
```bash
opencode run --continue "Execute the hyperspectral testing task specified in OPENCODE_TASK_HYPERSPECTRAL.md. Work through steps 1-5, report at each checkpoint, and generate the final validation report. Focus on Indian Pines and PaviaU datasets as priority."
```

---

Generated: 2025-10-31
Task ID: HYPERSPECTRAL_VALIDATION_001
