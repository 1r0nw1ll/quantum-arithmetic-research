# QA Vault Reconstruction Runs

This note collects the runnable entry points rebuilt from the vault cache and
documents how to execute them on real data.

---

## 1. Bell-Test Suite (`qa_chsh_bell_test.py`, `qa_i3322_bell_test.py`, `qa_platonic_bell_tests.py`)
- **Run everything (plots saved in repo root):**
  ```bash
  python qa_chsh_bell_test.py
  ```
- **I₃₃₂₂ framework (requires coefficient update before final validation):**
  ```bash
  python qa_i3322_bell_test.py --help
  ```
- **Platonic solid explorations:**
  ```bash
  python qa_platonic_bell_tests.py
  ```
- Outputs are written next to the scripts (`qa_chsh_landscape_N24.png`,
  `qa_chsh_n_dependence.png`, `qa_chsh_24gon_visualization.png`,
  `qa_platonic_solids_bell_tests.png`, `qa_platonic_solids_3d.png`).

## 2. Hyperspectral Pipeline (`qa_hyperspectral_pipeline.py`)
- **Synthetic sanity check (default):**
  ```bash
  python qa_hyperspectral_pipeline.py
  ```
- **Real data run (Matlab/NumPy formats supported):**
  ```bash
  mkdir -p data/hyperspectral
  # Place Indian Pines cubes (e.g. indian_pines_corrected.mat) inside the folder.
  python qa_hyperspectral_pipeline.py \
      --cube-path data/hyperspectral/indian_pines_corrected.mat \
      --dataset-key indian_pines_corrected \
      --bins 24 \
      --kmeans-k 8 \
      --dbscan-eps 0.35 \
      --sector-field Er \
      --pca-k 3
  ```
- `.npy`/`.npz` also work (`--dataset-key` optional). Loading `.mat` files requires
  `scipy` in the active environment.

## 3. Post-Calculus Engine (`qa_post_calculus_engine.py`)
- **Vault-style sin(x) experiment with plots:**
  ```bash
  python qa_post_calculus_engine.py --function sin --steps 128 --plot
  ```
- **Quantised harmonic increments (1/144) on exp(x):**
  ```bash
  python qa_post_calculus_engine.py \
      --function exp \
      --domain 0 2 \
      --steps 96 \
      --quantise 144 \
      --plot
  ```
- Results: console metrics + `qa_post_calculus_{function}_integral.png` and
  `qa_post_calculus_{function}_error.png`.

---

### Environment Notes
- All scripts assume the repository’s QA virtual environment is active.
- Install optional extras when needed:
  ```bash
  pip install scipy matplotlib
  ```
- Long Bell-sweep runs can take a couple of minutes; expect several PNGs in the repo root.

