# Multimodal Fusion Overview (Project Guide)

Purpose: Practical, repo‑aligned summary of multimodal fusion with pointers to our code, data layout, and reproducible commands.

---

## What It Is
- Combine multiple data types (e.g., HSI spectra, LiDAR elevation, multispectral bands) into a single representation or decision to exploit complementary information.
- In this repo, fusion primarily targets remote sensing: HSI + LIDAR + MS.

## Entry Points
- `test_multimodal_fusion.py:1` — Supervised fusion benchmark (HSI + LIDAR + MS) with baselines and chromogeometry features.
- `inspect_multimodal_data.py:1` — Data keys/shapes sanity check for `.mat` files.
- `qa_hyperspectral_pipeline.py:1` — Core QA/chromogeometry feature machinery used across experiments.

## Fusion Strategies
- Early (feature‑level): encode per‑modality, align dimensions, then concatenate or gated‑sum; simple and strong baseline.
- Early (gated‑concat): weight each modality’s features by per‑sample confidence (e.g., energy/entropy). Richardson‑style weights from the chromogeometry notes worked best here.
- Late (decision‑level): train per‑modality models, fuse logits/probabilities via averaging/weights; robust to missing modalities.
- Hybrid (intermediate): cross‑attention/co‑attention, FiLM/conditional BN, gated fusion, low‑rank bilinear pooling; best interactions, higher cost.

## Design Choices That Matter
- Alignment: ensure time/space sync (for images: matched crops/patch centers; for sequences: windowing). Cross‑attention can learn soft alignment.
- Scaling/normalization: per‑modality standardization; temperature scaling for logits to avoid one modality dominating.
- Fusion ops: concat + MLP (baseline), elementwise add/mul, learned gates, attention over modalities, low‑rank bilinear.
- Training schedule: pretrain encoders per modality, then train fusion head; unfreeze encoders later as needed.

## Handling Missing/Noisy Modalities
- Modality dropout during training to build robustness.
- Confidence‑aware gating (entropy/SNR/variance) to down‑weight bad inputs.
- Cross‑modal distillation or imputation when a modality is intermittently absent.
- Late‑fusion fallback path if an encoder fails at inference.

## Recommended Training Recipe (repo‑consistent)
- Encoders: PCA (HSI), identity (LIDAR elevation), optional PCA (MS) or chromogeometry features via `qa_hyperspectral_pipeline.py:1`.
- Fusion head: random forest or small MLP; keep a late‑fusion baseline of per‑modality classifiers.
- Regularization: weight decay, dropout (for MLP), label smoothing; modality dropout (p≈0.1–0.3).
- Auxiliary heads: optional per‑modality heads to balance gradients early.

## Evaluation & Diagnostics
- Ablations: HSI only, HSI+LIDAR+MS concat, Chromogeometry Fusion (matches `test_multimodal_fusion.py:1`).
- Stress: noise per modality, drop a modality, and misalignment checks.
- Calibration: verify fused confidence; temperature‑scale if overconfident.
- Interpretability: inspect gates/attention or per‑modality feature importances.

## Results Snapshot (from `CONSOLIDATED_HYPERSPECTRAL_FINDINGS.md`)
- Dataset: 2,832 patches (11×11), 15 classes; HSI (144) + LIDAR (1) + MS (8)
- Concatenation (HSI PCA 50 + LIDAR + MS): 96.59% (59D)
- Chromogeometry Fusion (HSI 5D + MS 5D + LIDAR 1D): 86.94% (11D) → ~5.4× compression for ~10% accuracy trade‑off.

## Reproduce Locally
Prereqs: Python 3.10+, `pip install numpy scipy scikit-learn` (plus others listed in repo QUICKSTART).

Data layout:
- Place modality files under `multimodal_data/` as used by the scripts:
  - `multimodal_data/HSI_Tr.mat`
  - `multimodal_data/LIDAR_Tr.mat`
  - `multimodal_data/MS_Tr.mat`
  - `multimodal_data/TrLabel.mat`

Commands (from repo root):
- Inspect data: `python inspect_multimodal_data.py`
- Run fusion benchmark: `python test_multimodal_fusion.py`
- Generate MS from HSI (8‑band): `python generate_ms_from_hsi.py --mode wv2` (or `--mode equal`)

Notes:
- If `HSI_Tr.mat` is missing, add it beside the other `.mat` files under `multimodal_data/`.
- `test_multimodal_fusion.py` seeds `RandomForestClassifier` and data split via `random_state=42` for repeatability.

## How This Fits the Repo
- Scripts live at repo root (see entry points above) and emit logs/metrics to the working directory, consistent with `AGENTS.md` and `PAPER_SUBMISSION_README.md`.
- Publication‑ready summaries belong in `Documents/`; this file surfaces the fusion summary and cross‑links runnable experiments.

## Next Steps (Optional)
- Add late‑fusion experiment variant (per‑modality RF/MLP + weighted logit sum) for robustness to missing modalities.
- Add a gated fusion head (simple learned gate over modalities) to compare vs concat.
- Consider a brief section in `PAPER_SUBMISSION_README.md` that links here and to `test_multimodal_fusion.py:1` for reviewers.
- Replace flat WV2 band means with bandpass‑weighted responses per wavelength to refine synthetic MS generation.
