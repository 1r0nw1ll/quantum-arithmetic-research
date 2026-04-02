# Running Experiments

All experiments are standalone Python scripts. No build system or test suite — this is research code.

## Signal Processing
```bash
python run_signal_experiments_final.py
```
Tests QA system on audio signals: pure tones, major/minor chords, tritones, white noise. Generates harmonic classification visualizations.

## Geometric Analysis
```bash
python geometric_autopsy.py
```
Angular spectrum analysis, TDA with persistent homology, dimensionality reduction (Isomap/DBSCAN).

## Generative Synthesis
```bash
python final_generative_test.py
```
Reverse-engineers geometric rules and synthesizes new instances of the "Harmonic Coherence Manifold." Uses KS tests for validation.

## Neural Network Co-Processor
```bash
python intelligent_coprocessor_v2.py
python statistical_validation_gauntlet.py
```
QA-guided neural network training via adaptive learning rates. Uses MNIST/CIFAR-10 from `/data/`.

## Financial Backtesting
```bash
python backtest_advanced_strategy.py
```
Harmonic Index applied to market regime detection with 200-day SMA. Note: finance scripts 30-37 are FROZEN — see `~/Desktop/qa_finance/`.

## Automated Theorem Generation
```bash
python geometrist_v4_gnn.py
```
GCN layers learn and generate valid geometric theorems from UHG concepts.

## Formal Documentation
```bash
pdflatex qa_formal_report.tex
python qa_proof_export.py
```

## Dependencies

Install as needed:
```bash
pip install numpy matplotlib pandas torch scikit-learn tqdm scipy ripser persim seaborn yfinance
```

Core stack: NumPy (numerics), PyTorch (NNs), scikit-learn (ML), matplotlib/seaborn (viz), ripser/persim (TDA), scipy (stats).

## Data Requirements

- MNIST: `/data/mnist/`
- CIFAR-10: `/data/cifar10/`
- Financial data via `yfinance` (ticker: SPY)
