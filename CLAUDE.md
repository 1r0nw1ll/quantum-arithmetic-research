# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mathematical research project exploring the **Quantum Arithmetic (QA) System** - a novel modular arithmetic framework with applications in signal processing, finance, neural network optimization, and automated theorem generation. The research is computational and experimental, not a traditional software project.

## QA Mapping Protocol (Gate 0)

The `qa_alphageometry_ptolemy/qa_meta_validator.py` family sweep enforces an **intake constitution**:

- Every certificate family root must contain **exactly one** of `mapping_protocol.json` (inline) or `mapping_protocol_ref.json` (reference).
- Protocol schemas + validators live at repo root: `qa_mapping_protocol/` and `qa_mapping_protocol_ref/`.

Quick checks:
```bash
python qa_mapping_protocol/validator.py --self-test
python qa_mapping_protocol_ref/validator.py --self-test
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## Core Mathematical Framework

### The QA System Architecture

The foundational concept appears across multiple files as either `QA_Engine` or `QASystem` classes. Key properties:

- **Modular arithmetic**: Operations use either mod 9 (theoretical work) or mod 24 (applied experiments)
- **State representation**: Pairs (b, e) that generate tuples (b, e, d, a) where:
  - `d = (b + e) % modulus`
  - `a = (b + 2*e) % modulus`
- **Multi-orbit structure**: State space partitions into three orbits:
  - 24-cycle "Cosmos" (72 starting pairs, 1D linear structure)
  - 8-cycle "Satellite" (8 pairs, 3D symmetric structure)
  - 1-cycle "Singularity" (fixed point at (9,9))

### Key Metrics

**E8 Alignment**: Projects 4D QA tuples into 8D space and measures cosine similarity to E8 root system (240 vectors). This metric appears throughout the codebase and is central to the "Harmonic Index."

**Harmonic Index (HI)**: Composite metric combining E8 alignment and loss:
```python
HI = E8_alignment × exp(-0.1 × loss)
```

## Running Experiments

All experiments are standalone Python scripts. No build system or test suite - this is research code.

### Signal Processing Experiments
```bash
python run_signal_experiments_final.py  # Latest version with corrected signal injection logic
```
Tests QA system on audio signals: pure tones, major/minor chords, tritones, white noise. Generates harmonic classification visualizations.

### Geometric Analysis
```bash
python geometric_autopsy.py  # Three-part geometric characterization
```
Performs angular spectrum analysis, topological data analysis (TDA with persistent homology), and dimensionality reduction (Isomap/DBSCAN).

### Generative Synthesis
```bash
python final_generative_test.py  # Latest version with topological clustering
```
Reverse-engineers geometric rules and synthesizes new instances of the "Harmonic Coherence Manifold." Uses Kolmogorov-Smirnov tests for validation.

### Neural Network Co-Processor
```bash
python intelligent_coprocessor_v2.py      # Main co-processor experiment
python statistical_validation_gauntlet.py # Statistical validation framework
```
Tests whether QA system can guide neural network training via adaptive learning rates based on geometric stress metrics. Uses MNIST/CIFAR-10 datasets from `/data/` directory (405MB).

### Financial Backtesting
```bash
python backtest_advanced_strategy.py  # S&P 500 strategy with Harmonic Index
```
Applies Harmonic Index to market regime detection combined with traditional technical analysis (200-day SMA).

### Automated Theorem Generation
```bash
python geometrist_v4_gnn.py  # Graph Neural Network theorem generator
```
Uses GCN layers to learn and generate valid geometric theorems from Universal Hyperbolic Geometry concepts (quadreas, quadrumes).

### Formal Mathematical Documentation
```bash
pdflatex qa_formal_report.tex  # Compile formal LaTeX report
python qa_proof_export.py      # Generate symbolic verification tables
```

## Dependencies

No requirements.txt provided. Install as needed:
```bash
pip install numpy matplotlib pandas torch scikit-learn tqdm scipy ripser persim seaborn yfinance
```

Core stack:
- **NumPy**: All numerical computations, modular arithmetic
- **PyTorch**: Neural networks (CNNs for MNIST/CIFAR-10, GCNs for theorem generation)
- **scikit-learn**: PCA, Isomap, DBSCAN, Random Forests
- **matplotlib/seaborn**: Visualization
- **ripser/persim**: Topological Data Analysis
- **scipy**: Statistical tests (KS test, t-test)

## File Evolution Pattern

Multiple experiments have versioned iterations showing progressive refinement:
- `run_signal_experiments.py` → `run_signal_experiments_corrected.py` → `run_signal_experiments_final.py`
- `generative_test.py` → `symmetry_generative_test.py` → `final_generative_test.py`

**Always use the `_final` or latest version** - earlier versions contain bugs or incomplete logic.

## Key Implementation Patterns

### QA State Updates

The core update logic follows this pattern (from `run_signal_experiments_final.py:71-100`):

1. Create signal-influenced proposed state
2. Calculate QA tuples for proposed state
3. Compute resonance/coupling via einsum operations: `np.einsum('ik,jk->ij', tuples, tuples)`
4. Calculate neighbor pull using weighted adjacency
5. Apply noise annealing: `noise * (NOISE_ANNEALING ** t)`
6. Update state with modular arithmetic

### Markovian Coupling

Weight matrices are dynamically updated based on tuple resonance, not pre-defined. This creates self-organizing network behavior where geometric alignment influences node coupling strength.

### Signal Injection

External signals (audio, price returns, gradients) are injected into the `b` state variable and influence coupling calculation before propagating through the network.

## Data Requirements

- **MNIST dataset**: `/data/mnist/` (for neural network experiments)
- **CIFAR-10 dataset**: `/data/cifar10/` (for neural network experiments)
- Financial data fetched via `yfinance` (S&P 500 ticker: 'SPY')

## Research Documentation

- **GEMINI.md**: High-level project overview (read this first)
- **qa_formal_report.tex**: Formal mathematical foundations and proofs
- **QAnotes/**: Obsidian vault with 100+ markdown research notes
- **Documents/**: Working drafts and context files

## Important Architectural Notes

1. **No zero element**: QA arithmetic uses {1,2,...,9} or {1,2,...,24}, not {0,1,...,N-1}
2. **Geometry is emergent**: The E8 alignment and orbital structures arise from dynamics, not explicit encoding
3. **Coupling is bidirectional**: Signal affects coupling, coupling affects evolution
4. **Statistical rigor**: Experiments use multi-trial validation, t-tests, and KS tests - check for honest reporting of failures
5. **Visualization-heavy**: Most scripts generate PNG outputs showing experimental results

## Working with This Codebase

- Scripts are **standalone** - run directly, no imports between experiment files
- Each script is **self-contained** with its own parameter configuration at the top
- Output files (PNG images) are saved to current directory
- Random seeds are set for reproducibility where critical (usually `np.random.seed(42)`)
- The `QA_Engine`/`QASystem` class is re-implemented in each file with domain-specific variations - there is no shared module
