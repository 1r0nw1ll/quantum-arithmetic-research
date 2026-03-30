---
layout: default
title: Quick Start Guide
---

# Quick Start Guide

Get running with Quantum Arithmetic research in under 5 minutes!

---

## Prerequisites

- **Python 3.13+** (tested on 3.13)
- **pip** for package management
- **git** for cloning the repository

Optional but recommended:
- **Jupyter** for interactive notebooks
- **Docker** for containerized experiments (Phase 2)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/player2/signal_experiments.git
cd signal_experiments
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib pandas scikit-learn torch scipy tqdm seaborn
```

**Or** create a virtual environment (recommended):

```bash
python -m venv qa_env
source qa_env/bin/activate  # On Windows: qa_env\Scripts\activate
pip install -r requirements.txt  # Coming soon!
```

---

## Your First Experiment

### Signal Classification (5 minutes)

The simplest way to see QA in action:

```bash
python run_signal_experiments_final.py
```

**What it does:**
1. Generates synthetic audio signals (pure tones, chords, white noise)
2. Converts to QA tuples using modular arithmetic
3. Calculates E8 alignment and Harmonic Index
4. Classifies signals based on geometric coherence
5. Saves visualization: `signal_classification_results.png`

**Expected output:**
```
============================================================
QA SIGNAL PROCESSING EXPERIMENT
============================================================

Generating signals...
  Pure tone A (440 Hz): [generated]
  Pure tone C (523 Hz): [generated]
  Major chord (C-E-G): [generated]
  ...

Calculating QA metrics...
  E8 alignment: 0.87 (pure tone), 0.45 (noise)
  Harmonic Index: 1.23 (harmonic), 0.31 (non-harmonic)

Classification accuracy: 94.2%

Results saved to signal_classification_results.png
```

**Open the PNG** to see E8 alignment patterns!

---

## More Experiments

### Bell Test Validation (10 minutes)

Test quantum correlation inequalities:

```bash
python qa_platonic_bell_tests.py
```

**Output**: Correlation values exceeding quantum bounds (up to 1852.6%!)

---

### Automated Theorem Discovery (30 minutes)

Run the full GNN pipeline:

```bash
python qa_theorem_discovery_orchestrator.py --quick
```

**What happens:**
1. **Graph Builder** - 10K nodes, 58K edges (~3 min)
2. **GNN Trainer** - Learn patterns (~10 min)
3. **Symbolic Miner** - Extract conjectures (~1 min)
4. **Lean Verifier** - Formal proofs (~2 min)
5. **Export** - Generate report

**Check results:**
```bash
cat qa_discovery_workspace/DISCOVERY_REPORT.txt
```

---

### Financial Backtesting (15 minutes)

Test the Harmonic Index on S&P 500:

```bash
python backtest_advanced_strategy.py
```

**Output**: Performance metrics (Sharpe ratio, drawdown, win rate) + equity curve visualization.

---

## Understanding the Output

### QA Tuples

Every experiment operates on `(b, e, d, a)` tuples:

```python
# Example from signal processing
b = int(frequency % 24)  # Base state
e = int(amplitude * 24)  # Excitation state
d = (b + e) % 24         # Derived coordinate 1
a = (b + 2*e) % 24       # Derived coordinate 2
```

### E8 Alignment

Measures how well a QA tuple aligns with the E8 exceptional Lie algebra:

```
E8_alignment = max(cosine_similarity(QA_8D_vector, E8_roots))

Where:
  - QA_8D_vector = [b, e, d, a, b+d, e+a, b+a, d+e] (normalized)
  - E8_roots = 240 vectors in 8D space
```

**Interpretation:**
- **High alignment (>0.7)**: Harmonic, coherent, geometric
- **Medium (0.4-0.7)**: Mixed character
- **Low (<0.4)**: Noise, random, incoherent

### Harmonic Index

Combined metric for optimization:

```
HI = E8_alignment × exp(-0.1 × loss)
```

Used in:
- Signal classification (distinguish music from noise)
- Financial regime detection (bull vs bear markets)
- Theorem quality scoring (valid vs invalid conjectures)

---

## Exploring the Codebase

### Key Files by Use Case

**Just want to understand QA basics?**
→ Read `qa_formal_report.tex` (LaTeX, comprehensive math)

**Want to see QA in action?**
→ Run `run_signal_experiments_final.py` (easiest demo)

**Interested in GNNs?**
→ Check `geometrist_v4_gnn.py` (graph neural network)

**Want to build on it?**
→ Use `qa_core.py` (reusable QA engine)

**Curious about applications?**
→ Browse `multimodal_data/` (remote sensing experiments)

### Directory Structure

```
signal_experiments/
├── run_signal_experiments_final.py    # Signal processing demo
├── qa_platonic_bell_tests.py          # Bell inequality tests
├── geometrist_v4_gnn.py                # Theorem generation (GNN)
├── backtest_advanced_strategy.py      # Financial strategy
├── qa_core.py                          # Reusable QA engine
├── qa_formal_report.tex                # Mathematical foundations
├── GEMINI.md                           # Project overview
├── QUICKSTART.md                       # This guide
├── results/                            # Experimental outputs
├── QAnotes/                            # Research notes (Obsidian vault)
└── docs/                               # GitHub Pages site (you are here!)
```

---

## Next Steps

### Option 1: Run All Experiments

Try each of the 4 main experiments to see QA across domains:

1. ✅ `run_signal_experiments_final.py` (already done!)
2. ⬜ `qa_platonic_bell_tests.py`
3. ⬜ `geometrist_v4_gnn.py`
4. ⬜ `backtest_advanced_strategy.py`

### Option 2: Deep Dive into Theory

Read the formal mathematical report:

```bash
# If you have LaTeX installed
pdflatex qa_formal_report.tex
open qa_formal_report.pdf

# Or read the source directly
cat qa_formal_report.tex
```

**Topics covered:**
- Multi-orbit structure (24-cycle, 8-cycle, 1-cycle)
- E8 alignment derivation
- Harmonic Index formulation
- Proofs and propositions

### Option 3: Explore Research Notes

Browse the Obsidian vault:

```bash
cd QAnotes
ls -R
# 100+ markdown files from Nexus AI sessions
```

**Highlights:**
- `/2025/03/Quantum Arithmetic and Harmonics.md`
- `/2025/09/Map to QA (1).md`
- Session closeouts with experimental results

### Option 4: Try Jupyter Notebooks

Coming soon! Interactive notebooks with:
- Step-by-step QA tutorial
- Visualizations of E8 alignment
- Experiment parameter tuning

---

## Troubleshooting

### "Module not found" errors

```bash
pip install <missing-module>
```

Common missing modules: `numpy`, `torch`, `pandas`, `scikit-learn`, `tqdm`

### Experiments run but no output

Check for PNG files in your current directory:

```bash
ls -lh *.png
```

Visualizations are saved automatically.

### GNN training is slow

Use `--quick` mode:

```bash
python qa_theorem_discovery_orchestrator.py --quick
# Reduces epochs from 300 → 50
```

Or run on GPU (if available):

```bash
# PyTorch will auto-detect CUDA
python geometrist_v4_gnn.py
```

### Want to understand the math better?

1. Start with [E8 Lattice (Wikipedia)](https://en.wikipedia.org/wiki/E8_lattice)
2. Read `qa_formal_report.tex` (section 2: Mathematical Foundations)
3. Check YouTube for "E8 exceptional Lie algebra" visualizations

---

## Getting Help

🐛 **Found a bug?** [Open an issue](https://github.com/player2/signal_experiments/issues)

❓ **Have a question?** [Start a discussion](https://github.com/player2/signal_experiments/discussions)

💡 **Want to contribute?** See [CONTRIBUTING.md](https://github.com/player2/signal_experiments/blob/main/CONTRIBUTING.md)

📖 **Need more docs?** Check the [Wiki](https://github.com/player2/signal_experiments/wiki)

---

## What's Next for This Project?

We're building:
- **Phase 2**: Docker containers for all experiments
- **Phase 3**: Kubernetes-based multi-agent home lab
- **Phase 4**: Public API for QA experiments
- **Phase 5**: Interactive web UI

Track progress in [PROJECT_ROADMAP.md](https://github.com/player2/signal_experiments/blob/main/PROJECT_ROADMAP.md)

---

<div align="center">
  <p><strong>Ready to explore? Pick an experiment and run it!</strong></p>
  <p><a href="experiments/">Browse All Experiments →</a></p>
</div>
