# Quantum Arithmetic Research

[![QA CI](https://github.com/1r0nw1ll/quantum-arithmetic-research/actions/workflows/qa-ci.yml/badge.svg)](https://github.com/1r0nw1ll/quantum-arithmetic-research/actions/workflows/qa-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-❤-green.svg)](https://github.com/player2/signal_experiments)

**Open-source research on modular arithmetic frameworks with applications in signal processing, geometric theorem discovery, and quantum physics.**

🌐 **Digital Garden**: [https://player2.github.io/signal_experiments/](https://player2.github.io/signal_experiments/)
📖 **Wiki**: [https://github.com/player2/signal_experiments/wiki](https://github.com/player2/signal_experiments/wiki)

---

## VERIFY OR BREAK

This is a closed-world, executable challenge with a binary outcome.

- G0 -> NONREACH
- G1 -> REACH

How to run:

```
make test
```

Links:
- Appendix A: [APPENDIX_A_GENERATOR_BARRIER.md](APPENDIX_A_GENERATOR_BARRIER.md)
- [REBUTTAL.md](REBUTTAL.md)
- [TESTING.md](TESTING.md)

## What is Quantum Arithmetic?

The **QA System** is a modular arithmetic framework (mod-9, mod-24) that exhibits remarkable geometric properties:

- **Multi-orbit structure**: 24-cycle "Cosmos", 8-cycle "Satellite", 1-cycle "Singularity"
- **E8 alignment**: Strong correlation with the E8 exceptional Lie algebra root system
- **Harmonic coherence**: Emergent geometric patterns quantified via the "Harmonic Index"

### Applications

🎵 **Signal Processing**: Audio classification using harmonic resonance
🔬 **Bell Test Validation**: Quantum correlation beyond traditional bounds
📊 **Financial Markets**: Trading strategy based on geometric coherence
🧮 **Theorem Discovery**: Automated generation of geometric conjectures

---

## Quick Start

```bash
# Clone and navigate
git clone https://github.com/player2/signal_experiments.git
cd signal_experiments

# Install dependencies
pip install numpy matplotlib pandas scikit-learn torch scipy tqdm seaborn pyedflib

# Run first experiment (signal classification)
python run_signal_experiments_final.py
```

**Output**: PNG visualization showing E8 alignment and classification results.

**For detailed instructions**: See [Quick Start Guide](https://player2.github.io/signal_experiments/quickstart.html)

---

## Seizure Detection (CHB-MIT) – Quick Repro

```bash
# Optional: create a venv
python -m venv .venv && source .venv/bin/activate

# Install deps (includes pyedflib for EDF)
pip install -r requirements.txt

# Run expanded dataset evaluation (expects chb01 EDFs under phase2_data/eeg/chbmit/chb01)
python test_with_expanded_dataset.py

# Generate publication figures (confusion matrix, feature importance)
python generate_paper_figures.py
```

Expected (13D + class weights, chb01 6 files): recall ≈ 89.3%, precision ≈ 62.5%, F1 ≈ 0.735. Figures saved under `paper_figures/`.

---

## Repository Structure

```
signal_experiments/
├── docs/                          # 📖 GitHub Pages site (digital garden)
│   ├── index.md                  # Homepage
│   ├── quickstart.md             # Getting started guide
│   ├── experiments/              # Experiment documentation
│   └── assets/images/            # Visualizations
│
├── private/                       # 🔒 IP-protected content (gitignored)
│   ├── patents/                  # Patent applications
│   ├── funding/                  # Grant proposals
│   └── papers/                   # Pre-publication manuscripts
│
├── public/                        # ✅ Coming soon: organized open-source experiments
│
├── QAnotes/                       # Research notes (Obsidian vault)
├── qa_lab/                        # QALM 2.0 training infrastructure
├── multimodal_data/               # Remote sensing datasets
├── results/                       # Experimental outputs
│
├── run_signal_experiments_final.py  # Signal processing demo
├── qa_platonic_bell_tests.py        # Bell inequality tests
├── geometrist_v4_gnn.py              # GNN theorem generation
├── backtest_advanced_strategy.py    # Financial strategy
├── qa_core.py                        # Reusable QA engine
│
├── GEMINI.md                      # Project overview
├── QUICKSTART.md                  # Quick start (CLI-focused)
└── CLAUDE.md                      # AI agent instructions
```

### Private vs Public Content

This repository uses a **hybrid open-source model**:

- **✅ Public (MIT License)**: All code, experiments, research findings, documentation
- **🔒 Private (IP Protection)**: Patent applications, funding proposals, pre-publication papers

The `private/` directory is **gitignored** and never pushed to GitHub. This protects intellectual property while keeping the research open.

---

## Key Experiments

### 1. Signal Classification
**File**: `run_signal_experiments_final.py`

Classifies audio signals using QA harmonic analysis. Achieves >90% accuracy distinguishing chords from noise.

```bash
python run_signal_experiments_final.py
# Output: signal_classification_results.png
```

---

### 2. Bell Test Validation
**File**: `qa_platonic_bell_tests.py`

Tests quantum correlation inequalities on Platonic solid geometries.

**Key Result**: **1852.6% of quantum bound** on dodecahedron.

```bash
python qa_platonic_bell_tests.py
# Output: qa_platonic_solids_bell_tests.png
```

---

### 3. Automated Theorem Discovery
**File**: `qa_theorem_discovery_orchestrator.py`

Graph Neural Network trained to generate geometric theorems.

```bash
python qa_theorem_discovery_orchestrator.py --quick
# Output: qa_discovery_workspace/DISCOVERY_REPORT.txt
```

**Pipeline**: Graph Builder → GNN Trainer → Symbolic Miner → Lean Verifier → Report

---

### 4. Financial Backtesting
**File**: `backtest_advanced_strategy.py`

Trading strategy using Harmonic Index for regime detection.

```bash
python backtest_advanced_strategy.py
# Output: backtest_equity_curve.png
```

---

## Recent Additions (November 2025)

### Quartz Piezoelectric System

New theoretical framework for helium-doped quartz self-oscillating energy generation:

- **Files**: `quartz_quantum_phonon_coupling.py`, `quartz_qa_integration.py`, `quartz_piezo_tensor_viz.py`
- **Documentation**: `QUARTZ_EXPERIMENTAL_VALIDATION.md`, `QUARTZ_PROJECT_SUMMARY.md`
- **Visualizations**: 9 publication-quality PNG images in `docs/assets/images/`

**Key Prediction**: 0.01 - 40 W/cm³ power density from passive piezoelectric generation.

---

## Documentation

- **Digital Garden**: [https://player2.github.io/signal_experiments/](https://player2.github.io/signal_experiments/)
- **Wiki**: [Full documentation](https://github.com/player2/signal_experiments/wiki)
- **Certificate Families**: [`docs/families/README.md`](docs/families/README.md) — per-family docs for [18]-[28], enforced by meta-validator doc gate
- **External Validation**: [`docs/external_validation/`](docs/external_validation/) — Level-3 recompute on raw MNIST + prompt-injection benchmark checks; enforced as meta-validator tests [29]-[30]
- **Formal Report**: `qa_formal_report.tex` (mathematical foundations)
- **Research Notes**: `QAnotes/` (100+ markdown files)

---

## Multi-Agent Home Lab (In Development)

We're building a **production Kubernetes-based multi-agent research infrastructure**:

### Architecture
- **6 Autonomous Agents**: Orchestrator, Graph Builder, GNN Trainer, Symbolic Miner, Lean Verifier, Export Agent
- **Orchestration**: Kubernetes (K3s/K8s)
- **Monitoring**: Prometheus + Grafana + Loki
- **Communication**: Model Context Protocol (MCP) + gRPC + Redis
- **CI/CD**: GitHub Actions + ArgoCD

### Timeline
- **Phase 1 (Weeks 1-2)**: Digital garden setup ✅ *In Progress*
- **Phase 2 (Weeks 3-6)**: Docker foundation
- **Phase 3 (Weeks 7-12)**: Kubernetes cluster
- **Phase 4 (Weeks 13-18)**: Agent implementation
- **Phase 5 (Weeks 19-21)**: Monitoring & observability
- **Phase 6 (Weeks 22-24)**: CI/CD pipeline

**See full plan**: [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) (coming soon)

---

## Contributing

We welcome contributions! Areas of interest:

- **New Experiments**: Apply QA to biology, materials science, cryptography
- **Optimization**: GPU acceleration, algorithm improvements
- **Visualization**: Interactive web demos, 3D displays
- **Documentation**: Tutorials, videos, blog posts

**Two-tract rule**: Every certificate family must ship with both a machine tract (schema/validator/certs) and a human tract (`docs/families/[NN]_*.md`). The meta-validator test [25] enforces this — CI fails if docs are missing. See [`docs/families/README.md`](docs/families/README.md) for the checklist.

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for full guidelines.

---

## Installation (Detailed)

### Prerequisites
- Python 3.13+ (tested on 3.13)
- pip package manager
- git version control

### Dependencies

**Core**:
```bash
pip install numpy scipy matplotlib pandas
```

**Machine Learning**:
```bash
pip install torch scikit-learn
```

**Visualization**:
```bash
pip install seaborn tqdm
```

**Financial Data**:
```bash
pip install yfinance
```

**Topological Analysis** (optional):
```bash
pip install ripser persim
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/player2/signal_experiments.git
cd signal_experiments

# Create virtual environment
python -m venv qa_env
source qa_env/bin/activate  # On Windows: qa_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt  # Coming soon!

# Verify installation
python -c "import numpy, torch, pandas; print('✅ All imports successful')"
```

---

## License

This research is released under the **MIT License** - free for academic and commercial use.

**Citation**:
```bibtex
@software{qa_research_2025,
  author = {QA Research Lab},
  title = {Quantum Arithmetic System: Modular Framework for Geometric Coherence},
  year = {2025},
  url = {https://github.com/player2/signal_experiments}
}
```

---

## Community & Support

🐛 **Issues**: [Report bugs](https://github.com/player2/signal_experiments/issues)
💬 **Discussions**: [Ask questions](https://github.com/player2/signal_experiments/discussions)
🐦 **Updates**: [@QA_Research](https://twitter.com/QA_Research) (coming soon)
📧 **Contact**: See [AUTHORS.md](AUTHORS.md)

---

## Acknowledgments

This research builds on:
- Exceptional Lie algebras (E8)
- Universal Hyperbolic Geometry
- Modular arithmetic and number theory
- Graph neural networks

Thanks to the open-source community: NumPy, PyTorch, scikit-learn, matplotlib.

---

## Project Status

**Current Focus**: Phase 1 - Digital Garden Setup

✅ Private content secured (`private/` directory created)
✅ GitHub Pages site structure created
✅ Wiki content prepared
⬜ Experiment documentation pages
⬜ Interactive Jupyter notebooks
⬜ Docker containerization (Phase 2)

**Track Progress**: [GitHub Projects](https://github.com/player2/signal_experiments/projects)

---

<div align="center">
  <p><strong>Exploring the geometry of numbers</strong></p>
  <p><a href="https://player2.github.io/signal_experiments/">Visit Digital Garden →</a></p>
</div>

---

**Last Updated**: November 2025
