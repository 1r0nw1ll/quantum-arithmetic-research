<div align="center">

# Quantum Arithmetic

**A discrete modular framework that finds structure where continuous methods see noise.**

[![QA CI](https://github.com/1r0nw1ll/quantum-arithmetic-research/actions/workflows/qa-ci.yml/badge.svg)](https://github.com/1r0nw1ll/quantum-arithmetic-research/actions/workflows/qa-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Families](https://img.shields.io/badge/cert%20families-186-orange.svg)](qa_alphageometry_ptolemy/)

</div>

---

## Try it in 60 seconds

```bash
git clone https://github.com/1r0nw1ll/quantum-arithmetic-research.git
cd quantum-arithmetic-research
pip install numpy scipy scikit-learn pandas
PYTHONPATH=qa_observer:. python -m qa_observer.demo
```

You'll see three things:
1. **Orbit structure** -- 576 state pairs self-organize into 3 orbit families (Cosmos, Satellite, Singularity)
2. **Coherence detection** -- the T-operator predicts structured sequences perfectly while scoring noise at chance
3. **Chromogeometry** -- Pythagorean triples emerge from Fibonacci direction vectors via Wildberger's theorem

---

## What is QA?

Quantum Arithmetic operates on pairs `(b, e)` in `{1,...,N}` under a Fibonacci-like shift:

```
d = b + e  (mod N)      a = b + 2e  (mod N)
```

This generates orbits with period 1, 8, or 24 (mod 24). The 16 derived invariants -- quadrances, products, sums -- turn out to be the **chromogeometric quantities** of rational trigonometry (Wildberger). The key identity `C^2 + F^2 = G^2` is Wildberger's Theorem 6 restricted to integer direction vectors.

The **QA Coherence Index (QCI)** measures how well a time series follows the T-operator prediction. High QCI = structured dynamics. Low QCI = noise. This single number adds predictive power beyond standard features in 6+ domains.

---

## What You Can Build

| Domain | What QCI Adds | Key Result |
|--------|--------------|------------|
| **EEG seizure detection** | Topographic orbit state beyond delta power | +0.21 R^2, 10/10 patients, p < 10^-33 |
| **EMG pathology** | Motor unit recruitment structure | +0.61 R^2, strongest single-domain lift |
| **Climate teleconnection** | ENSO orbit classification | La Nina 97% satellite, p < 10^-6 |
| **Financial volatility** | T-operator coherence predicts vol | Partial r = -0.22 beyond realized vol |
| **Audio classification** | Orbit transition rates | Partial r = +0.75, p = 0.020 beyond lag-1 AC |
| **Atmospheric reanalysis** | QCI vs future variability | Partial r = -0.20, p < 10^-5 |

Every result includes partial correlations controlling for the best conventional predictor. QA adds *on top of* existing methods, never replaces them.

---

## 186 Certificate Families

Every empirical claim is backed by a **machine-verifiable certificate**: schema, validator, witnesses, and documentation. The meta-validator runs in CI.

```bash
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
# 186/186 PASS
```

Families span: Pythagorean triples, E8 alignment, chromogeometry, Fibonacci resonance, Keely triune streams, megalithic geometry, planetary quantum numbers, graph community detection, and more.

Browse: [`qa_alphageometry_ptolemy/`](qa_alphageometry_ptolemy/) | Docs: [`docs/families/README.md`](docs/families/README.md)

---

## Repository Layout

```
qa_observer/          pip-installable QCI toolkit (TopographicObserver, QCI, SurrogateTest)
qa_lab/               Research lab: agents, PIM kernels, graph features, Rust acceleration
qa_alphageometry_ptolemy/   186 certificate families with validators
papers/               Research papers (published to this repo)
tools/                QA axiom linter, security audit, LaTeX claim linter
docs/                 Digital garden (GitHub Pages)
```

---

## Core Packages

**qa-observer** -- domain-general coherence measurement
```python
from qa_observer import TopographicObserver

obs = TopographicObserver(modulus=24, n_clusters=6, qci_window=63)
obs.fit(train_data)          # (n_samples, n_channels)
qci = obs.transform(data)   # QCI time series
result = obs.evaluate(data, target, lagged_control)
```

**qa-lab** -- research infrastructure
```bash
cd qa_lab && pip install -e .
qa status       # pipeline health
qa dispatch     # route tasks to agents
qa meta         # run meta-validator
```

**qa_pim** / **qa_graph** -- PIM kernels and graph feature extraction (24 + 12 tests)

---

## Axiom Compliance

Six non-negotiable axioms enforce discrete integrity. The linter runs as a pre-commit hook and in CI:

```bash
python tools/qa_axiom_linter.py --all    # 0 errors required
```

A1 (No-Zero), A2 (Derived Coords), T2 (Firewall), S1 (No `**2`), S2 (No float state), T1 (Path Time). See [`QA_AXIOMS_BLOCK.md`](QA_AXIOMS_BLOCK.md).

---

## Installation

```bash
# Core (runs everything)
pip install numpy scipy scikit-learn pandas matplotlib

# EEG experiments
pip install pyedflib

# Full dev environment
pip install -r requirements-dev.txt
```

Requires Python 3.9+. Tested on 3.11 and 3.13.

---

## Citation

```bibtex
@software{dale2026qa,
  author = {Dale, Will},
  title  = {Quantum Arithmetic: Modular Framework for Structural Coherence},
  year   = {2026},
  url    = {https://github.com/1r0nw1ll/quantum-arithmetic-research}
}
```

---

## License

MIT. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for attribution requirements on certificate families.

---

<div align="center">
<b>The arithmetic of geometry, applied.</b>
</div>
