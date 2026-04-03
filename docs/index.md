---
layout: default
title: Quantum Arithmetic Research
---

# Quantum Arithmetic

**A discrete modular framework that finds structure where continuous methods see noise.**

Quantum Arithmetic operates on pairs `(b, e)` in `{1,...,N}` under a Fibonacci-like shift operator. The resulting orbits, invariants, and coherence measures add predictive power beyond standard features in 6+ empirical domains.

---

## Try It

```bash
git clone https://github.com/1r0nw1ll/quantum-arithmetic-research.git
cd quantum-arithmetic-research
pip install numpy scipy scikit-learn pandas
PYTHONPATH=qa_observer:. python -m qa_observer.demo
```

The demo shows three things: orbit structure (576 pairs self-organize into 3 families), coherence detection (T-operator scores structured sequences at 1.0, noise at chance), and chromogeometry (Pythagorean triples from Fibonacci directions via Wildberger's theorem).

---

## Empirical Results

QCI (QA Coherence Index) adds predictive power *on top of* the best conventional predictor in each domain:

| Domain | What QCI Adds | Key Result |
|--------|--------------|------------|
| EEG seizure detection | Topographic orbit state beyond delta power | +0.21 R², 10/10 patients, p < 10⁻³³ |
| EMG pathology | Motor unit recruitment structure | +0.61 R², strongest single-domain lift |
| Climate teleconnection | ENSO orbit classification | La Nina 97% satellite, p < 10⁻⁶ |
| Financial volatility | T-operator coherence predicts vol | Partial r = -0.22 beyond realized vol |
| Audio classification | Orbit transition rates | Partial r = +0.75, p = 0.020 |
| Atmospheric reanalysis | QCI vs future variability | Partial r = -0.20, p < 10⁻⁵ |

---

## 186 Certificate Families

Every empirical claim is backed by a machine-verifiable certificate: schema, validator, witnesses, and documentation. The meta-validator runs in CI — 186/186 PASS.

Families span: Pythagorean triples, E8 alignment, chromogeometry, Fibonacci resonance, Keely triune streams, megalithic geometry, planetary quantum numbers, graph community detection, and more.

[Browse certificate families](certificates) | [View validator source](https://github.com/1r0nw1ll/quantum-arithmetic-research/tree/main/qa_alphageometry_ptolemy)

---

## Explore

- [**Demos**](demos) -- Interactive demos and the qa_observer API
- [**Certificate Families**](certificates) -- 186 machine-verifiable claims
- [**Running Experiments**](experiments/RUNNING_EXPERIMENTS) -- How to run every experiment
- [**Getting Started**](quickstart) -- Full setup and installation guide
- [**Specifications**](specs/PROJECT_SPEC) -- Technical spec and architecture

---

## Core Packages

**qa-observer** -- Domain-general coherence measurement

```python
from qa_observer import TopographicObserver

obs = TopographicObserver(modulus=24, n_clusters=6, qci_window=63)
obs.fit(train_data)          # (n_samples, n_channels)
qci = obs.transform(data)   # QCI time series
result = obs.evaluate(data, target, lagged_control)
```

**qa-lab** -- Research infrastructure with CLI

```bash
cd qa_lab && pip install -e .
qa status       # pipeline health
qa dispatch     # route tasks to agents
qa meta         # run meta-validator
```

---

## Links

- [GitHub Repository](https://github.com/1r0nw1ll/quantum-arithmetic-research)
- [v3.0.0 Release](https://github.com/1r0nw1ll/quantum-arithmetic-research/releases/tag/v3.0.0)
- [CHANGELOG](https://github.com/1r0nw1ll/quantum-arithmetic-research/blob/main/CHANGELOG.md)

---

*The arithmetic of geometry, applied.*
