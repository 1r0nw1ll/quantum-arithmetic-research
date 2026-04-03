---
layout: default
title: Demos
---

# Interactive Demos

## Quick Demo (60 seconds)

```bash
git clone https://github.com/1r0nw1ll/quantum-arithmetic-research.git
cd quantum-arithmetic-research
pip install numpy scipy scikit-learn pandas
PYTHONPATH=qa_observer:. python -m qa_observer.demo
```

This shows:

1. **Orbit Structure** -- 576 state pairs in mod-24 self-organize into 3 families:
   - **Cosmos** (552 pairs, period 24) -- the full Fibonacci cycle
   - **Satellite** (23 pairs, period 8) -- 3D sub-orbits
   - **Singularity** (1 pair, period 1) -- the fixed point at (24,24)

2. **QCI Coherence Detection** -- the T-operator predicts structured sequences with 100% accuracy while scoring random noise at chance (4.2%). This is the core of the QA Coherence Index.

3. **Chromogeometry** -- for any direction vector (d, e), the three quadrances C=2de, F=d²-e², G=d²+e² satisfy C²+F²=G². This is Wildberger's Theorem 6 -- QA restricts it to Fibonacci integer vectors.

---

## qa-observer API

The `qa-observer` package provides the full QCI pipeline:

```python
from qa_observer import TopographicObserver, QCI, SurrogateTest

# TopographicObserver: full pipeline from raw multi-channel data to QCI
obs = TopographicObserver(modulus=24, n_clusters=6, qci_window=63)
obs.fit(train_data)          # train_data: (n_samples, n_channels)
qci = obs.transform(data)   # returns QCI time series
result = obs.evaluate(data, target, lagged_control)

# QCI: compute coherence from pre-classified label sequence
qci = QCI(modulus=24, window=63)
coherence = qci.compute(labels)

# SurrogateTest: statistical validation against shuffled nulls
st = SurrogateTest(n_surrogates=200)
p_value = st.test(obs, data, target)
```

### Core Functions

```python
from qa_observer import qa_step, qa_mod, orbit_family

# A1-compliant modular reduction: result in {1,...,m}, never 0
qa_mod(25, 24)  # returns 1

# One T-operator step: (b, e) -> (e, b+e mod m)
qa_step(1, 1, 24)  # returns (1, 2)

# Orbit classification
orbit_family(1, 1, 24)   # "cosmos"
orbit_family(8, 8, 24)   # "satellite"
orbit_family(24, 24, 24) # "singularity"
```

---

## Domain-Specific Experiments

Each experiment is a standalone script. See [Running Experiments](experiments/RUNNING_EXPERIMENTS) for the full list.

**Signal Processing:**
```bash
python run_signal_experiments_final.py
# Output: signal_classification_results.png
```

**EEG Seizure Detection (CHB-MIT):**
```bash
pip install pyedflib
python eeg_chbmit_scale.py
# Output: per-patient QCI evaluation table
```

**Graph Community Detection:**
```bash
cd qa_lab
python -m pytest qa_graph/tests -v   # 12 tests
python qa_graph/pim_graph_integration.py  # full PIM+Graph pipeline
```

**QA Lab CLI:**
```bash
cd qa_lab && pip install -e .
qa status       # pipeline health check
qa dispatch     # route tasks to agents
qa meta         # run 186-family meta-validator
```

---

[Back to home](/)
