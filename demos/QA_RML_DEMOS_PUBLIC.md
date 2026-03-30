# QA-RML Public Demos: Understanding ≠ Prediction

## Two Demos, One Thesis

| Demo | Domain | Data | Key Result |
|------|--------|------|------------|
| **Rule-30** | Cellular automata | Generated | Proves non-periodicity with 64+ explicit witnesses |
| **EEG Seizure** | Clinical medicine | CHB-MIT (real) | 81.7% accuracy with interpretable QA invariants |

**The thesis**: World models predict *what happens*. QA-RML certifies *why some things cannot happen* (Rule-30) and *why classifications are made* (EEG).

---

## Demo 1: Rule-30 (Theoretical Flagship)

```bash
python demos/rule30_understanding_demo.py
```

**What it shows**:
- **Prediction is trivial**: Apply local rule, O(1) per cell
- **Understanding is hard**: Prove non-periodicity requires 1024+ explicit counterexamples

**Certificate output** (`rule30_understanding_cert.json`):
- 64 derived invariants (non_periodic_p1...p64)
- Each has derivation witness: (t, c[t], c[t+p]) where c[t] ≠ c[t+p]
- Strategy: exhaustive_counterexample_search
- All claims machine-verifiable

**One-liner**: *Any simulator can generate Rule 30. Only RML can prove why certain patterns are impossible.*

---

## Demo 2: EEG Seizure Detection (Practical Flagship)

```bash
python demos/eeg_understanding_demo.py
```

**What it shows**:
- **Prediction**: "This EEG segment is seizure/baseline"
- **Understanding**: "This is SEIZURE because QA invariants E, D, B crossed thresholds 186, 208, 177 respectively"

**Certificate output** (`eeg_understanding_cert.json`):
- 6 derived invariants (thresholds + effect sizes)
- Each has derivation witness (midpoint_threshold, cohens_d)
- Strategy: threshold_voting on top-3 discriminative invariants
- Accuracy: 81.7%, Recall: 68.6%

**One-liner**: *Any classifier can label seizures. Only RML explains which brain network features drove the decision.*

---

## The Three-Layer Stack

Both demos implement the same architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: UNDERSTANDING (QA-RML)                                │
│  • Derived invariants with derivation witnesses                 │
│  • Key steps with necessity certificates                        │
│  • Strategy (not free text - has derivation)                    │
│  • Compression ratio: explanation << raw data                   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: STRUCTURE (QAWM)                                      │
│  • Reachability predicates                                      │
│  • Failure classification                                       │
│  • Invariant computation                                        │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: DYNAMICS (World Model)                                │
│  • Raw data / transitions                                       │
│  • Simulation / prediction                                      │
│  • This is what everyone else does                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Matters

### Standard ML Says:
> "My model achieves 81.7% accuracy on seizure detection"

### QA-RML Says:
> "My model achieves 81.7% accuracy on seizure detection because:
> - QA invariant E (e²) has effect size 0.92 between seizure and baseline
> - Threshold 186 was derived as midpoint between baseline mean 84 and seizure mean 288
> - This threshold was verified on 140 training samples
> - Here is the derivation witness proving this claim
> - If you remove this invariant, classification degrades to X%"

The difference is **falsifiability**. Every claim in QA-RML has a witness.

---

## Validity Rules (Hard, Not Advisory)

Both demos use `qa_understanding_cert/v2` schema with strict validation:

1. **No ad-hoc state injection**: All derived invariants must have derivation witnesses
2. **No free-text strategy**: Strategy must have derivation witness
3. **Locked compression ratio**: `trace_len / (explanation + key_steps + invariants)`
4. **QARM transition log**: Schema `qarm_transition/v1` for replayability

Violating any rule raises `CertificateValidityError`.

---

## Running the Demos

### Prerequisites
```bash
pip install numpy
# Optional for real EEG: pip install pyedflib
```

### Execution
```bash
# Rule-30 (theoretical)
python demos/rule30_understanding_demo.py

# EEG (practical)
python demos/eeg_understanding_demo.py
```

### Outputs
```
demos/
├── rule30_understanding_cert.json   # Theoretical demo certificate
├── eeg_understanding_cert.json      # Practical demo certificate
├── RULE30_FLAGSHIP_DEMO.md          # Rule-30 one-pager
└── QA_RML_DEMOS_PUBLIC.md           # This file
```

---

## Reference

- Gupta & Pruthi (2025). "Beyond World Models: Rethinking Understanding in AI Models." arXiv:2511.12239v1.
- QA-RML Certificate Schema: `qa_understanding_cert/v2`
- CHB-MIT Scalp EEG Database: PhysioNet
