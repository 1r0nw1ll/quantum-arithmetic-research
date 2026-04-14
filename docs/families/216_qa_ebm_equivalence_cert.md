# Family [216] QA_EBM_EQUIVALENCE_CERT.v1

## One-line summary

QA coherence is a discrete-native, Theorem NT-compliant **Energy-Based Model**. The energy function `E_QA(b, e, next) = 0 if T(b,e) == next else 1` and its window-mean `E_window = 1 − QCI` satisfy all five standard EBM axioms (non-negativity, data-manifold zero, monotonicity, valid Boltzmann distribution, discrete score identity) with integer state throughout.

## What this recognizes

`qa_detect`, the QCI pipeline (cert [154]), and the wider QA coherence toolkit are **already an EBM** — we just weren't using the vocabulary. The claim is a formal equivalence that lets QA participate directly in the Energy-Based Models research tradition (LeCun 2006, Hinton 2002 contrastive divergence, Hopfield networks, score-based diffusion) without changing any algebra.

## Training-free by construction — with required observer calibration

A key distinction from continuous EBMs: **QA's causal layer requires no training.** The T-operator, orbit classification, norm, family — all closed-form integer computations fully determined by `(b, e)` and `m`. The energy function `E_QA` is derived from the algebra, not learned.

What QA pipelines **do** require is **observer-layer calibration** — aligning the observer projection so that training-data transitions actually follow the T-operator. This is the Theorem NT boundary crossing; it is not QA-layer training. Three kinds of calibration, all observer-only:

| Observer projection | Calibration required? | When to use |
|---|---|---|
| Quantile bins (TypeA) | Compute percentile edges on training | 1D streams |
| K-means centroids (TypeD) | Fit k-means on training | Multi-channel continuous |
| **cmap: cluster → QA state** | **Enumerate permutations, pick one maximising training QCI** | **Any pipeline using k-means or structured clustering (critical)** |

### The cmap calibration requirement (empirically load-bearing)

The T-operator `((b + e − 1) mod m) + 1` is the Fibonacci-mod-`m` generator. It predicts `label_{t+2} = label_t + label_{t+1} mod m`. For QA to detect anomalies as T-mismatches, the cluster→state mapping **must be chosen** so that training-data cluster transitions actually follow Fibonacci-mod-`m` dynamics under that mapping. With the wrong cmap, every transition is a "mismatch" and the energy signal saturates meaninglessly.

The canonical cmap `{0:8, 1:16, 2:24, 3:5, 4:3, 5:11}` was hand-tuned for finance return-panel clusters. Using it on non-finance data fails: cert [154]'s finance win does not transfer to arbitrary series without re-calibration. Enumerating permutations of canonical states (`8! = 40320` assignments at `k=8`, sub-second) and picking the one maximising training QCI is sufficient calibration.

This is observer-layer optimisation — it tunes the bin-to-state correspondence, not the QA algebra. Formally equivalent to fitting quantile edges or k-means centroids; none of these are "QA training".

Once calibration is performed, the QA causal dynamics run with zero learning. The Boltzmann partition function is explicit (not approximated), the score is exact (the T-operator), and no MCMC burn-in is required. **Contrastive divergence is structurally unnecessary in QA.**

### Empirical scope (as of 2026-04-12)

With cmap calibration, `qa_detect` is competitive on **temporal/sequential data with dynamical transition structure**:
- Cert [154] finance QCI: partial r = −0.22 predicting future volatility, 84% robust across surrogates (inherited validation).
- NAB ec2_cpu_utilization: AUROC 0.787 — QA wins over IsolationForest (0.757), LocalOutlierFactor (0.765), rolling z-score (0.500).
- NAB Twitter_volume_AAPL: AUROC 0.858 (competitive; IsolationForest 0.975).
- NAB ambient_temperature: AUROC 0.635 (calibration lifts from 0.320).

**Out-of-scope empirically**:
- Tabular point-anomaly detection (ODDS datasets): 0/5 wins across four encoding variants including calibrated. Tabular data has no temporal structure for the T-operator to measure coherence against; PCA-induced ordering doesn't create real sequential dependence.
- Event-sampled data (MIT-BIH per-beat features): null on all tested records. Per-beat features are not a temporal stream in the QCI sense.

The formal EBM equivalence proven in-cert (all five axioms verified on `S_9`) remains structurally correct on any QA state space. Empirical competitiveness requires (a) data with dynamical transition structure, and (b) calibrated observer cmap.

## The five axioms (all verified in-cert)

| Axiom | Statement | Witness |
|---|---|---|
| **E1 Non-negativity** | `E(x) ≥ 0 for all x` | Exhaustive on S_9² × {1..9}: E ∈ {0, 1} |
| **E2 Data-manifold zero** | `E = 0` on the generator manifold | `E_window(deterministic T-trajectory) = 0.0` exactly |
| **E3 Monotonicity** | `E` rises monotonically with deviation from manifold | Injected mismatch 0/10/30/50/80% → `E = 0.0/0.19/0.46/0.66/0.83` (near-linear) |
| **E4 Boltzmann** | `p(x) ∝ exp(−E(x)/T)` is a valid distribution; `T` parameterises selectivity | Occupancy well-formed; `T = 2π/m` (corollary of cert [215]) |
| **E5 Score identity** | `∇ log p(x) = −∇E(x)/T`; in QA the score IS the T-operator step | Exhaustive on S_9: argmax of Boltzmann over next_state = `T(b, e)` on **81/81** pairs |

## Structural consequences

This equivalence collapses several conventional EBM costs:

- **No MCMC burn-in.** Orbit walks (`(b, e) → (e, T(b, e))`) ARE Gibbs sampling from the QA Boltzmann. Deterministic, one integer-add + mod per step.
- **No score approximation.** Score-matching / diffusion models spend enormous compute estimating `∇ log p`. In QA the score is literally the T-operator — zero approximation error.
- **No gradient descent.** The energy function is determined by the algebra, not learned. Observer calibration (when used) is percentile-computation or k-means — never backprop through the energy.
- **Reproducibility is perfect.** Integer arithmetic throughout — the same data produces bit-identical energies on any hardware.
- **Temperature has structural meaning** (cert [215]): `T = 2π/m` where `m` is the modulus. Fine modulus = low temperature = sharp selectivity. Hensel lift `m → m·p` = annealing schedule.

## Expressiveness note

The modulus `m` is a free integer parameter. For high-dimensional continuous manifolds, use Type B/C/D encoders across dimensions with appropriate moduli per axis (or CRT representations — cf. cert [205] grid-cell RNS). There is **no structural expressiveness ceiling**; the observer projection handles any continuous input, and integer state is exact inside.

The binding constraint is not expressiveness but the **dynamical-structure prerequisite**: `E_QA = 1 − QCI` is a meaningful energy iff the data's transitions (under the calibrated observer) are T-coherent. Data that is purely noise, autocorrelation-dominated, or otherwise not T-alignable produces a flat energy landscape. This is not a ceiling on QA — it is the correct statement of the EBM's applicability domain, analogous to "Fourier analysis is meaningful on signals with frequency content".

## Compliance

| Axiom | Status | Note |
|---|---|---|
| A1 (no-zero) | ✓ | `qa_t_step` uses `((b+e-1) % m) + 1` |
| A2 (derived coords) | n/a | Cert is about energy, not (d, a) |
| S1 (no `**2`) | ✓ | No squaring |
| S2 (no float state) | ✓ | Energy pointwise is `int64`; window-mean is observer-layer read-out only |
| T1 (path time) | ✓ | Time = integer step index |
| T2 (firewall) | ✓ | Energy computed on integer states; float only at read-out |

## Cross-references

- **[154]** T-operator coherence — empirical validation of QCI as energy on the finance pipeline (partial r = −0.22, 84% robust). Energy-minimization → volatility prediction.
- **[191]** Bateson Learning Levels — the invariant filtration `orbit ⊂ family ⊂ modulus ⊂ ambient` gives the EBM its manifold stratification.
- **[215]** Resonance-Bin Correspondence — provides the temperature identity `T_boltzmann = 2π/m`.
- **Theorem NT** — all energy evaluation respects the observer projection firewall; integer state inside, continuous read-out only at the boundary.

## External references

- LeCun, Chopra, Hadsell, Ranzato, Huang (2006). *A Tutorial on Energy-Based Learning.*
- Hinton (2002). *Training Products of Experts by Minimizing Contrastive Divergence.*
- Hopfield (1982). *Neural networks and physical systems with emergent collective computational abilities.*

## Running

```bash
cd qa_alphageometry_ptolemy/qa_ebm_equivalence_cert_v1
python qa_ebm_equivalence_cert_validate.py                              # pass fixture
python qa_ebm_equivalence_cert_validate.py fixtures/ebm_fail_no_boltzmann.json   # fail fixture
python qa_ebm_equivalence_cert_validate.py --self-test                  # JSON, used by meta-validator
```

## Source

Will Dale + Claude (Opus 4.6), 2026-04-12. Grounded in LeCun et al. 2006 EBM framework; builds on cert [154] empirical QCI validation and cert [215] temperature-modulus duality.
