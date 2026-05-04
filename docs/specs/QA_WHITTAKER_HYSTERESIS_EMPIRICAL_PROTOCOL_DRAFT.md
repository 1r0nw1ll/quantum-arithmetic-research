# QA Whittaker Hysteresis Empirical Protocol Draft

**Status**: Design draft only. No cert family ID assigned. Do not register or
build from this draft until dataset inspection and hostile review.

**Proposed slug**: `qa_whittaker_hysteresis_loop_work_empirical_cert_v1`

**Purpose**: Define the first real-world physics test for the QA/Whittaker
bridge: whether measured magnetic hysteresis loop work can be represented or
predicted by a declared QA/Whittaker feature map under fixed calibration and
held-out evaluation.

This draft is not trying to avoid physics claims. It defines a measured
physics target that can make the claim fail.

---

## 1. Physical Question

Can measured magnetic hysteresis loop work:

```text
W_h = integral_closed_loop H dB
```

be represented or predicted by a QA/Whittaker finite feature map under fixed
calibration, with no refit on held-out loops?

The empirical claim shape should be:

```text
calibrate on declared training loops
freeze QA/Whittaker calibration constants
predict held-out loop work or loop-shape diagnostics
compare against declared baselines
```

The cert should fail if the held-out prediction error does not beat or match
the predeclared baseline threshold.

---

## 2. Primary Observable

For a measured closed `B-H` cycle:

```text
W_h = integral_closed_loop H dB
```

Numerically:

```text
W_h ~= sum_i 0.5*(H_i + H_{i+1})*(B_{i+1} - B_i)
```

The sign convention must be declared. The validator should either:

- use signed loop work with a declared orientation; or
- use absolute loop area with the sign convention recorded.

No fixture may mix conventions between calibration and evaluation.

---

## 3. Required Data Schema

Minimum useful record:

```text
dataset_id
source
material_id
temperature
frequency
drive_waveform
B_series
H_series
sample_count
units
cycle_id
split = calibration | validation | test
```

Recommended metadata:

```text
core_geometry
calibration_notes
instrument_notes
dc_bias
manufacturer_material_name
waveform_family
amplitude_descriptor
```

Hard requirements:

- `B_series` and `H_series` must be time-aligned.
- `B_series` and `H_series` must have the same length.
- A record must cover one complete closed or explicitly closeable cycle.
- Units must be declared.
- Frequency and temperature must be declared or explicitly marked missing.

---

## 4. Dataset Targets

### 4.1 Primary Candidate: Princeton/Dartmouth MagNet

Treat MagNet as the first dataset candidate to inspect.

Important status note:

```text
MagNet file layout has not yet been inspected in this protocol track.
Do not build a parser, fixture schema, or ingestion validator from summary
claims alone.
```

The MagNet Challenge paper describes the training data as paired single-cycle
`B(t)` and `H(t)` time sequences with 1024 steps, different switching
frequencies, and temperatures. It also states that `B-H` loop area determines
volumetric core loss. The MagNet Challenge data comes from the
Princeton-Dartmouth MagNet project.

Candidate MagNet fields to verify during file inspection:

```text
B(t)
H(t)
frequency
temperature
material
```

Initial useful materials include common ferrites such as:

```text
3C90, 3C94, 3E6, 3F4, 77, 78, N27, N30, N49, N87
```

Dataset/source references:

- MagNet project page, Princeton University:
  `https://www.princeton.edu/~minjie/magnet.html`
- MagNet Challenge paper page, Princeton University:
  `https://collaborate.princeton.edu/en/publications/magnet-challenge-for-data-driven-power-magnetics-modeling`
- MagNet Challenge DOI:
  `https://doi.org/10.1109/OJPEL.2024.3469916`

### 4.2 Secondary Target: Bath Hysteresis Dataset

Use the Bath dataset second, after inspecting file structure.

The Bath dataset supports a Journal of Physics D paper on major and minor loop
magnetic hysteresis and includes direct experimental data plus model-fit
results for magnetic hysteresis experiments.

Use cases:

- minor-loop behavior;
- major-loop behavior;
- temperature-inference experiments;
- robustness against non-power-ferrite materials and measurement contexts.

Dataset/source references:

- Bath dataset DOI:
  `https://doi.org/10.15125/BATH-01316`
- Bath dataset page:
  `https://researchportal.bath.ac.uk/en/datasets/dataset-for-a-simplified-model-for-minor-and-major-loop-magnetic-`
- Journal article DOI:
  `https://doi.org/10.1088/1361-6463/acf13f`

---

## 5. QA/Whittaker Feature Map Target

Start with a conservative feature map that can actually be falsified.

Candidate v1 feature families:

```text
loop_work_measured = integral_closed_loop H dB
qa_curvature_proxy = integral_closed_loop Pi dtheta
```

where:

```text
Pi(t) = alpha_X*X(t) + alpha_J*J(t) + alpha_K*K(t)
```

with canonical QA invariant names preserved:

```text
J = b*d
X = d*e
K = d*a
F = b*a
C = 2*e*d
G = e*e + d*d
```

The first empirical version may use one of two sources for `Pi(t)`:

1. A declared QA tuple/phase generator tied to the loop sample index.
2. A fixture-declared `Pi_series` derived by a documented transform.

The calibration constants:

```text
alpha_X
alpha_J
alpha_K
```

must be learned or assigned only on the calibration split and then frozen.

---

## 6. Train/Holdout Protocol

Minimum split:

```text
calibration: one material, one frequency/amplitude condition
validation: same material, different amplitude or waveform
test: same material, different frequency or temperature
```

Preferred split:

```text
calibration: subset of material/frequency/temperature grid
validation: held-out operating points for same material
test: held-out materials or material families
```

No-refit rule:

```text
Once alpha_X, alpha_J, alpha_K and any feature-map convention are fixed,
they may not be changed on validation or test loops.
```

Any adaptive change after seeing validation/test errors must create a new
protocol version.

---

## 7. Baselines

The QA/Whittaker feature map must be compared against simple baselines before
claiming evidence.

Required baselines:

```text
constant mean loop work
amplitude-only fit
frequency-amplitude Steinmetz-style fit
ellipse / lossy sinusoid loop approximation
polynomial features of B(t), dB/dt, frequency, temperature
```

Optional later baselines:

```text
Jiles-Atherton model
Preisach model
small kernel regression baseline
small neural baseline
```

The cert should report whether QA/Whittaker beats:

- no baseline;
- trivial baselines only;
- classical Steinmetz-style baseline;
- stronger hysteresis-model baselines.

---

## 8. Evidence Levels

Weak evidence:

```text
one-loop fit only
```

Useful evidence:

```text
calibrate on one loop or one condition
predict held-out loop work without refitting
beat constant and amplitude-only baselines
```

Stronger evidence:

```text
same calibration predicts multiple amplitudes/frequencies
beats Steinmetz-style frequency-amplitude baseline
failure modes are structured and material-dependent
```

Strong evidence:

```text
calibration generalizes across related material families
beats stronger hysteresis baselines under predeclared splits
predicts loop-shape diagnostics, not only scalar loop area
```

---

## 9. Failure Criteria

The empirical cert must be able to fail.

Failure conditions:

- missing or mismatched `B(t)` / `H(t)` arrays;
- unit ambiguity;
- non-closed loop with no declared closure convention;
- calibration constants changed between calibration and evaluation;
- held-out error worse than predeclared baseline threshold;
- observer-float preprocessing used without a declared boundary;
- loop-work sign convention changed between records;
- dataset split leakage.

If the QA/Whittaker model does not beat baselines, the result is still useful:
it becomes a negative empirical observation, not a prose failure to explain.

---

## 10. Validator Build Plan

First build should be a data-ingestion and protocol validator, not a physics
victory cert.

Proposed artifact:

```text
qa_whittaker_hysteresis_empirical_protocol_cert_v1/
```

Required gates:

```text
WHE_1: dataset schema and source fields
WHE_2: B/H array alignment and units
WHE_3: loop-work recomputation
WHE_4: split discipline and no-refit constants
WHE_5: baseline declarations
WHE_6: held-out metric recomputation
WHE_7: non-claim and failure-ledger checks
```

Do not register this until:

- at least one public dataset file has been inspected;
- the parser knows the real file layout;
- a small fixture subset can be redistributed or referenced legally.

---

## 11. Non-Claims

This protocol does not claim:

- QA proves Whittaker 1903.
- QA proves Dollard, Bearden, or Steinmetz.
- Whittaker scalar potentials are physically validated.
- Maxwell/EM is derived from QA.
- Vacuum/plenum energy claims are established.
- One fitted loop is meaningful evidence.

This protocol does claim that a real physical observable has been identified
and that a QA/Whittaker hypothesis can be tested against it under fixed
calibration and held-out evaluation.

---

## 12. Immediate Next Tasks

1. Inspect MagNet file layout and license/access constraints.
2. Identify a small material subset with paired `B(t)` and `H(t)` traces.
3. Write a read-only dataset inventory note with counts by material,
   frequency, temperature, waveform, and available splits.
4. Define the first fixture schema using real field names.
5. Build only the ingestion/protocol validator first.
