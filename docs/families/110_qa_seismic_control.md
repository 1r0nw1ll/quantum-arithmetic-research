# [110] QA Seismic Pattern Control

**Family ID**: 110
**Schema**: `QA_SEISMIC_CONTROL_CERT.v1`
**Scope**: Domain instance — second physical instantiation of `QA_PLAN_CONTROL_COMPILER_CERT.v1` [106]
**Directory**: `qa_alphageometry_ptolemy/qa_seismic_control/`

---

## Purpose

[110] is the second domain instance of the QA Plan-Control Compiler [106]. Together with [105] (cymatics), it proves that the kernel-governed compilation law is genuinely **cross-domain**: the same orbit-transition logic that governs Faraday instability patterns in cymatic experiments also governs elastic wave propagation in seismology, despite completely different underlying physics.

The core claim: the seismic path `quiet → p_wave → surface_wave` maps to the QA orbit trajectory `singularity → satellite → cosmos`, which is exactly the same orbit trajectory as the cymatic path `flat → stripes → hexagons` certified in [105]. Two completely different physical systems, same abstract control law.

---

## Physical Domain

> **Note on orbit assignments**: These are *abstract control labels* declared by the domain expert,
> not computed via `orbit_family(b,e,m)` arithmetic. The [110] cert validates legal control-graph
> paths and string-label consistency, not numeric orbit arithmetic.
> For the corrected arithmetic orbit rule see `qa_orbit_rules.py`:
> satellite requires `(m//3)|b AND (m//3)|e`; the direct (b,e) pairs in [110]'s state alphabet
> are all cosmos under this rule — satellite is reached via transition cross-products
> (see `seismic_orbit_classifier.py`).

| Seismic state | QA orbit family (abstract label) | Description |
|---------------|----------------------------------|-------------|
| `quiet`        | `singularity`   | Below detection threshold — fixed point, no wave energy |
| `p_wave`       | `satellite`     | Primary compressional wave — intermediate organized state |
| `s_wave`       | `satellite`     | Secondary shear wave — intermediate organized state |
| `coda`         | `satellite`     | Scattered tail wave — organized but decaying |
| `surface_wave` | `cosmos`        | Long-period Rayleigh/Love wave — fully organized cosmos state |
| `disordered`   | `out_of_orbit`  | Clipped/saturated signal — nonlinear escape |

**Generator alphabet** (seismic signal processing operations):

| Move | Effect |
|------|--------|
| `increase_gain` | Amplify signal; can cross the detection threshold |
| `decrease_gain` | Attenuate signal; returns to quiet or coda |
| `apply_bandpass` | Isolate a frequency band; selects s_wave from p_wave |
| `apply_lowpass` | Remove high-frequency content; selects surface wave |
| `apply_highpass` | (reserved) |

---

## Control Graph

```
quiet ──increase_gain──▶ p_wave ──apply_lowpass──▶ surface_wave
                          │
                          ├──apply_bandpass──▶ s_wave ──apply_lowpass──▶ surface_wave
                          │
                          └──increase_gain──▶ disordered (out_of_orbit)
surface_wave ──decrease_gain──▶ coda ──decrease_gain──▶ quiet
```

The legal path certified in [110]: `quiet → p_wave → surface_wave` (k=2).

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_PLAN_CONTROL_COMPILER_CERT.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'domain_instance'` | `SPEC_SCOPE_MISMATCH` |
| S1 | `path_length_k` equals `len(generator_sequence)` | `PATH_LENGTH_EXCEEDED` |
| S2 | `path_length_k ≤ max_path_length_k` | `PATH_LENGTH_EXCEEDED` |
| S3 | `final pattern_class` reaches `target_pattern_class` | `GOAL_NOT_REACHED` |
| S4 | All moves are legal edges in `control_graph` | `ILLEGAL_TRANSITION` |
| S5 | `final pattern_class` maps to recognized QA orbit family | `ORBIT_CLASS_MISMATCH` |
| S6 | `final_orbit_family` is not `out_of_orbit` | `NONLINEAR_ESCAPE` |

---

## Certified Fixtures

### `seismic_control_pass_surface_wave.json`

**Path**: `quiet → p_wave → surface_wave` (k=2, within max k=2)

**QA mapping**: `singularity → satellite → cosmos`

| Check | Result | Details |
|-------|--------|---------|
| IH1 | PASS | `inherits_from='QA_PLAN_CONTROL_COMPILER_CERT.v1'` |
| IH2 | PASS | `spec_scope='domain_instance'` |
| S1 | PASS | `path_length_k=2, len(generator_sequence)=2` |
| S2 | PASS | `2 ≤ 2` |
| S3 | PASS | `surface_wave == surface_wave` |
| S4 | PASS | Both transitions are legal edges in control_graph |
| S5 | PASS | `surface_wave → cosmos` |
| S6 | PASS | `cosmos ≠ out_of_orbit` |

### `seismic_control_fail_illegal_transition.json`

**Attempted path**: `quiet → surface_wave` via `apply_lowpass` (k=1, illegal)

No edge `(quiet, apply_lowpass, surface_wave)` exists in the control graph. The only legal move from `quiet` is `(quiet, increase_gain, p_wave)`. The state remains at `quiet` (singularity), never reaching `surface_wave` (cosmos).

**Fail types**: `ILLEGAL_TRANSITION` + `GOAL_NOT_REACHED`

---

## Cross-Domain Equivalence

The key result [110] establishes, in conjunction with [105]:

| Physical domain | Initial state | Path | Final state | QA orbit trajectory |
|-----------------|---------------|------|-------------|---------------------|
| Cymatics [105] | `flat` | `flat → stripes → hexagons` | `hexagons` | `singularity → satellite → cosmos` |
| Seismology [110] | `quiet` | `quiet → p_wave → surface_wave` | `surface_wave` | `singularity → satellite → cosmos` |

Different physics. Same orbit trajectory. This is what `QA_PLAN_CONTROL_COMPILER_CERT.v1` [106] claims: the compilation law is domain-generic.

---

## Inheritance Structure

```
[107] QA_CORE_SPEC.v1 (kernel)
    │
    └─── [106] QA_PLAN_CONTROL_COMPILER_CERT.v1 (family_extension)
               │
               ├─── [105] QA_CYMATIC_CONTROL_CERT.v1 (domain_instance, cymatics)
               │         [certified by inherit_pass_106_to_105.json in [109]]
               │
               └─── [110] QA_SEISMIC_CONTROL_CERT.v1 (domain_instance, seismology)
                         [certified by inherit_pass_106_to_110.json in [109]]
```

Both inheritance edges are certified by [109] QA Inheritance Compat.

---

## Running

```bash
# Self-test (2 fixtures)
python qa_seismic_control/qa_seismic_control_validate.py --self-test

# Single cert
python qa_seismic_control/qa_seismic_control_validate.py \
  --file qa_seismic_control/fixtures/seismic_control_pass_surface_wave.json

# Full meta-validator
python qa_meta_validator.py
```
