# [117] QA Control Stack

**Family ID**: 117
**Schema**: `QA_CONTROL_STACK_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; synthesis of [106], [105], [110]
**Directory**: `qa_alphageometry_ptolemy/qa_control_stack/`

---

## Purpose

[117] is the control-side analogue of [115]. It certifies that the QA plan-control compiler law is **domain-generic**: the orbit trajectory `singularity→satellite→cosmos` and `path_length_k=2` are structural invariants preserved across physically distinct domains under the same kernel-governed generator algebra.

> **Control Stack Theorem**: QA_PLAN_CONTROL_COMPILER_CERT.v1 [106] is not cymatics-specific. Two independent physical domains — Faraday instability (cymatics [105]) and elastic wave propagation (seismology [110]) — both instantiate the orbit trajectory `singularity→satellite→cosmos` with `path_length_k=2`, using entirely different state labels and move implementations. The compiler law governs abstract orbit structure; the physical domain provides only concrete instantiation.

---

## Cross-Domain Comparison

| Domain | Initial | Intermediate | Target | Orbit path | k | Moves |
|--------|---------|--------------|--------|------------|---|-------|
| Cymatics [105] | flat | stripes | hexagons | singularity→satellite→cosmos | 2 | increase_amplitude, set_frequency |
| Seismology [110] | quiet | p_wave | surface_wave | singularity→satellite→cosmos | 2 | increase_gain, apply_lowpass |

Different physics. Same orbit trajectory. The compiler law is structural.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1–IH3 | Kernel inheritance | `INVALID_KERNEL_REFERENCE` etc. |
| CS1 | `compiler_ref` is `QA_PLAN_CONTROL_COMPILER_CERT.v1` | `COMPILER_REF_MISMATCH` |
| CS2 | Each `domain_instance_refs` entry is a known [106] domain instance | `DOMAIN_INSTANCE_MISMATCH` |
| CS3 | All `domain_traces` have identical `orbit_trajectory` | `ORBIT_TRAJECTORY_MISMATCH` |
| CS4 | All `domain_traces` have equal `path_length_k` | `PATH_LENGTH_MISMATCH` |
| CS5 | All `domain_traces` have equal `initial_orbit_family` | `ORBIT_TRAJECTORY_MISMATCH` |
| CS6 | All `domain_traces` have equal `final_orbit_family` | `ORBIT_TRAJECTORY_MISMATCH` |
| CS7 | `cross_domain_claim.orbit_trajectory_preserved` consistent with CS3+CS5+CS6 | `CROSS_DOMAIN_CLAIM_INCONSISTENT` |
| CS8 | `cross_domain_claim.path_length_equal` consistent with CS4 | `CROSS_DOMAIN_CLAIM_INCONSISTENT` |
| CS9 | `cross_domain_claim.compiler_law_domain_generic` consistent with CS7+CS8 | `STACK_INCONSISTENCY` |
| CS10 | `canonical_orbit_trajectory` matches all trace orbit_trajectories | `ORBIT_TRAJECTORY_MISMATCH` |
| CS11 | `canonical_path_length_k` matches all trace path_length_k | `PATH_LENGTH_MISMATCH` |

---

## Certified Fixtures

### `control_stack_pass_cross_domain.json`

Both cymatics and seismology traces declare `orbit_trajectory=[singularity,satellite,cosmos]`, `path_length_k=2`. `cross_domain_claim`: all three flags true. All CS checks pass. **PASS.**

### `control_stack_fail_orbit_mismatch.json`

Seismology trace declares `orbit_trajectory=[singularity,satellite,satellite]` and `final_orbit_family=satellite` (should be cosmos). Cymatics correctly declares cosmos. `cross_domain_claim` asserts `orbit_trajectory_preserved=true` — a lie.

**Fail types**: `ORBIT_TRAJECTORY_MISMATCH` + `CROSS_DOMAIN_CLAIM_INCONSISTENT` + `STACK_INCONSISTENCY`. **FAIL.**

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  └── [106] QA_PLAN_CONTROL_COMPILER_CERT.v1 (compiler law)
        ├── [105] QA_CYMATIC_CONTROL_CERT.v1 (cymatics domain instance)
        └── [110] QA_SEISMIC_CONTROL_CERT.v1 (seismology domain instance)
  └── [117] QA_CONTROL_STACK_CERT.v1 (synthesis: domain-genericity certified)
        └── [118] QA_CONTROL_STACK_REPORT.v1 (reader-ready report)
```

---

## Running

```bash
python qa_control_stack/qa_control_stack_validate.py --self-test
python qa_meta_validator.py
```
