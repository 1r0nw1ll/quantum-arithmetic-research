# [259] QA HeartMath Coherence Firewall Cert

## What this is

Certifies Theorem NT (observer-projection firewall) applied to HeartMath-sourced cardiac-rhythm data. The cert enforces three structural discriminations:

1. **Rhythm-state labels are discrete orbit classes.** Every state in a rhythm trace is labeled as exactly one of `Singularity`, `Satellite`, or `Cosmos` — the three QA orbit classes on the mod-24 harmonic cycle. No other labels admitted.
2. **Continuous measurements are observer projections, not QA state.** Coherence-ratio, HRV, LF/HF, cross-coherence, and other continuous cardiac measurements appear only inside an `observer_projections` block with `direction` set to `input` or `output`, never as causal values of QA state variables `b`, `e`, `d`, `a`.
3. **At most two boundary crossings per state.** Per Theorem NT, the observer/QA boundary is crossed exactly twice — input projection (continuous signal → discrete orbit class) and output projection (discrete orbit class → continuous reading). No interior re-projection through a continuous intermediate.

Primary source: Oschman & Oschman (2015) *The Heart as a Bi-Directional Scalar Field Antenna* (J Vortex Sci Technol 2:121, DOI 10.4172/2090-8369.1000121). Four out of the 18 verbatim claims extracted from the HeartMath corpus (registered in `tools/qa_kg/fixtures/source_claims_heartmath.json`) act as witnesses:

- `oschman_2015_master_oscillator` — heart as master electrical oscillator broadcasting coherent frequencies system-wide → Cosmos orbit class
- `oschman_2015_rein_coherence` — ECG coherence under positive-emotion attention + heart-brain cross-coherence → Satellite orbit class (entrained 8-cycle)
- `oschman_2015_bidirectional_antenna` — antenna transmit/receive reciprocity → two-boundary-crossing symmetry in Theorem NT
- `edwards_2018_coherence_patterns` — population-scale coherence patterns consistent with orbit-class classification as the operational surface

The cert does **not** claim a numerical mapping between the QA Harmonic Index and HeartMath HRV coherence-ratio. That mapping requires primary-source data beyond the 4 Scholar-page results acquired in the Phase 4.8 kickoff; it is deferred to Phase 4.8 item 6 (McCraty peer-reviewed GCI papers + Radin IONS publications).

The cert does **not** do HRV measurement or biofeedback. It describes what the coherence phenomenon structurally reduces to *when the observer projection is declared properly*.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_heartmath_coherence_cert_v1/qa_heartmath_coherence_cert_validate.py` |
| Pass fixture | `qa_heartmath_coherence_cert_v1/fixtures/hmc_pass_oschman_rein_coherence.json` |
| Fail fixture | `qa_heartmath_coherence_cert_v1/fixtures/hmc_fail_float_qa_state.json` |
| Mapping ref | `qa_heartmath_coherence_cert_v1/mapping_protocol_ref.json` |
| Primary source PDF | `Documents/heartmath_corpus/oschman_2015_heart_bidirectional_scalar_antenna.pdf` |
| Excerpts | `docs/theory/heartmath_phase4_8_excerpts.md` |
| Claim fixtures | `tools/qa_kg/fixtures/source_claims_heartmath.json` |
| Theory doc | `docs/theory/QA_HEARTMATH_COHERENCE.md` |
| Phase 4.8 handoff | `docs/specs/QA_MEM_PHASE_4_8_HANDOFF.md` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_heartmath_coherence_cert_v1
python qa_heartmath_coherence_cert_validate.py --self-test
```

## Semantics

- **HMC_1**: `schema_version` matches `QA_HEARTMATH_COHERENCE_CERT.v1`.
- **HMC_RHYTHM_LABELS**: every `rhythm_trace[*].orbit_class` is in `{Singularity, Satellite, Cosmos}`.
- **HMC_NO_FLOAT_STATE**: every `rhythm_trace[*].qa_state` key is in `{b, e, d, a}`, every value is an integer (not bool, not float); top-level rhythm-state fields do not carry a continuous-measurement alias (`coherence_ratio`, `hrv`, `lf_hf`, `hr_bpm`, `ibi_ms`, `rmssd`, `sdnn`, `pnn50`, `vlf`, `total_power`, `spectral_entropy`).
- **HMC_BOUNDARY_CROSSINGS**: each `rhythm_trace[*].observer_projections` list has at most two entries; each entry has a `direction` in `{input, output}`; if two entries are present, one must be `input` and the other `output`.
- **HMC_SRC**: `source_attribution` names at least one HeartMath primary author (`Oschman`, `Danielson`, `Edwards`, or `Tomasino`), the project name (`HeartMath` or `heartmath`), and `Dale`.
- **HMC_WITNESS**: at least two `witnesses[*].claim_id` values match IDs in `tools/qa_kg/fixtures/source_claims_heartmath.json` (hardcoded set of 18 known IDs in the validator).
- **HMC_F**: `fail_ledger` is a list.

## Relation to other certs

- **Theorem NT (A1/A2/T2)** — the cert is the domain instance of Theorem NT for HeartMath-sourced rhythm data. Same structural discipline as [257] `qa_integer_state_pipeline_cert_v1` (GLM-5 TITO two-boundary invariant), applied to cardiac-rhythm measurement.
- **[122] `qa_empirical_observation_cert`** — the Empirical Observation cert is the generic bridge from captured experimental data to the cert ecosystem; this cert is a domain specialization for HeartMath data specifically, with hardcoded primary-source witness IDs.
- **[218] `qa_haramein_scaling_diagonal_cert_v1`** — structural template: both certs ground a claim in a published primary source (Haramein 2008 / Oschman 2015) and use primary-source PDF + excerpts MD + source_claims fixture as the witness chain.
- **[197] `qa_see_longitudinal_transverse_cert_v1`** — Oschman 2015's scalar-antenna framing ties to See 1917's longitudinal-transverse duality already certified in [197]; a future cert could formalize the explicit structural bridge (not this cert).
- **[234] `qa_chromogeometry_pythagorean_identity_cert_v1`** — two-element reconstruction `C² + F² = G²` on the HVMB Möbius surface is a potential follow-up claim (not this cert).

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `HMC_RHYTHM_LABELS` | A rhythm state carries an `orbit_class` not in `{Singularity, Satellite, Cosmos}` (e.g. `chaotic`, `disordered`, `coherent`). | Reclassify into one of the three QA orbit classes; the qualitative cardiac-rhythm distinctions in the HeartMath literature have a direct canonical mapping. |
| `HMC_NO_FLOAT_STATE` (T2-b) | A continuous measurement was assigned as a QA state variable, e.g. `qa_state.b = 0.89` (coherence ratio). | Move the continuous value into `observer_projections[].reading` with `direction=output`. The QA state must remain an integer tuple. |
| `HMC_BOUNDARY_CROSSINGS` | More than two projections per state, or a direction other than `input` / `output`, or two projections sharing the same direction. | Collapse to one input + one output; any additional projection implies a re-projection through a continuous intermediate, which is a Theorem NT violation. |
| `HMC_SRC` | Attribution missing a HeartMath primary author, the project name, or the cert author. | Add the missing field to `source_attribution`. |
| `HMC_WITNESS` | Fewer than two witnesses cite `claim_id` values registered in `source_claims_heartmath.json`. | Add witnesses citing at least two of the 18 known claim IDs (Oschman/Danielson/Edwards/Tomasino). |
| `HMC_F` | `fail_ledger` missing or not a list. | Ensure the field exists — empty list is valid for PASS fixtures. |

## Scope boundary

**The cert does NOT:**
- Derive HRV measurement methodology, compute coherence ratios, or perform any biofeedback analysis.
- Claim numerical equivalence between QA Harmonic Index and HRV coherence ratio.
- Validate the truth of any HeartMath empirical result; it validates only the structural form of the mapping between HeartMath rhythm-state data and QA orbit classes.
- Cover the full HeartMath corpus. The 18 primary-source claims are from one Scholar page (`start=130`); McCraty's peer-reviewed GCI papers and Radin IONS publications are deferred to Phase 4.8 item 6.

**The cert DOES:**
- Prevent continuous HRV / coherence measurements from being treated as causal QA inputs (T2-b firewall breach).
- Force every rhythm-state label in a HeartMath-derived dataset to be one of the three QA orbit classes.
- Require every such dataset to cite at least two primary-source claim IDs registered in the QA-MEM source_claims fixture.

This is the same discipline as [257] applied to GLM-5's token pipeline, rebased on cardiac-rhythm data instead of LLM tokens.
