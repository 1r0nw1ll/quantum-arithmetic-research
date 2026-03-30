# [122] QA Empirical Observation Cert

**Schema:** `QA_EMPIRICAL_OBSERVATION_CERT.v1`
**Family root:** `qa_alphageometry_ptolemy/qa_empirical_observation_cert/`
**Status:** PASS (127/127 → 128/128)

---

## Purpose

This family is the **pipe between the observation layer and the cert ecosystem**.

Before [122], experimental results (Open Brain captures, experiment script outputs, paper results) and the cert families ([107]–[121]) ran as completely separate tracks. [122] makes that connection formal and machine-checkable.

A cert in this family answers one question:

> "Is this specific empirical observation CONSISTENT, CONTRADICTS, PARTIAL, or INCONCLUSIVE with respect to a named claim in a parent cert?"

---

## What it is NOT

- Not a claim that QA *caused* the observed result
- Not a proof — verdicts are empirical, not deductive
- Not a cert of the experiment *code* — it certifies the *result* against a parent cert *claim*
- A CONTRADICTS cert with result=PASS is a **good outcome**: it means an honest negative result is correctly documented

---

## Schema fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | must be `"QA_EMPIRICAL_OBSERVATION_CERT.v1"` |
| `cert_type` | string | yes | must be `"qa_empirical_observation_cert"` |
| `certificate_id` | string | yes | unique ID |
| `title` | string | yes | human-readable description |
| `created_utc` | string | yes | ISO 8601 |
| `observation.source` | string | yes | one of `open_brain`, `experiment_script`, `paper_result`, `external_dataset` |
| `observation.summary` | string | yes | what happened |
| `observation.domain` | string | yes | e.g. `audio_signal_processing`, `quantitative_finance`, `eeg` |
| `parent_cert.schema_version` | string | yes | which cert family this observation addresses |
| `parent_cert.claim` | string | yes | specific claim being evaluated |
| `verdict` | string | yes | `CONSISTENT` \| `CONTRADICTS` \| `PARTIAL` \| `INCONCLUSIVE` |
| `evidence` | array | yes | at least one item required (V5) |
| `fail_ledger` | array | yes | required if verdict=CONTRADICTS (V4); empty otherwise |
| `result` | string | yes | `PASS` or `FAIL` (validator outcome, not verdict) |

---

## Validator checks

| ID | Check | Fail type |
|---|---|---|
| V1 | `observation.source` ∈ known sources | `UNKNOWN_OBSERVATION_SOURCE` |
| V2 | `parent_cert.schema_version` is a non-empty string | `INVALID_PARENT_CERT_REF` |
| V3 | `verdict` ∈ {CONSISTENT, CONTRADICTS, PARTIAL, INCONCLUSIVE} | `INVALID_VERDICT` |
| V4 | `verdict==CONTRADICTS` → `fail_ledger` nonempty | `CONTRADICTS_WITHOUT_FAIL_LEDGER` |
| V5 | `evidence` nonempty | `EMPTY_EVIDENCE` |

---

## Fixtures

| File | Verdict | Result | Domain | Notes |
|---|---|---|---|---|
| `eoc_pass_audio_orbit_consistent.json` | CONSISTENT | PASS | audio | Sine tones above chance, white noise at null — from `qa_audio_orbit_test.py` 2026-03-25 |
| `eoc_pass_finance_contradicts.json` | CONTRADICTS | PASS | finance | Script 26 curvature→vol FAIL — pre-declared criteria not met cross-asset |
| `eoc_fail_empty_evidence.json` | CONSISTENT | FAIL | test | Empty evidence list; EMPTY_EVIDENCE fires |

---

## The synthesis this enables

| Before [122] | After [122] |
|---|---|
| Open Brain observation: "audio test shows orbit coherence" | Certifiable: `verdict=CONSISTENT` against [107] state_space claim |
| Finance script 26: "curvature → vol FAIL" | Certifiable: `verdict=CONTRADICTS` with documented fail_ledger entry |
| Meta-validator knows nothing about experiments | Meta-validator: 128/128 includes empirical bridge |

---

## Writing a new observation cert

1. Run the experiment or retrieve the Open Brain capture
2. Identify which parent cert claim the result addresses
3. Fill in the schema — be specific about `parent_cert.claim`
4. Choose verdict honestly: CONTRADICTS results with evidence are as valuable as CONSISTENT
5. If CONTRADICTS, document the specific invariant_diff in `fail_ledger`
6. Run `python qa_empirical_observation_cert_validate.py --file your_cert.json`
7. Add to the family and add a FAMILY_SWEEPS pointer if creating a new sub-family

---

## Relationship to other families

- **[107] QA_CORE_SPEC.v1**: most observations will reference this as parent_cert (state_space, generators, orbit structure claims)
- **[111]–[116] obstruction spine**: arithmetic obstruction claims can be observed in experiments
- **[117]–[118] control spine**: cross-domain control claims have empirical correlates
- **[119] dual spine unification**: synthesis-level claims about the two spines can be observed in cross-domain experiments
