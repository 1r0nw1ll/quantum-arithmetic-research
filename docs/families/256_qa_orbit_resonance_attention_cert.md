# [256] QA Orbit-Resonance Attention Cert

## What this is

Certifies a QA-native attention operator in which attention is a deterministic pairwise resonance relation on T-orbit tuples, with no learned parameters and no stochastic top-k selection. Three resonance rules (`family_match`, `norm_match`, `chromogeometry`) are certified orbit-invariant under T evolution on S_9: the attention matrix at integer path-time `t` equals the matrix at `t=0` for every `t`. Structurally eliminates the GLM-5 DSA non-deterministic-`topk` entropy-collapse failure mode (arXiv:2602.15763, Feb 2026) by construction.

The design claim: attention is not a learned scoring function followed by discrete selection. Attention *is* orbit resonance. A deterministic function of integer orbit structure has no non-determinism to patch, no gradient path to freeze, and no top-k operator to stabilize.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_orbit_resonance_attention_cert_v1/qa_orbit_resonance_attention_cert_validate.py` |
| Pass fixture | `qa_orbit_resonance_attention_cert_v1/fixtures/ora_pass_default.json` |
| Fail fixture | `qa_orbit_resonance_attention_cert_v1/fixtures/ora_fail_wrong_invariance.json` |
| Mapping ref | `qa_orbit_resonance_attention_cert_v1/mapping_protocol_ref.json` |
| Reference prototype | `qa_lab/qa_orbit_resonance_attention.py` |
| Design doc | `docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md` |

All paths relative to `qa_alphageometry_ptolemy/` unless otherwise noted.

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_orbit_resonance_attention_cert_v1
python qa_orbit_resonance_attention_cert_validate.py --self-test

# Or via meta-validator
cd qa_alphageometry_ptolemy
python qa_meta_validator.py
```

## Semantics

- **ORA_1**: `schema_version == "QA_ORBIT_RESONANCE_ATTENTION_CERT.v1"`.
- **ORA_DET**: declared `determinism.bitwise_identical == true` and `determinism.repeats >= 100`; independently re-verified by calling each rule 100+ times and comparing matrices bitwise.
- **ORA_A1**: all `canonical_witness_tokens` lie in `{1..9}^2`.
- **ORA_INV_FAM / _NORM / _CHR**: independent recomputation confirms `family_match`, `norm_match`, and `chromogeometry` attention matrices are invariant across 24+ T-steps on declared tokens.
- **ORA_GRAN**: on exhaustive S_9 pairs, both directions of the family–norm crosscut exist (same-family-different-norm pair present AND same-norm-different-family pair present); declared crosscut pairs independently verified.
- **ORA_SRC**: source attribution references GLM-5 arXiv `2602.15763`.
- **ORA_WIT**: at least five witnesses covering all five T-orbit families — Fibonacci, Lucas, Phibonacci, Tribonacci, Ninbonacci.
- **ORA_F**: `fail_ledger` well-formed.

## T-orbit family classification on S_9 (from cert [214])

| Family | Coarse class | Representative | Eisenstein norm mod 9 |
|---|---|---|---|
| Fibonacci | cosmos (length 24) | (1, 1) | {1, 8} |
| Lucas | cosmos (length 24) | (1, 3) | {4, 5} |
| Phibonacci | cosmos (length 24) | (1, 4) | {2, 7} |
| Tribonacci | satellite (length 8) | (3, 3) | {0} |
| Ninbonacci | singularity (length 1) | (9, 9) | {0} |

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `ORA_INV_*` | A declared orbit-invariant is false under recomputation, OR steps_tested < 24. | Verify rule really is T-invariant; if not, it is not QA-native by this cert's criterion. |
| `ORA_DET` | Determinism either not declared `true` or fails under recomputation. | Source of non-determinism is a compliance violation — same class as GLM-5's non-deterministic topk. |
| `ORA_A1` | Token outside `{1..m}^2`. | Fix input projection to respect A1. |
| `ORA_GRAN` | Declared crosscut pair is self-contradictory OR no crosscut exists on S_9. | Pick pairs that actually cross the two partitions. |
| `ORA_WIT` | Fewer than 5 witnesses or missing a T-orbit family. | Add witnesses covering all five families. |
| `ORA_SRC` | Source attribution missing GLM-5 reference. | Cite `arXiv:2602.15763`. |

## Relation to GLM-5

GLM-5 (arXiv:2602.15763, Feb 2026) reports that DSA's lightning indexer with non-deterministic CUDA `topk` caused entropy collapse within a few RL steps; they forced `torch.topk` (deterministic, slower) and froze the indexer during RL to stabilize. That sequence of patches is the observed symptom of a continuous-learning system with a fragile discrete-selection patch bolted on. The QA-native operator certified here has no scoring function, no top-k, and no gradient path; the failure mode has nothing to attach to.

## Changelog

- **v1** (2026-04-19): Initial release. Reference prototype (qa_lab/qa_orbit_resonance_attention.py) validated end-to-end: 11 self-tests PASS, orbit-invariance confirmed over 24 T-steps on 10 diverse tokens across all five T-orbit families.
