# [263] QA Failure Density Enumeration Cert

## What this is

The **first sharp-claim cert** derived from the Kochenderfer 2026 *Algorithms for Validation* bridge ([Kochenderfer 2026](../../docs/theory/kochenderfer_validation_excerpts.md); [bridge spec](../specs/QA_KOCHENDERFER_BRIDGE.md)). Anchors the existing cert [194] `qa_cognition_space_morphospace_cert_v1` (Sole 2026; arxiv:2601.12837 via Dale 2026) and re-expresses its load-bearing ratios in Kochenderfer Ch. 7 vocabulary, then validates the recasting against Kochenderfer's own direct-sampling estimator.

**Primary source**:
- Kochenderfer, M. J., Wheeler, T. A., Katz, S., Corso, A., & Moss, R. J. (Kochenderfer, 2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. 441 pp. Chapter 7 (Failure Probability Estimation) §7.1 (direct estimation MLE) + eq. 7.3 (Bernoulli standard error). Verbatim anchor: [`docs/theory/kochenderfer_validation_excerpts.md#val-7-1-direct-estimation-pfail`](../theory/kochenderfer_validation_excerpts.md).

**Construction**:

```
p_fail = E_{τ ~ p(·)}[1{τ ∉ ψ}] = ∫ 1{τ ∉ ψ} p(τ) dτ            (Kochenderfer eq. 7.1)

specializes on a finite QA orbit graph S_m of cardinality m² to:

p_fail = |{s ∈ S_m : s ∉ ψ}| / |S_m|                              (exact enumeration)

with variance identically zero, while the canonical direct-sampling estimator
Algorithm 7.1 has standard error

σ̂ = sqrt(p (1 − p) / N)                                          (Kochenderfer eq. 7.3)
```

**On S_9 (mod-9), this reproduces cert [194] ratios exactly**:
- `singularity` ↦ `1/81`
- `satellite` ↦ `8/81`
- `cosmos` ↦ `72/81`

**Sampling-comparison gate**: at `N ∈ {100, 1000, 10000}` per orbit class, with seed=42 (primary fixture) and seeds 42/1337/2024 (variance-decay fixture), the seeded direct-sampling estimator p̂ falls inside the |error| ≤ 4σ envelope predicted by Kochenderfer eq. 7.3 — load-bearing falsifiable empirical claim distinguishing this cert from a definition cert.

## Claim scope

This cert does **not** prove that QA enumeration is novel as probability theory. It proves that an existing QA reachability cert ([194]) can be re-expressed in Kochenderfer's validation vocabulary and gains an exact finite-state estimator with zero sampling variance, while the corresponding direct estimator exhibits the expected Bernoulli sampling error. Outsider-clean framing.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_failure_density_enumeration_cert_v1/qa_failure_density_enumeration_cert_validate.py` |
| Utility module | `tools/qa_kg/orbit_failure_enumeration.py` |
| Primary PASS fixture | `qa_failure_density_enumeration_cert_v1/fixtures/pass_mod9_cognition_morphospace.json` |
| Variance-decay PASS fixture | `qa_failure_density_enumeration_cert_v1/fixtures/pass_sampling_comparison.json` |
| FAIL fixture (ratio mismatch) | `qa_failure_density_enumeration_cert_v1/fixtures/fail_bad_ratio.json` |
| Mapping ref | `qa_failure_density_enumeration_cert_v1/mapping_protocol_ref.json` |
| Source verbatim anchor | `docs/theory/kochenderfer_validation_excerpts.md#val-7-1-direct-estimation-pfail` |
| Bridge spec | `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §3 |
| Anchor cert | `qa_cognition_space_morphospace_cert_v1/` [194] |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_failure_density_enumeration_cert_v1
python qa_failure_density_enumeration_cert_validate.py --self-test
```

## Gates

- **FDE_1** — `schema_version == "QA_FAILURE_DENSITY_ENUMERATION_CERT.v1"`.
- **FDE_RATIO** — Declared `exact_enumeration.counts` and `exact_enumeration.ratios` reproduce the cert [194] ratios `1/81`, `8/81`, `72/81` bit-exact via the orbit-class enumeration utility (mod-9 anchor; mod-24 deferred until a published mod-24 classifier lands).
- **FDE_SAMPLING** — Each declared `(target_class, n_samples, seed)` sampling case is re-run by the validator with the same seed; the result must match the declared `expected_p_hat` and `expected_n_in_target` bit-exactly (seeded determinism), and `|p_hat − p_true| ≤ 4σ` per Kochenderfer eq. 7.3.
- **FDE_STDERR** — Theoretical `σ = sqrt(p (1 − p) / N)` reproduced for each `(p, N)` in declared cases; `enumeration_error` field, when present, must equal `0` (exact).
- **FDE_UTIL** — `tools/qa_kg/orbit_failure_enumeration.py` is importable and exposes the four named functions: `enumerate_orbit_class_counts`, `exact_success_failure_probability`, `direct_sampling_estimate`, `theoretical_standard_error`.
- **FDE_SRC** — `source_attribution` cites Kochenderfer 2026 + cert [194].
- **FDE_WIT** — at least 3 witnesses, one per orbit class.
- **FDE_F** — `fail_ledger` is well-formed (list); FAIL fixtures (with `result == "FAIL"`) early-return after FDE_1 + FDE_F per the cert [194] convention.

## Why mod-9 only in v1

Cert [194] `qa_cognition_space_morphospace_cert_v1` is the canonical mod-9 anchor: it specifies the orbit-class classifier (`singularity` ⇔ `(9,9)`; `satellite` ⇔ `b%3 == 0 ∧ e%3 == 0 ∧ ¬singularity`; `cosmos` otherwise) that this cert reuses. A mod-24 extension would need a published mod-24 orbit-family classifier landing in a future cert before this cert's enumeration utility can be safely extended; rather than silently extrapolating, `_classifier_for_modulus` raises `NotImplementedError` for non-9 moduli.

## Cross-references

- [194] `qa_cognition_space_morphospace_cert_v1` — anchor cert; this family reproduces its `1/81 / 8/81 / 72/81` ratios bit-exact.
- [191] `qa_bateson_learning_levels_cert_v1` — candidate consumer of the enumeration utility for tiered-reachability ratios on `S_9` (26% / 52.67% / 20%).
- [193] `qa_levin_cognitive_lightcone_cert_v1` — candidate consumer for orbit-radius / agency ratios on cancer ↔ Cosmos transitions.
- `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §3 — controlled mapping table; this cert flips the bridge row from `candidate` to `established` once it lands.

## Future work

- mod-24 extension once a canonical mod-24 orbit-family classifier is published.
- Three other candidate sharp-claim certs flagged in the bridge spec but not built: `qa_discrete_robustness_cert_v1`, `qa_runtime_odd_monitor_cert_v1`, `qa_counterfactual_descent_cert_v1`.
- Adopt the utility at certs [191] and [193] (replace hand-rolled enumeration with `orbit_failure_enumeration` calls) to make this cert load-bearing rather than ornamental.
