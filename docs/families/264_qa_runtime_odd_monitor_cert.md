# [264] QA Runtime ODD Monitor Cert

## What this is

The **second sharp-claim cert** derived from the Kochenderfer 2026 *Algorithms for Validation* bridge (see [bridge spec §5](../specs/QA_KOCHENDERFER_BRIDGE.md), ODD-monitor row), and the first to demonstrate the deterministic-vs-continuous-classifier comparison gate that the bridge spec flagged as `open` since 2026-04-26. Anchored at Kochenderfer Validation §12.1 *Operational Design Domain Monitoring* superlevel-set ODD construction.

**Primary source**:
- Kochenderfer, M. J., Wheeler, T. A., Katz, S., Corso, A., & Moss, R. J. (Kochenderfer, 2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. Chapter 12 §12.1 'Operational Design Domain Monitoring' — superlevel-set ODD definition (`P(in_ODD | x) > 0.5` threshold). Verbatim anchor at [`docs/theory/kochenderfer_validation_excerpts.md#val-12-2-aleatoric-vs-epistemic-uncertainty`](../theory/kochenderfer_validation_excerpts.md) (the §12.1 ODD construction itself was not anchored as a quote in the v1 ingestion; this cert references the §12.1 framing alongside the §12.2 aleatoric/epistemic distinction that grounds the input-noise robustness claim).

## Claim (narrow)

For QA finite orbit-class regimes on `S_9`, ODD membership can be monitored by deterministic orbit-family membership with `FP=0` and `FN=0` relative to the declared discrete ODD. On continuous observer projections `(b, e) → ((b-1)/8, (e-1)/8) ∈ [0,1]²` of QA-discrete inputs, a Kochenderfer-style 1-NN classifier-superlevel-set baseline produces non-zero classification error near orbit-class boundaries that scales with input-noise σ.

**Claim scope**:
- Claim does **not** generalize to all runtime ODD monitoring.
- Claim does **not** say continuous classifiers are bad globally.
- The exactness holds only on the QA-discrete side of the Theorem NT firewall (when the ODD is declared as an orbit-family subset and the monitor does an integer-only membership check).
- The classifier-baseline leakage is specific to the boundary geometry of orbit families on `S_9` under the canonical 1/8-spaced unit-square embedding.

## Construction

**Deterministic monitor** (QA-discrete side, FP=0 / FN=0 by construction):
```
in_ODD(b, e) := orbit_family_s9(b, e) ∈ declared_odd
```
where `declared_odd ⊆ {singularity, satellite, cosmos}` and `orbit_family_s9` is the canonical mod-9 orbit-class classifier from `tools/qa_kg/orbit_failure_enumeration.py` (cert [263]'s utility).

**Kochenderfer classifier-superlevel-set baseline** (continuous observer-projection side):
1. Embed each `(b, e) ∈ {1..9}²` to `((b-1)/8, (e-1)/8) ∈ [0,1]²` — canonical 1/8-spaced unit square.
2. Build training set: 81 `(clean_embedding, orbit_family)` pairs.
3. Inject Gaussian noise at seeded `σ ∈ {0.05, 0.1, 0.2}`: `noisy_embed = clean_embed + N(0, σ²I)`.
4. Predict label by 1-NN on the training set (clean embeddings).
5. Compare `predicted_label ∈ declared_odd` to `true_label ∈ declared_odd`.
6. Count FP (predicted in, true out) + FN (predicted out, true in).

The 1-NN baseline is a simplification of Kochenderfer's full superlevel-set construction (§12.1 uses a probabilistic classifier with `P(in_ODD | x) > 0.5` threshold) sufficient to establish the leakage claim while keeping the cert dependency-light. The leakage gap grows with `σ` until `σ` exceeds the `1/8` grid spacing, at which point the 1-NN label saturates toward the locally dominant class.

## Empirical results (declared in fixtures, recomputed by validator)

| ODD | σ | seed | classifier FP | classifier FN | total |
|---|---|---|---|---|---|
| `{cosmos}` | 0.05 | 42 | 1 | 3 | 81 |
| `{cosmos}` | 0.1  | 42 | 5 | 6 | 81 |
| `{cosmos}` | 0.2  | 42 | 5 | 5 | 81 |
| `{cosmos}` | 0.05 | 1337 | 3 | 2 | 81 |
| `{cosmos}` | 0.1  | 1337 | 8 | 3 | 81 |
| `{cosmos}` | 0.2  | 1337 | 9 | 7 | 81 |
| `{satellite}` | 0.05 | 42 | 3 | 1 | 81 |
| `{satellite}` | 0.1  | 42 | 5 | 5 | 81 |
| `{satellite}` | 0.2  | 42 | 4 | 5 | 81 |

Deterministic monitor: `FP = 0, FN = 0` in all cases (by construction).

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_runtime_odd_monitor_cert_v1/qa_runtime_odd_monitor_cert_validate.py` |
| Utility (shared) | `tools/qa_kg/orbit_failure_enumeration.py` (cert [263] anchor) |
| Primary PASS fixture (cosmos ODD) | `qa_runtime_odd_monitor_cert_v1/fixtures/pass_s9_declared_odd_exact_membership.json` |
| Boundary-comparison PASS fixture (satellite ODD) | `qa_runtime_odd_monitor_cert_v1/fixtures/pass_classifier_boundary_comparison.json` |
| FAIL fixture (bad ODD label) | `qa_runtime_odd_monitor_cert_v1/fixtures/fail_bad_odd_label.json` |
| FAIL fixture (no leakage observed) | `qa_runtime_odd_monitor_cert_v1/fixtures/fail_continuous_boundary_leakage.json` |
| Mapping ref | `qa_runtime_odd_monitor_cert_v1/mapping_protocol_ref.json` |
| Bridge spec | `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §5 ODD-monitor row |
| Anchor cert (utility provider) | `qa_failure_density_enumeration_cert_v1/` [263] |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_runtime_odd_monitor_cert_v1
python qa_runtime_odd_monitor_cert_validate.py --self-test
```

## Gates

- **ODD_1** — `schema_version == "QA_RUNTIME_ODD_MONITOR_CERT.v1"`.
- **ODD_DECL** — `declared_odd` is a non-empty list and every entry is in the canonical orbit-class set `{singularity, satellite, cosmos}`.
- **ODD_EXACT** — Deterministic monitor reproduces declared `fp=0, fn=0` on all 81 `S_9` states (verified by independent recomputation).
- **ODD_CLF** — Each declared `(σ, seed)` classifier-test-case is re-run by the validator with the same seed; `expected_classifier_fp` and `expected_classifier_fn` must match the recomputation bit-exactly (seeded determinism).
- **ODD_LEAK** — At least one declared `(σ, seed)` case must produce `FP + FN > 0` (the leakage claim requires empirical evidence that the continuous classifier baseline misclassifies under noise).
- **ODD_SRC** — `source_attribution` cites Kochenderfer 2026 + cert [263].
- **ODD_WIT** — at least 3 witnesses, one per orbit class.
- **ODD_F** — `fail_ledger` is well-formed (list); FAIL fixtures (with `result == "FAIL"`) early-return after `ODD_1` + `ODD_F` per the cert [194] / [263] convention.

## Theorem NT compliance

The deterministic monitor stays integer-only: `orbit_family_s9` returns a string label from integer `(b, e)`. The classifier baseline operates on continuous embedding — declared as **observer projection at the input boundary** (single noise injection via `random.Random(seed).gauss` per state), and the comparison metrics (FP/FN counts) are integer-valued. The Theorem NT firewall is crossed exactly twice: input-side at the noise injection, output-side at the FP/FN tally. No interior re-projection.

## Why mod-9 only in v1

Same constraint as cert [263]: the canonical mod-9 orbit-family classifier is fixed in `orbit_family_s9` (cert [194] anchor). Cert [263]'s utility raises `NotImplementedError` for non-9 moduli rather than silently extrapolating; this cert inherits that constraint. mod-24 extension would need a published mod-24 orbit-family classifier landing first.

## Why synthetic, not HeartMath

Per Will + ChatGPT scoping 2026-04-27: the v1 cert claim is mathematical (deterministic-vs-classifier on the canonical embedding); HeartMath data introduces unnecessary surface area (HRV preprocessing, continuous sensor semantics, prior-cert dependency on [259]). Synthetic-first keeps the claim sharp. v2 may add cert [259] HeartMath as a secondary applied-ODD fixture if the deterministic-vs-classifier comparison stays small.

## Cross-references

- [263] `qa_failure_density_enumeration_cert_v1` — utility provider; this cert reuses `orbit_family_s9` from the same `tools/qa_kg/orbit_failure_enumeration.py` module.
- [194] `qa_cognition_space_morphospace_cert_v1` — canonical mod-9 orbit-family classifier; this cert's deterministic monitor is exactly that classifier under a different framing.
- [257] `qa_integer_state_pipeline_cert_v1` — Theorem NT two-boundary-crossing invariant; this cert's deterministic monitor + classifier baseline together cross the firewall exactly twice (input projection-in, output FP/FN-count-out).
- [259] `qa_heartmath_coherence_cert_v1` — candidate v2 secondary fixture; HeartMath HRV → orbit-class projection is the canonical applied ODD case.
- `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §5 ODD-monitor row — flips from `open` to `established (2026-04-27, mod-9 synthetic)` once this cert lands.

## Future work

- v2: add cert [259] HeartMath secondary fixture for an applied ODD with real continuous-sensor inputs.
- mod-24 extension once a canonical mod-24 orbit-family classifier is published.
- Tighten the classifier baseline: replace 1-NN with a probabilistic classifier + threshold matching Kochenderfer §12.1 exactly; the leakage results should be qualitatively similar but quantitatively different.
- Sweep `σ` finer at the noise scales where the FP+FN count saturates (around `σ ≈ 1/8` grid spacing).
