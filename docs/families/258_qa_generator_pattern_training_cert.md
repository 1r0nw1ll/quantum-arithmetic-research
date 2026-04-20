# [258] QA Generator-Pattern Training Cert

## What this is

Certifies QA-native training: identification of an integer generator pattern `(b_0, e_0)` whose T-orbit matches a target trace by exact-match count, via exhaustive discrete search over `m^2 = 81` starting tuples on S_9. No gradients, no float optimizer state (no Adam `m`/`v` buffers, no momentum, no EMA, no learning rate), no importance-sampling corrections, no policy-version staleness, no trust region, no optimizer reset. Evaluation is an integer-valued orbit-trace metric (exact match count), not a scalar loss.

Structurally eliminates the GLM-5 async-RL policy-staleness failure class (arXiv:2602.15763, §4.1.2) by construction: there is no float policy to drift, so the entire apparatus of staleness bounds (`|w_current - w_0| > δ`), double-sided importance sampling with `[1-ε, 1+h]` trust region, hard masking outside, and optimizer reset on weight push, is inapplicable.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_generator_pattern_training_cert_v1/qa_generator_pattern_training_cert_validate.py` |
| Pass fixture | `qa_generator_pattern_training_cert_v1/fixtures/gpt_pass_default.json` |
| Fail fixture | `qa_generator_pattern_training_cert_v1/fixtures/gpt_fail_gradient_optimizer.json` |
| Mapping ref | `qa_generator_pattern_training_cert_v1/mapping_protocol_ref.json` |
| Reference prototype | `qa_lab/qa_orbit_resonance_attention.py` (`identify_generator`, `identify_family`) |
| Design doc | `docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md` §Failure-Mode-3 |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_generator_pattern_training_cert_v1
python qa_generator_pattern_training_cert_validate.py --self-test
```

## Semantics

- **GPT_1**: schema_version matches.
- **GPT_INT_GEN**: `trained_generator` is an integer tuple `(b, e)` in `{1..m}^2`.
- **GPT_NO_GRAD**: `training_path.uses_gradients == false`; `optimizer.kind == "discrete_search"`; `optimizer.float_state` empty.
- **GPT_DISCRETE**: `training_path.search_space_size == m*m` (81 on S_9); `steps_enumerated` ≤ search space size.
- **GPT_ORBIT_EVAL**: `evaluation.metric` ∈ `{exact_match_count, orbit_trace_hamming, family_plurality}`; `scalar_loss_used` is not true.
- **GPT_DET**: validator independently recomputes `identify_generator(canonical_target_trace)` twice and asserts bitwise equality; declared `trained_generator` and `declared_match_score` match recomputation.
- **GPT_SRC**: source attribution references `2602.15763`, `[209]`, and `[256]`.
- **GPT_WIT**: at least 5 `(target_trace, generator)` witnesses, covering all five T-orbit families (Fibonacci, Lucas, Phibonacci, Tribonacci, Ninbonacci).
- **GPT_F**: `fail_ledger` well-formed.

## Relation to GLM-5

GLM-5 manages the drift of a continuous policy with three layers of apparatus: staleness bounds to drop trajectories whose rollout-time policy version diverged too far, double-sided importance sampling with hard masking to bound trust-region divergence, and optimizer reset to prevent continuous accumulation across discrete policy transitions. Each layer is a patch for drift of a float object. With the float object removed — replaced by an integer generator pattern `(b_0, e_0)` — all three apparatus layers become unnecessary. Training is discrete search; the result is bitwise reproducible; there is no continuum of "mostly-current" policies.

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `GPT_NO_GRAD` | Declared `uses_gradients=true` or `optimizer.kind != "discrete_search"` | Remove gradient path; replace Adam with discrete enumeration. |
| `GPT_DISCRETE` | `search_space_size != m*m` or `steps_enumerated` out of range | Enumerate the full generator space. |
| `GPT_ORBIT_EVAL` | Evaluation uses scalar loss (MSE, cross-entropy, etc.) | Switch to exact orbit-trace match; evaluation must be integer-valued. |
| `GPT_DET` | Declared trained_generator contradicts recomputation | Regenerate; the function should be deterministic. |
| `GPT_INT_GEN` | Trained generator is float/Fraction | Trained generator must be an integer tuple. |
| `GPT_WIT` | Missing witnesses for one or more T-orbit families | Add canonical traces for all five families (Fibonacci/Lucas/Phibonacci/Tribonacci/Ninbonacci). |

## Changelog

- **v1** (2026-04-19): Initial release. Together with [256] (attention) and [257] (pipeline), completes the QA-native architecture cert triple addressing the three GLM-5 failure-mode classes.
