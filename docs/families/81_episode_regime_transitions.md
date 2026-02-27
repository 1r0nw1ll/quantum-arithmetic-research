# Family [81] — QA Episode Regime Transitions Cert v1.0

**Machine tract:** `qa_episode_regime_cert_v1/`
**Schema version:** `QA_EPISODE_REGIME_CERT.v1.0`
**Upstream:** Family [80] `episode_labels` trajectory classifier

---

## What it certifies

Session-level dynamics over an ordered sequence of episode primary labels (produced by
Family [80]'s `episode_labels` classifier):

> **Given** an ordered list of episodes, each tagged with a `primary_label`,
> the cert verifies the transition matrix, regime composition counts, drift
> declaration, and max-run stability — using only integer comparisons.

---

## Label class mapping

| Primary label (Family [80]) | Regime class |
|---|---|
| `MONOTONE_ESCALATION` | `ESCALATION` |
| `MONOTONE_RECOVERY` | `RECOVERY` |
| `RETURN_TO_REF` | `RECOVERY` |
| `OSCILLATORY` | `NEUTRAL` |
| `STASIS` | `NEUTRAL` |

---

## Drift declaration rules (integer-only, exhaustive)

| Condition | drift_declaration |
|---|---|
| esc > rec AND esc > neu | `ESCALATING` |
| rec > esc AND rec > neu | `RECOVERING` |
| neu > esc AND neu > rec | `STABLE` |
| otherwise (ties, no dominant class) | `MIXED` |

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity · `schema_version` const `QA_EPISODE_REGIME_CERT.v1.0` · required fields |
| Gate 2 | Every `primary_label` in `ALLOWED_PRIMARY_LABELS` (`RETURN_TO_REF` / `MONOTONE_ESCALATION` / `MONOTONE_RECOVERY` / `OSCILLATORY` / `STASIS`) |
| Gate 3 | Recompute `regime_sequence`, `escalation_count`, `recovery_count`, `neutral_count`, `transition_matrix` (sparse, sorted lex) |
| Gate 4 | Recompute `drift_declaration` from integer counts |
| Gate 5 | Recompute `max_run_length` + `max_run_label` (first-occurrence tie-break) |

---

## Failure taxonomy

| `fail_type` | Trigger condition |
|---|---|
| `SCHEMA_INVALID` | schema_version mismatch, missing required fields, type violations |
| `LABEL_INVALID` | `primary_label` not in `ALLOWED_PRIMARY_LABELS` |
| `COUNT_MISMATCH` | `episode_count`, `escalation_count`, `recovery_count`, or `neutral_count` differs from computed |
| `REGIME_SEQUENCE_MISMATCH` | Computed `regime_sequence` differs from declared |
| `TRANSITION_MISMATCH` | Computed sparse transition matrix differs from declared |
| `DRIFT_MISMATCH` | Computed `drift_declaration` differs from declared |
| `MAX_RUN_MISMATCH` | Computed `max_run_length` or `max_run_label` differs |

---

## Output-only fields (non-cert)

When all 5 gates pass, the result envelope includes:

- **`transition_probabilities`**: exact rational `p(from→to) = count / (episode_count − 1)`,
  as `{from_label, to_label, count, prob_num, prob_den}` sorted lex
- **`first_recovery_index`**: 0-based index of first RECOVERY-class episode, or `null`
- **`last_escalation_index`**: 0-based index of last ESCALATION-class episode, or `null`
- **`cert_sha256`**: SHA-256 of canonical JSON of the cert input

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `PASS_RECOVERING.json` | PASS | 3 episodes: OSCILLATORY→MONOTONE_RECOVERY→RETURN_TO_REF; drift=RECOVERING |
| `PASS_ESCALATING.json` | PASS | 3 episodes: STASIS→MONOTONE_ESCALATION×2; drift=ESCALATING; max_run=2 |
| `PASS_MIXED.json` | PASS | 6 alternating RTR/ME episodes; drift=MIXED (tie esc=rec=3); RTR→ME ×3, ME→RTR ×2 |
| `FAIL_LABEL.json` | FAIL | `LABEL_INVALID` — episode has `primary_label: "THREATENING"` |
| `FAIL_TRANSITION.json` | FAIL | `TRANSITION_MISMATCH` — declares MONOTONE_RECOVERY→RETURN_TO_REF count=2 (actual=1) |
| `FAIL_DRIFT.json` | FAIL | `DRIFT_MISMATCH` — escalating episodes but declares drift=RECOVERING |

---

## Research context

Family [81] bridges Family [80] episode-level telemetry to **session-level behavioral
dynamics**. The label class mapping (ESCALATION / RECOVERY / NEUTRAL) mirrors the
generator family structure of CAPS_TR: fear-family generators produce ESCALATION,
love-family generators produce RECOVERY.

The drift declaration provides a single certified integer-arithmetic verdict on
the net trajectory of a session — stable, escalating, recovering, or mixed.
The transition matrix enables downstream Markov modeling of behavioral sequences.

---

## Self-test

```bash
python3 qa_episode_regime_cert_v1/validator.py --self-test
```

Expected: `RESULT: PASS` (6/6 fixtures).
