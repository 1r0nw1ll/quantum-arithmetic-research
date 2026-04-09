# Pre-registration — Bearden Phase-Conjugate via Continuous Stress Proxy

**Locked:** 2026-04-05 (after implementation-bug correction; BEFORE computing any entropy/gzip stats on the cached data)
**Script:** `qa_bearden_stress_proxy.py`
**Cert family targeted:** [155] QA_BEARDEN_PHASE_CONJUGATE_CERT.v1
**Runner:** Claude (per 2026-03-09 autonomy override + Will's 2026-04-05 directive to not give up)

## Why this experiment exists

The previous two Bearden injection pilots (domain-2 deepset, domain-3 xTRam1) used **binary attack label** as the "stress" target. Two implementation errors were found after a failed replication:

1. **Gap sign convention inverted** — fixed in code 2026-04-05. Under corrected `gap = local − global` (per cert fixture), domain-2 WEAKLY replicates the finance sign (partial_r = −0.1393) and domain-3 inverts (partial_r = +0.1399).
2. **Attack label ≠ stress.** Finance Bearden uses continuous future volatility as stress. Attack label is a poor text analog because benign distributions differ systematically across injection datasets — deepset benigns are short neutral questions (low complexity stress); xTRam1 benigns are elaborate Q/A / code review (high complexity stress). The label↔stress mapping inverts.

This experiment replaces the binary label with a **continuous, dataset-independent stress proxy** — matching the finance Bearden setup structurally rather than by proxy.

## Primary hypothesis (sign LOCKED from [155] finance direction)

**H1:** `raw_r(qci_gap, char_entropy)` is **negative** on **both** datasets, with magnitude ≥ 0.10 and p < 0.05 on both, where `qci_gap = qci_local − qci_global` per [155] cert fixture definition.

- Sign: LOCKED NEGATIVE (matches finance partial_r = −0.17 to −0.42)
- Magnitude thresholds inherited from the original domain-2/3 pre-registrations (0.10/0.15)
- No retuning of window, modulus, or cmap from the original

**Why this matches Bearden's mechanism:** in Bearden's model, stress (high information-processing demand) causes global coherence to tighten (pump beam) and local trajectories to scatter (conjugate response). Character entropy is a proper analog because high-entropy prompts carry more information per character → more processing demand → more "stress" in the Bearden sense. Low-entropy prompts (repetitive, simple) are low-stress. This is structurally the same quantity finance uses (realized volatility = information-per-timestep), just for text.

## Secondary hypotheses (reported, not thresholded)

- **H2:** `raw_r(qci_gap, gzip_ratio)` is negative on both datasets. Gzip compressibility ratio is a second complexity proxy; same direction expected as entropy but via a different mechanism (redundancy detection rather than character-level Shannon).
- **H3:** `raw_r(qci_gap, scanner_hit_count)` is negative on both datasets. Scanner hits measure adversarial/complex linguistic density; should track stress.
- **H4:** `raw_r(qci_gap, length)` — reported without sign expectation; length is the trivial stress proxy and serves as a sanity check.
- **H5 (confirmatory):** partial_r(qci_gap, char_entropy | length) — tests whether the entropy effect is beyond mere length.

## Data

**Both datasets used simultaneously, same script, same pipeline.** Cached from earlier runs — no new pulls.

- `deepset/prompt-injections`: 662 rows, sha256 `9268e51031e3...`
- `xTRam1/safe-guard-prompt-injection`: 10,296 rows, sha256 `438e1ae273e5...`

## Observer projection (unchanged from domain-2)

`states[i] = (ord(text[i]) % 24) + 1` — single Theorem-NT boundary crossing, A1-compliant.

## Feature extraction (gap definition CORRECTED, everything else inherited)

Per prompt:
- `length` = char count (as before)
- `scanner_hits` = count of threat patterns (as before)
- `qci_local` = nanmean of `QCI.compute(states)`, window=7
- `qci_global` = nanmean, window=min(63, len(states)-2)
- **`qci_gap = qci_local − qci_global`** (corrected per [155] cert fixture)
- **`char_entropy`** = Shannon entropy of character distribution: `-Σ p(c) * log2(p(c))`, where `p(c)` is empirical char frequency in the prompt. Bits per char. Single scalar per prompt.
- **`gzip_ratio`** = `len(gzip(text.encode('utf-8'))) / max(len(text.encode('utf-8')), 1)`. Higher = less compressible = higher complexity.

All features are computed on the EXACT SAME rows used in domain-2 and domain-3 runs. Prompts with length < 10 excluded (as before).

## Statistical tests (per dataset)

Per dataset:
- `raw_r(qci_gap, char_entropy)` with Pearson p (primary)
- `raw_r(qci_gap, gzip_ratio)` (secondary)
- `raw_r(qci_gap, scanner_hits)` (secondary)
- `raw_r(qci_gap, length)` (secondary)
- `partial_r(qci_gap, char_entropy | length)` (confirmatory H5)
- Permutation p on primary test: 10,000 entropy shuffles, seed=42

## Decision criteria (LOCKED before computing any entropy/gzip values)

Let `r2, p2` = primary stats on domain 2 (deepset), `r3, p3` = primary stats on domain 3 (xTRam1). Both computed on the primary statistic `raw_r(qci_gap, char_entropy)`.

| Outcome | Conditions | Action |
|---|---|---|
| **STRONG** | `r2 <= -0.15` AND `r3 <= -0.15` AND `p2 < 0.01` AND `p3 < 0.01` | Fresh empirical witness for [155] across both text datasets; new fixture, propose cert amendment, paper candidate |
| **WEAK** | `r2 < 0` AND `r3 < 0` AND (`r2 <= -0.10` OR `r3 <= -0.10`) AND `p2 < 0.05` AND `p3 < 0.05` | Consistent sign across both datasets; pilot successful but magnitude-short; thread alive, writeup as "partial replication" |
| **SINGLE** | `r2 < 0` AND `r3 < 0` AND exactly one meets `|r| >= 0.10, p < 0.05` | Asymmetric. Report honestly; stronger dataset = replication, weaker = robustness question. |
| **NULL** | signs inconsistent across datasets OR both non-significant with wrong sign | Two datasets, same sign → stress proxy doesn't carry Bearden signature in text even with continuous target. Honest thread retirement. |

**Sign-inconsistency NULL rule:** if `r2 < 0` and `r3 > 0` (or vice versa) at significance, this counts as NULL. But — and this is the point of this whole experiment — a continuous, dataset-independent stress proxy should NOT exhibit the label↔stress mapping inversion that binary labels suffer from. If entropy shows sign flip across datasets under this test, the thread has a different problem (something deeper than label choice).

## What success / failure looks like

- **Success** (STRONG or WEAK): Bearden phase-conjugate mechanism transfers from finance to text under a proper continuous stress mapping. Label-choice inversion was the implementation error. Thread alive, possibly paperable.
- **SINGLE**: still alive but with a caveat about dataset sensitivity. Needs a third dataset.
- **NULL with sign flip**: the stress mapping itself (not just the label proxy) inverts between datasets. Something structural about how text datasets encode complexity differs and I don't yet understand what. Honest retirement of the injection-specific Bearden thread, BUT [155] finance cert untouched.
- **NULL with same-sign but sub-threshold**: signal exists but too weak for the magnitude bar; possibly needs a better observer projection than char-level `(ord%24)+1`.

## Guardrails

- [x] Gap sign correction verified against cert fixture (`QCI_gap = QCI_local - QCI_global`)
- [x] Sign lock inherited from finance direction (negative), not re-derived
- [x] Magnitude thresholds inherited from original pre-reg (no retuning)
- [x] Data reuse declared and accepted (same rows, different target — structurally new hypothesis, not a retest of the same hypothesis)
- [x] Entropy values NOT computed before writing this pre-reg (verified: no `entropy` anywhere in existing per_prompt.csv files)
- [x] No library dependency changes (entropy + gzip are stdlib)
- [x] T2-b, A1, S1, S2, T1 all unchanged
- [x] Both datasets tested under a single experiment, not cherry-picked

## Pre-execution commitment

Whatever this script produces, the result is final and goes into the project memory under the classification above. I will not retune the observer, the window, the modulus, or the stress proxy after seeing the numbers. If NULL, the thread is retired honestly; if STRONG, the cert gets amended. The point of this pre-reg is to prevent another round of motivated reinterpretation.
