# Pre-registration — Bearden Phase-Conjugate Injection Denial

**Locked:** 2026-04-05 (before any results file exists)
**Script:** `qa_bearden_injection_denial.py` (repo root, to be written after this file is committed)
**Cert family extended:** [155] QA_BEARDEN_PHASE_CONJUGATE_CERT.v1
**Runner:** Claude (per 2026-03-09 autonomy override)

## Hypothesis

**H1 (primary, sign-locked):** Under prompt-injection attack, QA coherence exhibits the Bearden phase-conjugate signature — global tightens, local scatters. Operationally:

> `QCI_gap = QCI_global − QCI_local` is **more positive** on malicious prompts than on benign, with
> `partial_r(QCI_gap, attack_label | char_length, scanner_hit_count) ≥ +0.15, p < 0.01`.

The positive sign is mandatory and locked before execution. A negative or null result is NOT reinterpreted post-hoc; it is a null.

**Why this sign:** [155] established the phase-conjugate mapping as pump→global (coupling tightens) + conjugate→local (trajectories scatter). An injection prompt acts as stress on an assistant; by Bearden's model, stress is a pumper. The prediction is therefore: attack prompts show elevated global coherence (tighter global resonance through the instruction-override pattern) alongside depressed local coherence (adversarial scatter at token level). `QCI_gap > 0` is the fingerprint.

## Data

- **Source:** `deepset/prompt-injections` (HuggingFace, Apache-2.0), full train+test = 546+116 = 662 rows
- **Pull mechanism:** HuggingFace `datasets-server.huggingface.co/rows` JSON API, paginated, zero library dependency
- **Cache:** `qa_alphageometry_ptolemy/external_validation_data/deepset_prompt_injections_full/deepset_prompt_injections_full.jsonl` + `.MANIFEST.json` with SHA256 lock
- **Frozen 30-row subset at `prompt_injection_benchmark_subset.jsonl` is NOT modified** — kept immutable
- **Fields used:** `text` (prompt string), `label` (0=benign, 1=injection)

## Observer projection (the single Theorem-NT boundary crossing)

For each prompt text `s`, the state sequence is:

```
states[i] = (ord(s[i]) % 24) + 1      for i in 0 .. len(s)-1
```

This is the **only** float/continuous → QA crossing in the pipeline. Declared observer projection. A1-compliant (states ∈ {1..24}, never 0). No tokenizer, no embedding model, no hidden vocab. Pure char-stream → integer. Deterministic, reproducible, minimal.

## Features

Per prompt:
- `length` = len(prompt) in characters
- `scanner_hits` = count of threat patterns matched by `qa_guardrail.threat_scanner.scan_for_threats`
- `qci_local` = nanmean of `QCI.compute(states)` with window = 7
- `qci_global` = nanmean of `QCI.compute(states)` with window = min(63, len(states)-2)
- `qci_gap` = `qci_global − qci_local`

QCI is invoked with identity cmap `{i: i for i in range(1, 25)}` since states are already in {1..24}.

Prompts with `len(states) < 10` are excluded from statistical tests (window-too-short — documented in results).

## Statistical test

Primary test: partial correlation

```
partial_r(qci_gap, label | length, scanner_hits)
```

via ordinary-least-squares residualization on both `qci_gap` and `label` against `[length, scanner_hits, intercept]`, then Pearson r on residuals. P-value via 10,000 label permutations (labels shuffled, all other features held fixed — surrogates the null, does not surrogate the QA state per `qa_observer.surrogates` convention).

Seed = 42 for the permutation RNG. Deterministic otherwise.

## Baselines (pre-declared — the real bar)

| # | Feature set | Purpose |
|---|---|---|
| B0 | `length` only | Trivial floor |
| B1 | `scanner_hits` | The real bar — existing threat scanner |
| B2 | Character bigram counts (1024 hashed buckets), logistic regression, 5-fold CV accuracy | Bag-of-patterns strawman |
| QA | `qci_gap` as partial-r(\| B0, B1) | Must show non-zero after controlling for B0+B1 |

Report: raw_r, partial_r(|B0), partial_r(|B0,B1), plus B2 accuracy for context.

## Decision criteria (locked)

| Outcome | Condition | Action |
|---|---|---|
| **STRONG** | partial_r ≥ +0.15, p<0.01, sign positive | Domain #2 certified; new fixture added to [155]; OB capture; propose cert amendment |
| **WEAK** | partial_r ≥ +0.10, 0.01 ≤ p < 0.05, sign positive | Pilot positive only; NOT added to cert; flag for domain-3 follow-up |
| **NULL** | \|partial_r\| < 0.10, OR p ≥ 0.05, OR sign wrong | Honest negative; OB capture; fall back to Rank-2 (ATT&CK × [191] Bateson reachability) |

Sign flip (negative partial_r at significance) counts as NULL — not "interesting opposite result". Phase-conjugate sign is a theoretical commitment from [155], not a fitting parameter.

## Deliverables

1. `qa_bearden_injection_denial.py` — standalone script at repo root
2. `external_validation_data/deepset_prompt_injections_full/deepset_prompt_injections_full.jsonl` + `.MANIFEST.json`
3. `results/bearden_injection/per_prompt.csv`
4. `results/bearden_injection/summary.json`
5. `results/bearden_injection/permutation_histogram.png`
6. OB captures: pre-reg commit, data pull complete, first stats, final outcome (one per milestone, not batched)
7. If STRONG: new fixture `qa_bearden_phase_conjugate_cert_v1/fixtures/bpc_injection_domain_pass.json` (existing finance fixture untouched)

## Guardrails checklist

- [x] T2-b: single char→state projection, no continuous→QA feedback
- [x] T2-D: no stochastic QA state generation; surrogates shuffle labels only
- [x] A1: states in {1..24}, never 0
- [x] S1: no `**2`
- [x] S2: all state arrays are `int` numpy dtype
- [x] T1: path length is integer char count
- [x] No secrets committed (Apache-2.0 public dataset)
- [x] Frozen subset immutable
- [x] Sign locked before first run
- [x] Baselines include scanner (guards circularity)
- [x] Permutation test holds real state fixed

## What would make me wrong

- Sign flip at significance → the Bearden mapping does not transfer from finance to text, even though both are "stress" domains. This would narrow [155]'s scope and is important information.
- Null at full power → QA coherence is not the relevant signal at char level; tokenizer-level might still work but that's a different experiment.
- Strong result only when scanner is EXCLUDED from controls → QA is rediscovering the scanner's patterns, not adding signal. This is why B1 is mandatory in the control set.
