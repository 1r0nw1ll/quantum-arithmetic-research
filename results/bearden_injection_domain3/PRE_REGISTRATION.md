# Pre-registration — Bearden Phase-Conjugate Domain #3 Replication

**Locked:** 2026-04-05 (before any results file exists)
**Script:** `qa_bearden_injection_denial_domain3.py`
**Cert family targeted:** [155] QA_BEARDEN_PHASE_CONJUGATE_CERT.v1
**Runner:** Claude (per 2026-03-09 autonomy override)

## Purpose

Direct replication of the domain-2 Bearden injection pilot (2026-04-05, WEAK) on a **larger, independent dataset** to test whether the phase-conjugate QCI_gap signal persists or was dataset-specific to `deepset/prompt-injections`.

Domain-2 result: `partial_r(qci_gap, label | length, scanner_hits) = +0.1393`, perm `p = 0.0006`, n=661, sign correct, classified WEAK (partial_r short of STRONG threshold +0.15 by 0.007 — the p-value was 17× tighter than required but the effect size was under).

## Hypothesis (sign-locked, inherited from domain-2 pre-registration)

**H1**: On `xTRam1/safe-guard-prompt-injection` (10,296 rows), `partial_r(qci_gap, attack_label | char_length, scanner_hit_count) ≥ +0.15, p < 0.01`, sign positive, using **exactly the same char-level `(ord(c) % 24) + 1` observer projection and window parameters** as the domain-2 script.

Nothing is retuned. This is a clean replication, not an exploration.

## Data

- **Source:** `xTRam1/safe-guard-prompt-injection` (HuggingFace)
- **Size:** train 8,236 + test 2,060 = **10,296 rows**
- **Label balance:** ~70% benign (5,740) / 30% injection (2,496) on train (from HF `/statistics`)
- **Schema:** `text` (string) + `label` (int, 0=benign, 1=injection) — drop-in compatible with domain-2 pipeline
- **License:** public HuggingFace dataset, no auth required; will check the HF card at pull time for license string and record in manifest
- **Pull mechanism:** `datasets-server.huggingface.co/rows` API, paginated (100/call), zero library dependency
- **Cache:** `qa_alphageometry_ptolemy/external_validation_data/xtram1_safeguard_injection/xtram1_safeguard_injection.jsonl` + `MANIFEST.json` with SHA256
- **Sample verification (pre-pull inspection):** 6 rows sampled show clear benign (math, reading comp) vs clear injection (unrestricted mode, elaborate jailbreak framings) — consistent with binary-classification structure.

## Observer projection

**Identical to domain-2**: `states[i] = (ord(text[i]) % 24) + 1`. No retuning. Any change would void the replication claim.

## Feature extraction

**Identical to domain-2**:
- `length` = char count
- `scanner_hits` = count of threat patterns from `qa_guardrail.threat_scanner.scan_for_threats`
- `qci_local` = nanmean of `QCI.compute(states)` with window=7
- `qci_global` = nanmean with window=min(63, len(states)-2)
- `qci_gap = qci_global - qci_local`
- Prompts with `len(states) < 10` excluded

## Statistical test (identical)

- Primary: `partial_r(qci_gap, label | length, scanner_hits)` via OLS residualization, Pearson on residuals
- Permutation p via 10,000 label shuffles, seed=42

## Baselines (identical)

- B0: length
- B1: scanner_hits
- B2: char-bigram 5-fold CV logistic regression

## Decision criteria (identical, sign-locked positive)

| Outcome | Condition | Action |
|---|---|---|
| **STRONG** | partial_r ≥ +0.15, perm p < 0.01, sign positive | Replication successful. [155] gets domain #2 (both pilots combined) via new fixture. Writeup candidate. |
| **WEAK** | partial_r ≥ +0.10, 0.01 ≤ perm p < 0.05, sign positive | Both pilots WEAK = consistent but weak. Not cert-added. Thread retired until further notice. |
| **NULL** | \|partial_r\| < 0.10, OR p ≥ 0.05, OR sign wrong | Thread killed. Domain-2 finding is likely dataset-specific or false-positive. OB + memory capture. Move to Rank-3 (GRASP IAM). |

**Power note:** n = ~10,000 vs domain-2 n = 661. If the domain-2 effect (r = 0.139) is real, this run should land STRONG (r ≥ 0.15 is ~1.1σ from the domain-2 point estimate, well within sample-size error bars). If it is also exactly 0.139 here, the permutation p will be *vanishingly* small but the classification will still land WEAK by the rubric. That's the correct behavior — the effect-size threshold is the real bar and we hold it.

## Deliverables

1. `qa_bearden_injection_denial_domain3.py` — standalone, linter-clean; may import shared pure-functions from `qa_bearden_injection_denial.py` if importable without side effects
2. `external_validation_data/xtram1_safeguard_injection/` (data + manifest, SHA256 locked)
3. `results/bearden_injection_domain3/`:
   - `per_prompt.csv` (~10k rows, text column dropped)
   - `summary.json`
   - `permutation_histogram.png`
4. OB captures at: data pull, first stats, final outcome
5. If STRONG: new fixture `qa_bearden_phase_conjugate_cert_v1/fixtures/bpc_injection_domain_pass.json` (covers both pilots)
6. No modification to frozen/existing files

## Guardrails (identical to domain-2 + collision note from Bateson NULL)

- [x] T2-b: single char→state projection, input-side only
- [x] A1: states in {1..24}
- [x] S1: no `**2`
- [x] T1: integer char-length path time
- [x] Sign locked before run
- [x] Scanner hit count in controls (not features) — guards against QA rediscovering scanner patterns
- [x] Permutation null holds real state fixed, shuffles labels only
- [x] Frozen domain-2 artifacts untouched
- [x] No retuning of any hyperparameter from domain-2; pure replication

## What would make me wrong

- **STRONG on domain-3 but not domain-2**: would suggest the deepset pilot was underpowered, not wrong. Combined evidence strong.
- **Also WEAK on domain-3**: signal is consistent but modest across datasets. Honest call is retire the thread — two weak pilots don't compound to a strong one without a theoretical reason to expect super-linear aggregation.
- **NULL on domain-3 at n=10k**: almost certainly means the deepset result was dataset-specific or artifact. Kill the thread cleanly; move to Rank-3.
- **Direction flip on domain-3**: rare but possible — would indicate a systematic difference in how the two datasets frame injection attempts (e.g., domain-2 relies on "ignore previous instructions" pattern, domain-3 may use different evasion styles). Would be publishable as a negative finding about Bearden sign portability.
