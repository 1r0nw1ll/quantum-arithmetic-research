# SWE-Bench Calibration Dashboard (Pass 15)

Combined view of harness accuracy against two complementary truth sources.

## 1. Designed truth — hand-crafted fixtures

Truth comes from each fixture's `hidden_label/expected_*.json`. These are
the labels the SWE-Bench domain was authored against.

- Total fixtures scored: **8**
- Labeled fixtures: **8**
- Match: **8**
- Mismatch: **0**
- **Accuracy on designed truth: 100.0%**

### Confusion matrix
| expected \ got | accept | revise | reject |
|---|---:|---:|---:|
| accept | 3 | 0 | 0 |
| revise | 0 | 0 | 0 |
| reject | 0 | 0 | 5 |

### Per-fixture
| layer | case | expected | got | match |
|---|---|---|---|---|
| review | `good_canonical_patch` | accept | accept | ✓ |
| review | `polished_bad_overclaim_patch` | reject | reject | ✓ |
| review | `sparse_legit_minimal_diff` | accept | accept | ✓ |
| repair | `reject_test_removal` | reject | reject | ✓ |
| repair | `revise_wrong_file_touched` | accept | accept | ✓ |
| deception | `deception_irrelevant_symbols` | reject | reject | ✓ |
| deception | `deception_no_actual_change` | reject | reject | ✓ |
| deception | `deception_overclaim_with_placeholder` | reject | reject | ✓ |

## 2. Executed truth — Pass-13c FAIL_TO_PASS

Truth comes from real test execution against cloned repos at base_commit.
- Total executed-truth datapoints: **13**
- Testable on this machine (py3.13): **13**
- Untested (env unavailable, astropy py-version mismatch): **0**
- Bugfixers among testable: **13**
- **True positives (heuristic accept ∧ actually fixes bug): 13**
- **False accepts (heuristic accept ∧ does not fix bug): 0**
- **False reject/revise (heuristic non-accept ∧ actually fixes bug): 0**

### Per-patch
| label | patch | applies | FAIL_TO_PASS | current decision | class |
|---|---|---|---|---|---|
| `django-11477/canonical` | canonical | YES | 3/3 | accept | true_positive |
| `django-11477/minimal_tests` | codex_live_agent | YES | 3/3 | accept | true_positive |
| `django-11211/canonical` | canonical | YES | 1/1 | accept | true_positive |
| `django-11211/minimal_tests` | codex_live_agent | YES | 1/1 | accept | true_positive |
| `astropy-14539/canonical` | canonical | YES | 2/2 | accept | true_positive |
| `astropy-14539/looks_done` | codex_live_agent | YES | 2/2 | accept | true_positive |
| `astropy-14539/minimal_tests` | codex_live_agent | YES | 2/2 | accept | true_positive |
| `django-14915/baseline` | codex_live_agent_v2 | YES | 1/1 | accept | true_positive |
| `django-14915/overclaim` | codex_live_agent_v2 | YES | 1/1 | accept | true_positive |
| `django-15375/baseline` | codex_live_agent_v2 | YES | 1/1 | accept | true_positive |
| `django-16136/baseline` | codex_live_agent_v2 | YES | 2/2 | accept | true_positive |
| `django-16136/overclaim` | codex_live_agent_v2 | YES | 2/2 | accept | true_positive |
| `django-16612/baseline` | codex_live_agent_v2 | YES | 2/2 | accept | true_positive |

## 3. Per-gate progression on the Pass-12 live-agent set (25 codex outputs)

Same 25 patches, same base, scored under successively tightened harness states:

| stage | accept | revise | reject | timeout | note |
|---|---:|---:|---:|---:|---|
| pre-13b heuristic-only (Pass-12 sample, n=25) | 17 | 2 | 4 | 2 | Heuristic alone — overstates real correctness; pure-text gates |
| post-13b + apply-check (Pass-12 sample) | 3 | 1 | 19 | 2 | Adds `git apply --check` against base_commit. Catches malformed diffs (hunk-header count mismatch, non-ASCII whitespace) the heuristic missed. |
| post-14a + tiered patch-relevance (Pass-12 sample) | 4 | 0 | 19 | 2 | Softens canonical-files-touched from binary to tier-1/2/3/4. Recovers django-11211/minimal_tests false-revise without opening any new false accepts. |
| Pass-V1.3 expansion heuristic-only (n=60) | 28 | 4 | 27 | 1 | 30 new SWE-Bench Verified tasks × 2 prompt variants. 47% accept rate before apply-check is wired. |
| Pass-V1.3 expansion + apply-check | 6 | 4 | 49 | 1 | Apply-check rejects 22/28 heuristic-accepts (79% structural malformation rate at scale — hunk-count mismatches, non-ASCII whitespace). Survivors are 6 codex patches across 4 unique django tasks. |

**Δ accept rate (heuristic-only → current):** 17/25 (68%) → 4/25 (16%). Of the
13 patches that flipped, all were due to objective tool-native gates (apply-check)
or empirically-justified relaxation (Pass-14a recovering 1 false-revise).

## 4. Headline — current state of harness

- Designed-truth accuracy: **100.0%** (8/8)
- Executed-truth precision (TP / (TP + FA)) on testable Django subset: **100.0%**
- Executed-truth recall (TP / (TP + FR)) on testable Django subset: **100.0%**
- Untested executed truth (astropy): **0/13** awaiting Docker / pyenv (Pass 14b)

**Open questions the dashboard cannot yet close:**
- Astropy correctness (3 patches blocked on env)
- Larger SWE-Bench corpus (n=20 in current sample; real coverage needs ~hundreds)
- Other domains' precision/recall on execution truth (TLA+, Lean 4, Upwork have
  no execution-truth source yet — only designed-truth fixtures)
