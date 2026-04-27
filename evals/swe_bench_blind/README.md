# SWE-Bench Verified Blind Eval Suite (Pass 10 pilot)

Fourth domain for the anti-slop legitimacy harness, after
[tla_blind](/home/player2/signal_experiments/evals/tla_blind),
[lean4_blind](/home/player2/signal_experiments/evals/lean4_blind), and
[upwork_blind](/home/player2/signal_experiments/evals/upwork_blind).

**Goal:** test whether the shared cross-domain core (`evals/_blind_core/`,
extracted in Pass 9) generalizes to a fourth domain with a different
artifact shape — unified-diff patches against real Python repos, with
external acceptance structure (FAIL_TO_PASS test sets, canonical
maintainer patches) — without re-duplicating helpers.

## Source

[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
— the human-verified subset of SWE-Bench. 500 tasks across 12 popular
Python repos (django, sympy, sphinx, matplotlib, scikit-learn, astropy,
xarray, pytest, pylint, requests, seaborn, flask).

Pilot sample: 20 tasks at `<15 min fix` or `15 min - 1 hour` difficulty,
sampled with repo diversity (8 repos), problem_statement < 4000 chars,
patch < 1500 chars. Stored at
`/home/player2/upstream_corpora/swe_bench_verified/sample.json`.

## Scope of pilot

- **Text heuristics only.** No test execution. Pass 10 validates the harness
  shape; Pass 10.5+ would integrate FAIL_TO_PASS verification.
- 5 generation tasks (canonical-patch deterministic generator).
- 7 review/repair/deception fixtures.
- Reuses `_blind_core` for ORDER, worst_of, bucket_for_finding,
  load_expected, bundle_present.

## Score axes

7 axes per ChatGPT's Pass-10 spec:

- **task_validity_score** (top-line) — well-formed diff addressing core issue
- **external_admissibility_score** (top-line) — maintainer would accept
- **requirement_coverage_score** — touches the right files
- **patch_relevance_score** — references issue symbols
- **scope_honesty_score** — commit message matches diff scope
- **deliverable_fit_score** — single coherent unified diff
- **source_fidelity_score** — uses real repo paths + issue-named symbols

Plus `reviewer_rejection_risk_score` aggregate.

## Layout

```text
evals/swe_bench_blind/
├── README.md                    (this file)
├── runner.py
├── execute_current_system.py
├── rubrics/
│   ├── rubric.md
│   └── scorecard_schema.json
├── tasks/
│   └── generation/              (5 generation tasks from sample)
├── review_corpus/               (good / polished-bad / sparse-legit)
├── repair_cases/                (reject / revise)
├── deception_corpus/            (overclaims / removes-tests / wrong-symbol)
└── results/
```

## Run

```bash
python3 evals/swe_bench_blind/runner.py list
python3 evals/swe_bench_blind/runner.py show review good_canonical_patch
python3 evals/swe_bench_blind/execute_current_system.py
```

## Wires into

- `evals/blind_benchmark/benchmark_current_corpus.py` — cross-domain accuracy sweep
- `evals/deception_regression/run_regression.py` — precision regression
