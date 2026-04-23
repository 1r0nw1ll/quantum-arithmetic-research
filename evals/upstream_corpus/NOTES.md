# Upstream-Approved-Corpus Blind Benchmark

This directory benchmarks the current formal-publication-gate harness against
the **external community-approved** work in each formal domain, rather than
against the internal curated fixtures under `evals/tla_blind/` and
`evals/lean4_blind/`.

The internal corpus benchmark (`evals/blind_benchmark/`) reports 14/14 because
Codex wrote both the fixtures and the scoring heuristics. The question this
directory answers is different:

> How does the as-is harness score real upstream-approved artifacts, whose
> authors had no knowledge of our bundle format, rubric phrasing, or
> required-artifact list?

## Corpora (not vendored — cloned locally, provenance tracked)

| Domain | Source | Commit | Artifacts used |
|---|---|---|---|
| TLA+ | [`tlaplus/Examples`](https://github.com/tlaplus/Examples) | `d9ce4db7` | every spec directory under `specifications/` |
| Lean 4 | [`leanprover-community/mathematics_in_lean`](https://github.com/leanprover-community/mathematics_in_lean) | `2bf0e10d` | every completed `Solutions_*.lean` file |

Both repos live under `/home/player2/upstream_corpora/` (outside the repo
tree). The benchmark script re-reads the SHAs at runtime for provenance.

## Pass (a): as-is, no charitable adaptation

Per explicit scope: **no evidence bundles are synthesized**. The adapter only
performs file discovery and provenance tracking. If the harness requires
`source_grounding.json` and the upstream spec has none, that is reported as
a finding — the absence of those files is part of the current harness behavior
and is what is being measured.

## Finding buckets

Every rejection reason is classified into one of six buckets so the
headline reject rate can be decomposed:

- `missing_required_artifact` — harness requires a bundle file the upstream
  artifact simply does not have (`source_grounding.json`, `repo_fit_review.json`,
  etc.)
- `missing_explicit_evidence` — artifact exists but does not contain the exact
  phrases the harness greps for (source excerpts, grounding claims)
- `weak_outsider_translation` — artifact does not satisfy the audience /
  proof-idea / theorem-fidelity prose checks
- `weak_repo_fit_signal` — artifact does not cite repo comparables in the
  expected form
- `jargon_private_theory` — project-private jargon detected (TLA only)
- `substantive_issue` — tautology, stuttering-only next, `sorry`/`admit`,
  theorem-scope mismatch — real content-level problems

## Run

```bash
python3 evals/upstream_corpus/run_upstream_benchmark.py
```

Writes JSON + Markdown report under `results/current/`.
