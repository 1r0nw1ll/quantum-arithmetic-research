# Family [224] QA_BENCHMARK_PROTOCOL.v1

## One-line summary

An enforceable design contract for benchmarks that compare a QA method
against baseline methods on named datasets with declared metrics —
validated by a five-gate JSON-schema validator and enforced at file
level by the linter rule `BENCH-1`.

## Why

Two recent incidents defined the scope:

**2026-04-13 — qa_detect cmap silent failure.** A hand-tuned cluster
map calibrated for finance-return clusters was carried as the default
and silently killed every non-finance benchmark (0/5 tabular, 0/4
NAB). The bug was not in the QA method; the calibration domain was
undeclared and the recalibration step was skipped. Gate 3
(`calibration_provenance`) makes that class of incident non-silent.

**2026-04-05 — Bearden text-transfer framework mismatch.** A QA
method was scored against a domain where the observer framework
(windows, feature streams, clustering scheme) differed from the prior
working cert, but the difference was not enumerated. Three attempts
were spent before the framework-vs-domain split was identified. Gate
4 (`framework_inheritance`) makes deviations explicit at design time.

## What it certifies

For every benchmark script there exists a concrete object

`B = (Q, L, D, P, C, F, M)`

with

- `Q` QA method (name, description, observer projection)
- `L` non-empty baseline list (name + implementation reference)
- `D` non-empty dataset list (name + source)
- `P` parity contract (same seed, same split, same preprocessing — all true)
- `C` calibration provenance (`learned_on`, `procedure`, `domain_of_origin`)
- `F` framework inheritance (`inherit` | `ported` | `novel`; `prior_cert` required if not novel)
- `M` non-empty metric list

## Gates (validator)

1. Schema Validity — conforms to `qa_benchmark_protocol/schema.json`
2. Baseline Parity — all three `parity_contract.same_*` flags `true`, baselines ≥ 1
3. Calibration Provenance — `procedure` + `learned_on` + `domain_of_origin` non-empty
4. Framework Inheritance — `mode` declared; `prior_cert` required if `inherit` or `ported`
5. Metrics Non-Empty — metrics list ≥ 1 with non-empty strings

## How to run

```bash
python qa_benchmark_protocol/validator.py --self-test
python qa_benchmark_protocol/validator.py path/to/benchmark_protocol.json
```

## How scripts declare compliance

Inline: `BENCHMARK_PROTOCOL_REF = "path/to/benchmark_protocol.json"`
or place `benchmark_protocol.json` next to the script. Linter rule
`BENCH-1` gates both forms on any file importing sklearn baselines
(`sklearn.ensemble`, `sklearn.neighbors`, `sklearn.svm`,
`sklearn.linear_model`, `sklearn.naive_bayes`, `sklearn.tree`,
`sklearn.cluster`) together with either a metric call
(`roc_auc_score`, `adjusted_rand_score`, `normalized_mutual_info_score`,
`f1_score`, `accuracy_score`, `precision_recall_fscore_support`) or
an explicit `baselines = [...]` / `methods = {...}` structure.

## Authority

`EXPERIMENT_AXIOMS_BLOCK.md` Part B (B1–B4). Mirrors the enforcement
shape of family [35] QA Mapping Protocol.
