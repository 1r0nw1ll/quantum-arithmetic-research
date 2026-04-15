# QA Benchmark Protocol (Machine Tract)

Defines **QA_BENCHMARK_PROTOCOL.v1**: an enforceable contract for any
benchmark that compares a QA method against baseline methods on named
datasets with declared metrics, source mapping, SOTA/null-result
expectations, ablation, and reproducibility.

Authority: `EXPERIMENT_AXIOMS_BLOCK.md` (Part B, B1–B4).

## What it is

Every benchmark must produce a concrete object:

`B = (Q, L, D, P, C, F, M, S, T, A, R)` where

- `Q`: QA method (name, description, observer projection)
- `L`: baseline list (non-empty, with implementation refs)
- `D`: dataset list (non-empty, with source IDs)
- `P`: parity contract (same seed, same split, same preprocessing)
- `C`: calibration provenance (learned-on, procedure, domain_of_origin)
- `F`: framework inheritance (`inherit` | `ported` | `novel`; prior_cert required if not novel)
- `M`: metrics (non-empty list)
- `S`: source mapping cross-reference (`primary_source` must occur in `theory_doc`)
- `T`: SOTA baseline threshold, or explicit null-result acceptance
- `A`: ablation contract
- `R`: reproducibility manifest

## Why this cert family exists

The 2026-04-13 qa_detect incident: a hand-tuned cmap calibrated for
finance-return clusters was carried as the default and silently
killed every non-finance benchmark (0/5 tabular, 0/4 NAB). The bug
was not in the QA method — the calibration domain was undeclared and
the recalibration step was skipped. Gate 3 (calibration_provenance)
makes that class of incident non-silent.

The 2026-04-05 Bearden text-transfer incident: the QA method was
scored against a domain where the observer framework (windows,
feature streams, clustering scheme) differed from the prior working
cert, but that difference was not enumerated. Gate 4 (framework
inheritance) makes that deviation explicit.

## Files

- `schema.json` — JSON Schema for `QA_BENCHMARK_PROTOCOL.v1`
- `validator.py` — Gate 1–9 validator
- `canonical_benchmark_protocol.json` — canonical shared object
- `fixtures/valid_min.json` — minimal passing fixture (qa_detect tabular)
- `fixtures/invalid_missing_calibration_provenance.json` — intentionally failing fixture

## How to run

```bash
python qa_benchmark_protocol/validator.py --self-test
python qa_benchmark_protocol/validator.py qa_benchmark_protocol/fixtures/valid_min.json
```

## How benchmark scripts declare compliance

Inline form:

```python
BENCHMARK_PROTOCOL_REF = "path/to/benchmark_protocol.json"
```

Or place `benchmark_protocol.json` in the same directory as the script.
`qa_axiom_linter.py` rule `BENCH-1` gates both forms.
