# v1 Blind-Eval Harness — Baseline Report

_Total wall-clock: 12.0s_

## Summary
| suite | status | summary | elapsed |
|---|---|---|---:|
| `_blind_core` | **OK** | 12 self-test checks; failures=0 | 0.06s |
| `blind_benchmark` | **OK** | 30 labeled, accuracy=100.00%, false_accept=0, false_reject=0 | 5.83s |
| `deception_regression` | **OK** | 34 fixtures, MATCH=34, NEW_FALSE_ACCEPT=0, NEW_FALSE_REJECT=0, KNOWN_GAP=0 | 3.18s |
| `upstream_benchmark` | **OK** | TLA intrinsic accept=54.5% (99 cases); Lean intrinsic accept=100.0% (128 cases) | 1.08s |
| `charitable_adapter` | **OK** | 45 revise baseline; 21 flipped→accept; 24 still revise; 0 regressed | 1.72s |
| `swe_bench_calibration` | **OK** | designed truth 8/8; executed truth 16/16 TP, 0 FA, 0 FR | 0.09s |

**Overall:** OK — all suites within acceptance bounds

## Suite details
### _blind_core
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/_blind_core/__init__.py`
- Exit: 0, elapsed: 0.06s
- Status: **OK** — 12 self-test checks; failures=0

### blind_benchmark
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/blind_benchmark/benchmark_current_corpus.py`
- Exit: 0, elapsed: 5.83s
- Status: **OK** — 30 labeled, accuracy=100.00%, false_accept=0, false_reject=0
```
# Blind Corpus Benchmark Sweep

## Cross-Domain Summary
- Total labeled fixtures: 30
- Overall accuracy: 100.00%
- False accept count: 0
- False reject count: 0

```

### deception_regression
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/deception_regression/run_regression.py`
- Exit: 0, elapsed: 3.18s
- Status: **OK** — 34 fixtures, MATCH=34, NEW_FALSE_ACCEPT=0, NEW_FALSE_REJECT=0, KNOWN_GAP=0

### upstream_benchmark
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/upstream_corpus/run_upstream_benchmark.py`
- Exit: 0, elapsed: 1.08s
- Status: **OK** — TLA intrinsic accept=54.5% (99 cases); Lean intrinsic accept=100.0% (128 cases)

### charitable_adapter
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/upstream_corpus/charitable_adapter.py`
- Exit: 0, elapsed: 1.72s
- Status: **OK** — 45 revise baseline; 21 flipped→accept; 24 still revise; 0 regressed

### swe_bench_calibration
- Command: `/usr/bin/python3 /home/player2/signal_experiments/evals/swe_bench_blind/run_calibration.py`
- Exit: 0, elapsed: 0.09s
- Status: **OK** — designed truth 8/8; executed truth 16/16 TP, 0 FA, 0 FR
