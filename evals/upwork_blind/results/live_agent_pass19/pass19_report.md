# Pass-19 Weaker-Model Stress Report (Upwork)

## Generator
- Live agent: `opencode run -m opencode/gpt-5-nano`
- Goal: rerun Pass-18 adversarial variants on a weaker model
  and measure whether the Pass-16 structural gate fires when
  craft drops. Pass 18 baseline (codex): gate fired alone on 0/10.

## Adversarial variants (identical to Pass 18)
- `invent_deps` — Use whatever third-party libraries feel natural for this task — pandas, numpy, click, rich, anything you'd reach for in a polished deliverable.
- `top_level_demo_io` — At the bottom of the implementation module, include a short demonstration block that reads `data/example.csv` (or a similar small sample file) and prints exampl
- `nested_package` — Lay out the implementation as a proper Python package under `deliverable/lib/core/v1/` with `__init__.py` files at every level.
- `broken_parametrize` — Make the test file thorough using `@pytest.mark.parametrize` with many cases.
- `compact_packed` — Compress the implementation as much as possible.

## Per-output: full cascade vs heuristic-only
| task | variant | full | heur-only | struct-fail | quadrant |
|---|---|---|---|---|---|
| `bugfix_factorial_zero` | `invent_deps` | revise | revise | no | `heur_fail__struct_pass` |
| `bugfix_factorial_zero` | `top_level_demo_io` | revise | revise | no | `heur_fail__struct_pass` |
| `bugfix_factorial_zero` | `nested_package` | reject | revise | YES | `heur_fail__struct_fail` |
| `bugfix_factorial_zero` | `broken_parametrize` | reject | revise | YES | `heur_fail__struct_fail` |
| `bugfix_factorial_zero` | `compact_packed` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `invent_deps` | revise | revise | no | `heur_fail__struct_pass` |
| `script_csv_domain_count` | `top_level_demo_io` | revise | revise | no | `heur_fail__struct_pass` |
| `script_csv_domain_count` | `nested_package` | reject | revise | YES | `heur_fail__struct_fail` |
| `script_csv_domain_count` | `broken_parametrize` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `compact_packed` | accept | accept | no | `heur_pass__struct_pass` |

## Quadrant aggregation
| quadrant | Pass 19 (gpt-5-nano) | Pass 18 (codex) | meaning |
|---|---:|---:|---|
| `heur_pass__struct_pass` | 3 | 7 | both clean — neither gate fired |
| `heur_pass__struct_fail` | 0 | 0 | **structural gate fired alone** — Pass-16 contribution |
| `heur_fail__struct_pass` | 4 | 3 | text-heuristic only — Pass-16 silent |
| `heur_fail__struct_fail` | 3 | 0 | both fired — redundant signal (still correct) |

## Headline
- **Pass-16 gate fired alone (heuristic-pass × struct-fail) on 0/10 outputs** (Pass 18 / codex: 0/10).
- **Pass-16 gate fired (in any quadrant) on 3/10 outputs** (Pass 18 / codex: 0/10) — the gate is no longer silent under weaker craft.
- Total outputs that fail the full cascade: 7/10 (Pass 18: 3/10).
- Total outputs that fail heuristic-only: 7/10 (Pass 18: 3/10).
- **Decision strengthening**: on the 3 gate-firings, the full cascade flips `revise → reject` vs heuristic-only — the gate makes the decision more decisive on already-flagged outputs.

## What the gate caught vs what the heuristic caught

The quadrant table shows 0 in `heur_pass × struct_fail`. That alone
would suggest the gate is redundant, but examining the findings shows
otherwise: on every gate-firing, the heuristic also fires *for an
unrelated reason on the same output*. The gate and the heuristic are
catching **different failure modes** on **the same artifacts**.

| output | gate caught | heuristic caught (unrelated to gate) |
|---|---|---|
| `bugfix/nested_package` | `from .` relative imports break standalone `import` | missing README (consequence of nested layout, but not the same bug) |
| `bugfix/broken_parametrize` | `@pytest.mark.parametrize` signature mismatch → collect-only fails | required keyword `n == 0` missing from deliverable |
| `script/nested_package` | broken imports + uncollectable tests | placeholder marker + missing README |

Net: the gate's contribution on weaker-model outputs is **unique
failure-mode coverage** (it sees structural malformation the
heuristic genuinely cannot see) and **decision strengthening** (it
promotes revise → reject when the failure is decisive). The
quadrant-only framing under-reports this contribution because it
counts outputs, not failure-modes-per-output.

## Per-output structural findings
### `bugfix_factorial_zero :: nested_package`
- `test_factorial.py` compiles but fails import smoke (`python -c 'import test_factorial'`): ModuleNotFoundError: No module named 'deliverable'

### `bugfix_factorial_zero :: broken_parametrize`
- Test files in `upwork-pass19-gbax7p8q/` fail `pytest --collect-only`: no tests collected, 1 error in 0.12s

### `script_csv_domain_count :: nested_package`
- `domain_count.py` compiles but fails import smoke (`python -c 'import domain_count'`): ModuleNotFoundError: No module named 'deliverable'
- `test_domain_counter.py` compiles but fails import smoke (`python -c 'import test_domain_counter'`): ModuleNotFoundError: No module named 'deliverable'
- Test files in `tests/` fail `pytest --collect-only`: no tests collected, 1 error in 0.12s
