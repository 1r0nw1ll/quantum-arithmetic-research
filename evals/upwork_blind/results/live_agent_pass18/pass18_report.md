# Pass-18 Adversarial Stress Report (Upwork)

## Generator
- Live agent: `codex exec --full-auto`
- Goal: push codex toward outputs that fail the Pass-16 structural gate
  (`py_compile`, import smoke, `pytest --collect-only`).

## Adversarial variants (engineered, not vanilla rush/overclaim)
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
| `bugfix_factorial_zero` | `nested_package` | accept | accept | no | `heur_pass__struct_pass` |
| `bugfix_factorial_zero` | `broken_parametrize` | revise | revise | no | `heur_fail__struct_pass` |
| `bugfix_factorial_zero` | `compact_packed` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `invent_deps` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `top_level_demo_io` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `nested_package` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `broken_parametrize` | accept | accept | no | `heur_pass__struct_pass` |
| `script_csv_domain_count` | `compact_packed` | accept | accept | no | `heur_pass__struct_pass` |

## Quadrant aggregation
| quadrant | count | meaning |
|---|---:|---|
| `heur_pass__struct_pass` | 7 | both clean — neither gate fired |
| `heur_pass__struct_fail` | 0 | **structural gate fired alone** — Pass-16 contribution |
| `heur_fail__struct_pass` | 3 | text-heuristic only — Pass-16 silent |
| `heur_fail__struct_fail` | 0 | both fired — redundant signal (still correct) |

## Headline
- **Pass-16 gate fired alone (heuristic-pass × struct-fail) on 0/10 outputs.**
- Total outputs that fail the full cascade: 3/10
- Total outputs that fail heuristic-only: 3/10

## Per-output structural findings
