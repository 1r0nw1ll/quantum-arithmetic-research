# [157] QA PIM Kernel Cert

**Schema**: `QA_PIM_KERNEL_CERT.v1`
**Status**: PASS (1 PASS + 1 FAIL fixture)

## What it certifies

Correctness of QA-native Processing-In-Memory (PIM) kernels extracted from the `qa_graph_pim_repo_v2` vault artifact (Sept 2025) into `qa_lab/qa_pim/`.

### Kernels

| Kernel | Purpose |
|--------|---------|
| `RESIDUE_SELECT` | Select vertices by sector ID mask |
| `TORUS_SHIFT` | Shift values on phase torus modulo P |
| `RADD_m` | Residue addition modulo m (coordinate-level, 0-indexed) |
| `RMUL_m` | Residue multiplication modulo m (coordinate-level, 0-indexed) |
| `MIRROR4` | Fold/mirror quadrant operation |
| `ROLLING_SUM_PHASE` | Circular rolling sum over phase ring |

### CRT (Chinese Remainder Theorem)

General solver handling non-coprime moduli via extended GCD. Key QA case: `x = 1 (mod 9)` and `x = 1 (mod 24)` yields `x = 1 (mod 72)` since `lcm(9,24) = 72`.

### A1 compliance note

`RADD_m` and `RMUL_m` operate at the **coordinate layer** (0-indexed, range `0..m-1`). QA state-level A1 enforcement (range `1..m`) is handled by `qa_observer.core.qa_mod()` at a higher layer. PIM kernels are analogous to CPU instructions.

## How to run

```bash
# Unit tests (24 tests)
cd qa_lab && PYTHONPATH=. python -m pytest qa_pim/tests/ -v

# Validator self-test
cd qa_alphageometry_ptolemy/qa_pim_kernel_cert_v1
python qa_pim_kernel_cert_validate.py --self-test
```

## What breaks

- CRT returning wrong solution (modulus verification fails)
- Kernel witness input/output mismatch
- Missing required kernel witnesses (RESIDUE_SELECT, TORUS_SHIFT, ROLLING_SUM_PHASE)

## Sources

- Hardy, G.H. & Wright, E.M. (2008), *An Introduction to the Theory of Numbers*, 6th ed., Oxford University Press, ISBN 978-0-19-921986-5, Ch. II — Chinese Remainder Theorem, extended GCD. This cert is a from-scratch internal CRT/kernel implementation, not an external-paper reproduction.

## Verification Note (2026-07-06)

Ran the real unit test suite (`cd qa_lab && PYTHONPATH=. python -m pytest
qa_pim/tests/ -v`) — all **24 tests pass** genuinely against the real
`qa_lab/qa_pim/` implementation (`crt.py`, `kernels.py`, `schema.py`,
`graph.py`).

Independently reproduced every value in the fixture by directly calling
the real code: `crt.crt_join_general(a1,m1,a2,m2)` for all 4 CRT cases
(coprime, non-coprime solvable, non-coprime unsolvable, and the
QA-relevant mod-9/mod-24 case) and all three kernel witnesses
(`RESIDUE_SELECT`, `TORUS_SHIFT`, `ROLLING_SUM_PHASE`) — every declared
value matched the live computation exactly. No bugs found in the
fixture's actual numeric claims.

**Hardened the validator anyway**: `PIM_CRT` previously only checked
that a declared `x` satisfied its own declared congruences (never
checking `lcm`, and never independently verifying `solvable`/
unsolvable-case correctness), and `PIM_KERNEL` only checked that
input/output fields were *present*, never that the output was *correct*.
Added live imports of `qa_lab/qa_pim/crt.py` and `kernels.py` (with a
graceful degrade-to-warning if `qa_lab` isn't importable in a given
environment) so both checks now genuinely recompute against the real
code and compare to every declared field (`x`, `lcm`, `solvable`,
kernel `output`/`output_first`/`output_length`). Verified the hardened
checks correctly reject a planted wrong CRT `x` and a planted wrong
`TORUS_SHIFT` output. `--self-test` passes on both fixtures with the
live recompute path active (no import-fallback warnings).
