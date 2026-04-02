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
