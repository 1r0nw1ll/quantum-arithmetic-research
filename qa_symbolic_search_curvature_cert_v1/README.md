# QA Symbolic Search Curvature Cert v1

Pins a symbolic-search curvature scalar `kappa` together with the substrate curvature `H_QA` and an optimizer update-rule witness.

## Definitions

- `G = e*e + d*d` (use `*`, not `**`)
- `F = b*a`
- `eps = 1e-12`
- `H_raw = 0.25 * (F/(G+eps) + e*d/(a+b+eps))`
- `H_QA = abs(H_raw) / (1 + abs(H_raw))`
- Update witness: `p_after = p_before - lr * sym_gain * H_QA * grad`
- `kappa = 1 - abs(1 - lr * sym_gain * H_QA)`

## Gates

- Gate A (substrate): recompute `H_raw`, `H_QA` and pin `claimed.H_QA`.
- Gate B (update): strict `sym_gain ∈ (0,2]` and pin the update witness.
- Gate C (kappa): recompute `kappa` and pin it to `claimed.kappa`.

## Run

```bash
python qa_symbolic_search_curvature_cert_v1/validator.py --self-test
```

