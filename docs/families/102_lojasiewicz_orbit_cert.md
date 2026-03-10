# Family [102] — QA Lojasiewicz Orbit Descent Cert

**Cert root:** `qa_lojasiewicz_orbit_cert_v1/`
**Validator:** `qa_lojasiewicz_orbit_cert_v1/validator.py --self-test`
**Schema:** `QA_LOJASIEWICZ_ORBIT_CERT.v1.schema.json`

## What it certifies

This cert closes the theory-to-cert loop for the proved Lojasiewicz orbit-window
theorem (paper §8.3, B2a). It certifies per-orbit-window phi-contraction for
smooth, Lojasiewicz-conditioned losses under a QA cosmos orbit update:

```
phi_{t+L} <= phi_t - (1-alpha) * C(O)
```

where `phi_t = V_t^{1-alpha}` is the Lojasiewicz change-of-variable
(`V_t = L(w_t) - L*`) and `C(O)` is the orbit-computable sum:

```
C(O) = sum_{t=0}^{L-1} 2*mu * eta_eff_t * (1 - beta/2 * eta_eff_t)
```

This is the additive phi-decrement version of the §8.2 multiplicative
contraction: instead of `rho_PL * V_t`, the orbit contributes a fixed
phi-budget reduction `(1-alpha)*C(O)` per window, independent of V_t.

## Theoretical grounding

The theorem proved in B1 + B2a (papers `b1_phi_lemma.md`, `b2_orbit_theorem.md`):

- **B1 (one-step):** Under H-smooth + H-Loj + H-crit, phi_{t+1} <= phi_t - (1-alpha)*c_t
  where c_t = 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t) > 0 (orbit-computable).
- **B2a (orbit-window):** Telescope over L steps to get the phi-contraction bound.
- **B2b (genericity):** H-crit holds for a.e. initial condition w_0 when L is semi-algebraic.

**H-crit** (`h_crit_witnessed`) is an explicit witness-side boolean in v1 (not derived
from the orbit alone — deriving it intrinsically is the remaining open problem).

## Witness fields

| Field | Type | Description |
|---|---|---|
| `orbit.eta_eff` | float[] | Effective step sizes eta_eff_t for t=0..L-1 |
| `orbit.mu` | float > 0 | Lojasiewicz constant |
| `orbit.beta` | float > 0 | Smoothness constant |
| `alpha` | float in (0,1) | Lojasiewicz exponent |
| `phi_t` | float > 0 | Initial phi_t = V_t^{1-alpha} |
| `h_crit_witnessed` | bool | H-crit: nabla L(w_s) != 0 for all s in orbit |
| `claimed.C_O` | float > 0 | Claimed C(O) — recomputed by Gate 2D |
| `claimed.phi_tL_bound` | float > 0 | Claimed upper bound on phi_{t+L} |
| `claimed.convergence_orbits_bound` | int >= 1 | ceil(phi_t / ((1-alpha)*C(O))) |

## Gates

| Gate | Check | Fail type |
|---|---|---|
| Schema | Validate against QA_LOJASIEWICZ_ORBIT_CERT.v1 | SCHEMA_INVALID |
| A | Orbit feasibility: eta_eff_t in (0, 2/beta) for all t | ORBIT_INFEASIBLE |
| 2D | Recompute C(O) = sum 2*mu*eta_eff_t*(1-beta/2*eta_eff_t); verify claimed.C_O | CO_MISMATCH |
| B | Verify h_crit_witnessed = true | HCRIT_NOT_WITNESSED |
| C | Verify phi_tL_bound <= phi_t - (1-alpha)*C(O) | PHI_BOUND_INVALID |
| D | Verify convergence_orbits_bound = ceil(phi_t / ((1-alpha)*C(O))) | ORBITS_BOUND_MISMATCH |

## Comparison with §8.2 (family [89]/[101] regime)

| | §8.2 (PL, alpha=1) | §8.3 (Lojasiewicz, alpha<1) |
|---|---|---|
| Orbit quantity | rho_PL = product(1 - c_t) | C(O) = sum(c_t) |
| Per-orbit statement | V_{t+L} <= rho_PL * V_t (multiplicative) | phi_{t+L} <= phi_t - (1-alpha)*C(O) (additive) |
| Convergence rate | Geometric (exponential in orbits) | Polynomial (phi-budget / decrement) |
| H-crit needed | No (PL guarantees no flat regions) | Yes (explicit hypothesis in v1) |

## Canonical fixture

- `orbit.eta_eff`: [0.5, 0.5, 0.5, 0.5] (L=4)
- `orbit.mu`: 0.1, `orbit.beta`: 1.0
- `alpha`: 0.5, `phi_t`: 2.0
- `C(O)` recomputed: 0.3 (c_t = 0.075 per step)
- `phi_tL_bound`: 1.85 (= 2.0 - 0.5*0.3)
- `convergence_orbits_bound`: 14 (= ceil(2.0 / (0.5*0.3)))
