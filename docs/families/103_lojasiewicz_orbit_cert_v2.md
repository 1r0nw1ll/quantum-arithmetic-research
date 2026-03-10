# Family [103] — QA Lojasiewicz Orbit Descent Cert v2 (Intrinsic)

**Cert root:** `qa_lojasiewicz_orbit_cert_v2/`
**Validator:** `qa_lojasiewicz_orbit_cert_v2/validator.py --self-test`
**Schema:** `QA_LOJASIEWICZ_ORBIT_CERT.v2.schema.json`
**Supersedes:** Family [102] (v1 with explicit `h_crit_witnessed`)

## What changed from v1

v1 required `h_crit_witnessed: true` as an explicit witness field.
v2 removes that field entirely. H-crit is now **derived**, not witnessed.

The derivation (B3, companion note `b3_intrinsic_theorem.md`):

1. **Lemma 1** (Łojasiewicz → H-crit): if V_s > 0 and (H-Łoj) holds,
   then `||∇L(w_s)||² ≥ 2μ V_s^α > 0`, so H-crit holds automatically.

2. **Lemma 2** (fixed-point propagation): if V_s = 0 at any intermediate step,
   gradient descent fixes at the minimizer (`∇L(w*) = 0` → `w_{s+k} = w_s`),
   so V_{t+L} = 0, contradicting Case B (V_{t+L} > 0). Hence all intermediate
   V_s > 0 in Case B.

Together: in Case B, H-crit is a theorem, not a hypothesis. Only `phi_t > 0`
(the orbit window starts from a non-minimizer) is required — and this is already
enforced by the schema (`exclusiveMinimum: 0`).

## What it certifies

Same claim as v1: for β-smooth, Łojasiewicz-conditioned L under QA orbit,
either exact convergence occurs within the orbit window, or:

```
phi_{t+L} <= phi_t - (1-alpha) * C(O)
```

where `phi_t = V_t^{1-alpha}` and `C(O) = sum_t 2*mu*eta_eff_t*(1 - beta/2*eta_eff_t)`.

## Gates (vs v1)

| Gate | v1 | v2 |
|---|---|---|
| Schema | QA_LOJASIEWICZ_ORBIT_CERT.v1 | QA_LOJASIEWICZ_ORBIT_CERT.v2 |
| A | Orbit feasibility | Orbit feasibility (unchanged) |
| 2D | Recompute C(O); verify claimed.C_O | Recompute C(O); verify claimed.C_O (unchanged) |
| B | `h_crit_witnessed = true` required | **Removed** — H-crit derived from phi_t > 0 |
| C | phi bound valid | phi bound valid (unchanged) |
| D | convergence_orbits_bound correct | convergence_orbits_bound correct (unchanged) |

## Witness fields (vs v1)

| Field | v1 | v2 |
|---|---|---|
| `family` | 102 | 103 |
| `orbit` | same | same |
| `alpha` | same | same |
| `phi_t > 0` | same | same (enforces V_t > 0 via schema) |
| `h_crit_witnessed` | required `true` | **removed** |
| `claimed` | same | same |

## Scope

The fixed-point argument (Lemma 2) applies to **exact** gradient descent.
For stochastic GD, the stochastic gradient at w* may be non-zero, breaking
the fixed-point property. The update rule in the theorem is w_{s+1} = w_s − η∇L(w_s)
(exact gradient throughout).

## Canonical fixture

Same orbit as [102] v1: eta_eff=[0.5,0.5,0.5,0.5], mu=0.1, beta=1.0,
alpha=0.5, phi_t=2.0. C(O)=0.3, phi_tL_bound=1.85, convergence_orbits_bound=14.
Note the absence of `h_crit_witnessed`.
