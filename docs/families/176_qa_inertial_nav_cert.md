# Family [176] QA_INERTIAL_NAV_CERT.v1

## One-line summary

Formal proof that QA T-operator navigation has zero computational drift vs classical INS O(epsilon * sqrt(N)) error growth.

## Mathematical content

Classical INS per step: x += d * sin(theta) + noise. After N steps with per-step noise epsilon: RMS error = epsilon * d * sqrt(N).

QA per step: (b,e) -> (e, b+e) mod m. After N steps: T^N * (b0,e0) mod m. Error = 0 for all N.

Error budget at 10000 steps:

| Noise source | sigma | Classical drift | QA drift |
|---|---|---|---|
| IEEE 754 ULP | 1e-15 | ~1e-10 m | 0 |
| Trig table | 1e-10 | ~6e-6 m | 0 |
| MEMS IMU | 1e-6 | ~0.06 m | 0 |
| Cheap IMU | 1e-3 | ~57 m | 0 |

The ratio classical/QA diverges to infinity. Theorem NT: the ONLY error in QA nav is observer projection at input/output boundaries.

## Cross-references

- [163] QA_DEAD_RECKONING — T-operator mechanics
- [164] QA_GNOMONIC_RT — gnomonic chart for waypoints
- [168] QA_ELLIPSOID_GEODESIC — curvature of navigation surface

## Validator checks

| Check | Description |
|-------|-------------|
| IN_1 | schema_version == 'QA_INERTIAL_NAV_CERT.v1' |
| IN_QA_EXACT | T-operator `T^n(b0,e0)` matches the declared final state exactly |
| IN_DRIFT | classical_error genuinely recomputed via a 300-trial averaged Monte Carlo simulation and compared to the declared value within 30% (hardened 2026-07-06 — was fixture-trusted, only checked positivity) |
| IN_ZERO | qa_error == 0 for all witnesses |
| IN_RATIO | classical/QA ratio documented |
| IN_W | ≥3 route witnesses |
| IN_F | fail detection |

## Fixtures

| Fixture | Result |
|---------|--------|
| `qa_inertial_nav_cert_v1/fixtures/in_pass_error_budget.json` | PASS |
| `qa_inertial_nav_cert_v1/fixtures/in_fail_nonzero_qa.json` | FAIL |

## Verification Note (2026-07-06)

Independently reconfirmed `T^100(1,1) mod 24 = (5,8)` and
`T^1000(1,1) = T^10000(1,1) mod 24 = (13,16)` exactly, matching declared
values — `IN_QA_EXACT` was already genuinely computed and correct.

**Found and fixed a real fixture-trusting gap**: the module defines a
genuine Monte Carlo simulator, `classical_dr_error()`, but `validate()`
never called it — `IN_DRIFT` only checked that the declared
`classical_error` was positive. Worse, when I actually ran
`classical_dr_error()` with the fixture's own declared parameters, the
shipped values turned out to be a **single, unconverged Monte Carlo
sample** that did not reproduce the theorem it's meant to demonstrate:
the doc/fixture claimed "~√10 scaling" from 100→1000 steps, but the
actual declared numbers only grew ~1.54x (and ~2.19x from 1000→10000),
nowhere near the theoretical √10≈3.16x. A single random-walk realization
doesn't reliably show the ensemble-average scaling law — you need to
average over many trials for that.

Added `classical_dr_error_rms()` (300-trial averaged RMS) and wired
`IN_DRIFT` to genuinely recompute and compare against it (30% relative
tolerance, given inherent stochasticity). The 300-trial average
reproduces √10 scaling to within ~3% (observed ratios 3.06x and 3.14x
vs theoretical 3.16x) — regenerated the fixture with these converged
values. The ULP noise level (σ=1e-15, at float64 epsilon) is
intentionally left unsimulated/unverified — Monte Carlo averaging isn't
meaningful that close to floating-point precision limits — and is
clearly marked illustrative-only in the fixture.

Verified the hardened check rejects the old, unconverged values (planted
regression test: declared 57.1m vs genuine recompute 141m, correctly
flagged at 59.5% relative difference).

Also fixed a stale docstring typo: the validator's header comment said
"family [170]", not matching its actual `qa_meta_validator.py`
registration as [176].
