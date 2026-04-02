# Family [163] QA_DEAD_RECKONING_CERT.v1

## One-line summary

The QA T-operator provides exact dead reckoning on a mod-m lattice with zero computational drift, replacing classical sin/cos navigation that accumulates float error at every leg.

## Mathematical content

### Classical DR vs QA DR

| | Classical DR | QA DR |
|---|---|---|
| State | (x, y) floating point | (b, e) integers mod m |
| Step | x += d·sin(θ), y += d·cos(θ) | (b,e) → (e, b+e) mod m |
| N steps | N sin/cos evaluations | T^N via matrix exponentiation |
| Error | ~ε·√N·d (accumulates) | 0 (exact integer arithmetic) |
| Closed loop | drift > 0 always | returns to exact origin |

### T-operator as navigation

The QA T-operator `T = [[0,1],[1,1]]` is the Fibonacci shift. In A1-compliant form:

```
b_{k+1} = ((e_k - 1) % m) + 1
e_{k+1} = (((b_k + e_k) - 1) % m) + 1
```

After k steps: compute `T^k` via augmented 3x3 matrix exponentiation (the A1 encoding introduces an affine +1 shift, handled by augmenting with a constant row).

### Pisano periodicity

The T-operator has period π(m) (Pisano period):
- π(24) = 24 (mod-24 cosmos orbit)
- π(9) = 24 (mod-9 theoretical)
- After π(m) legs, the navigator returns to the exact starting state.

### Three chromogeometric metrics per bearing

Every QA direction (d, e) simultaneously carries:

| Metric | Formula | Navigation use |
|--------|---------|---------------|
| G = d²+e² | Blue / Euclidean | Position fix (GPS) |
| F = d²-e² | Red / Minkowski | Hyperbolic fix (LORAN/Decca) |
| C = 2de | Green / Area | Cross-track error |

Identity: C² + F² = G² (Wildberger Theorem 6)

### Compass rose mod-24

- 24-cycle **cosmos** = full circumnavigation bearings (all 8 principal winds)
- 8-cycle **satellite** = reduced navigational states (e.g., (8,8))
- 1-cycle **singularity** = no bearing / fixed point ((24,24))

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_dead_reckoning_cert_v1
python qa_dead_reckoning_cert_validate.py --self-test
```

## Validator checks

| Check | Description |
|-------|-------------|
| DR_1 | schema_version == QA_DEAD_RECKONING_CERT.v1 |
| DR_TOP | T-operator iteration produces correct state evolution |
| DR_EXACT | Matrix exponentiation matches step-by-step iteration |
| DR_DRIFT | Classical float DR accumulates error; QA has zero |
| DR_CHROMO | C²+F²=G² for all witness directions |
| DR_COMPASS | Mod-24 orbit classification of compass bearings |
| DR_W | At least 3 route witnesses |
| DR_F | Fail detection |

## Fixtures

| Fixture | Result | Content |
|---------|--------|---------|
| dr_pass_routes.json | PASS | 4 routes (100/1000/500/48 steps), 4 chromo checks, 10 compass entries |
| dr_fail_wrong_state.json | FAIL | 0-indexed arithmetic instead of A1 — off-by-one in final states |

## Key insight

Classical navigation error has three sources: measurement, computation, accumulation. QA eliminates the last two by construction. The only error is the observer projection at input (continuous bearing → QA direction) and output (QA state → chart position) — Theorem NT boundary crossed exactly twice.

## Applications

1. **Embedded/FPGA navigation** — no trig lookup tables needed, integer-only
2. **Inertial navigation systems** — accumulated trig error is THE failure mode; QA eliminates it
3. **GPS-denied environments** — underwater AUV, underground, contested — DR is the only option
4. **Historical reconstruction** — Babylonian/Polynesian navigators used integer-ratio methods = proto-QA DR

## Cross-references

- [135] QA_PYTHAGOREAN_TREE_CERT.v1 — Berggren tree = discrete geodesic navigation
- [125] QA_CHROMOGEOMETRY_CERT.v1 — C, F, G = green, red, blue quadrances
- [156] QA_WGS84_ELLIPSE_CERT.v1 — Earth = QA quantum ellipse
- [164] QA_GNOMONIC_RT_CERT.v1 — gnomonic chart for DR waypoints
