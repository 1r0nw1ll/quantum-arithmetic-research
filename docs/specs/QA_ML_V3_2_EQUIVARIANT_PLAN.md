# QA-ML v3.2 — Equivariant Model Plan

> Status: **design + build plan**. Per v3.1 findings
> (`docs/specs/QA_ML_V3_FINDINGS.md`), the clean stop is at v3.1's
> partial-success result; v3.2 is the explicit research project Will
> Dale greenlit 2026-05-15 to push `rediscover_277` toward 0.95.

> Primary source for the algebra-aware ML pattern: Pepe, A. (2025),
> *Machine Learning with Geometric Algebra: Multivectors for Modelling,
> Understanding and Computing*, PhD thesis, University of Cambridge.
> Companion file `/Users/player3/Downloads/2025-pepe.pdf`. Captured to
> Open Brain 2026-05-15.

## Equivariance precondition — VERIFIED

Empirical check (2026-05-15, before designing the architecture):

| Test | Scope | Match rate |
|---|---|---|
| Orbit period equivariance | m ∈ {7..25}, c ∈ {2..7}, all (b,e) | **900 / 900 = 100%** |
| Failure-mode equivariance, cert [277] scope | m = 15k, k ∈ {1..5}, missed pairs × c | **768 / 768 = 100%** |
| Failure-mode equivariance, cert [278] scope | 3 ∤ m overclaim pairs × c | 36 / 243 = 14.81% |

The asymmetry is structural: **orbit_period is invariant under
(b, e, m) → (cb, ce, cm) for every valid scaling**, but the divisor
shortcut's `(m // 3)` predicate is not (scaling changes whether m // 3
divides cb, ce, etc). Therefore the canonical orbit_family is fully
equivariant, but the shortcut's failure mode is only equivariant within
the regime where `m // 3` happens to commute with scaling — i.e., the
m = 15k under-count regime (cert [277]). The 3 ∤ m over-claim regime
(cert [278]) is NOT equivariant.

**Implication for v3.2:** an architecturally equivariant model will
push [277] rediscovery toward 1.0 by construction, but it will degrade
on [278]. The clean architecture is **hybrid**: enforce equivariance
on the [277] task and keep v3.1 features as a fallback for [278].

## Equivariance contract

A model `f: (b, e, m) → failure_mode` is **modulus-factor-equivariant**
iff for all valid (b, e, m) and all positive integers c such that
(cb, ce, cm) is also a valid state:

```text
f(b, e, m) = f(cb, ce, cm)
```

Equivalently, f factors through the canonical-representative map
`κ: (b, e, m) → (b / g, e / g, m / g)` where `g = gcd(b, e, m)`. Two
states are equivalent (and must receive the same prediction) iff their
canonical representatives are equal.

## Architecture options

### Option β.0 — Canonical features + standard ML head (v3.2.0)

Add canonical features `(b', e', m', g)` where `g = gcd(b, e, m)` and
primed variables are gcd-quotients. Append to the v3.1 packet. Train
sklearn DecisionTreeClassifier with the same protocol.

**Pros:** trivial to implement; tests whether the canonical
representation alone is enough.

**Cons:** doesn't enforce equivariance — the model could ignore the
canonical features and use raw (b, e, m). Not a structural fix.

### Option β.1 — Canonical-only model (v3.2.1)

Compute canonical (b', e', m'). Train and predict using ONLY canonical
features. The model never sees the original (b, e, m), only the
canonical representative. Equivariance is **enforced by construction**.

**Pros:** strictly equivariant. Predictions provably identical for
c-scaled inputs.

**Cons:** training set shrinks dramatically (many input states map to
the same canonical). Fallback for the non-equivariant [278] regime is
absent — the model can't see whether the original (b, e, m) had a
specific structure that the canonical loses.

### Option β.2 — Hybrid: canonical-equivariant + fallback (v3.2.2)

Two-stage model:
1. Compute canonical (b', e', m').
2. If canonical (b', e', m') is in the [277] regime (3 | m' and 15 | m'
   in some normalized sense), apply the equivariant head trained on
   canonical features only.
3. Otherwise (and especially in [278] regime), apply the v3.1
   non-equivariant model with raw features.

**Pros:** captures the regime-dependent equivariance correctly.
[277] benefits from the strict invariance; [278] is unaffected.

**Cons:** the "is canonical in [277] regime?" decision is a hand-
designed gate. A purer architecture would learn the gate.

### Option γ — Shared-parameter GCN (v3.3, deferred)

Build a GCN where edges between (b, e, m) and (cb, ce, cm) (when both
are valid states across distinct moduli) share weights. This is the
structural Pepe-style equivariance: parameters live in the quotient
space, the network operates on the original space, equivariance is a
property of the message-passing.

**Deferred to v3.3** if v3.2 results justify the structural complexity.
v3.2 ships at Option β.2 (hybrid).

## Phased implementation plan

### Phase 0 — Precondition verification (DONE)

Empirical equivariance check above. ✓

### Phase 1 — v3.2.0: canonical features as additions

Extend `qa_features_v3.py` with:

```python
g = gcd(gcd(b, e), m)
canonical_b = b // g
canonical_e = e // g
canonical_m = m // g
canonical_k = canonical_m // 15 if canonical_m % 15 == 0 else 0
canonical_g = g
```

5 new features. Packet grows from 25 to 30.

Re-run `04_orbit_structure_discovery.py` with `QA_ML_V3_OPTION=v3_2_0`.
Measure rediscover_277 with the augmented packet.

**Pass criterion**: rediscover_277 ≥ 0.70. If yes, the canonical
features are useful but the model still chooses freely; promote to
v3.2.2 hybrid.

### Phase 2 — v3.2.1: canonical-only model

Drop all non-canonical features. Train on canonical (b', e', m')
features only. Evaluate.

**Pass criterion**: rediscover_277 ≥ 0.95 on [277] test moduli;
rediscover_278 may degrade (acceptable — see hybrid).

### Phase 3 — v3.2.2: hybrid model

Combine: canonical-equivariant head for inputs whose canonical
representative is in the [277] regime; v3.1 non-canonical head for
everything else. Gate by `canonical_m % 15 == 0`.

**Pass criterion**: rediscover_277 ≥ 0.95 AND rediscover_278 ≥ 0.95
simultaneously. **This is the v3.2 success target.**

### Phase 4 — Findings synthesis

Update `docs/specs/QA_ML_V3_FINDINGS.md` (or add v3.2-specific findings)
with results, interpretation, and decision on v3.3 (structural GCN) vs
closure.

## Test split (held constant across phases)

```text
M_train_v3_2 = [9, 10, 11, 12, 15, 18, 20, 21, 24, 25, 27, 30, 36, 45,
                60, 90, 150]                       # 17 moduli
M_test_v3_2  = [7, 8, 13, 33, 75, 105, 120]        # 7 moduli incl. m=75
                                                   # boundary + m=8/m=105/m=120
```

`m=75` is the canonical [277] held-out test: it must extrapolate
across factor structures via the canonical map. `m=8` is the [278]
boundary. `m=105` (3·5·7) and `m=120` (2³·3·5) test factor-pattern
breadth.

## Success criteria

| Tier | Criterion | Action |
|---|---|---|
| **Strong** | rediscover_277 ≥ 0.95 AND rediscover_278 ≥ 0.95 AND m=75 = 1.000 AND m=8 = 1.000 | Close v3 thread cleanly; write canonical-equivariance findings note. |
| **Partial** | rediscover_277 ≥ 0.80 with m=75 ≥ 0.5 OR rediscover_278 ≥ 0.95 | Promote to v3.3 (structural GCN) or accept and close. |
| **Null** | No phase produces rediscover_277 ≥ 0.5 | Report negative result. Equivariance hypothesis fails. |

## Decision gates

- After Phase 1: if rediscover_277 doesn't beat v3.1 (0.458), abort and
  report. The canonical features are then either redundant with
  k-quotient or actively misleading.
- After Phase 2: if canonical-only model can't hit 0.95 on the equiv-
  ariant [277] regime, the hypothesis itself is wrong — there's a
  hidden non-equivariance we haven't detected.
- After Phase 3: hybrid is the v3.2 ship. Either ≥ 0.95 (strong) or
  reported partial.

## Non-goals (do not pursue in v3.2)

- Building the full shared-parameter GCN (v3.3 candidate).
- Adding a new cert family. The v3 thesis rules this out.
- Cross-task multi-head model (T1 + T2). v3.2 is T1-only.
- Symbolic regression / PySR / RIPPER mining beyond decision-tree
  distillation. v3.2 uses CART for consistency with v3.1.

## File layout

```text
tools/qa_ml/
  qa_features_v3.py              # extended with canonical features (phase 1)
  qa_equivariant_v3_2.py         # hybrid model class (phase 3)

experiments/qa_ml/
  04_orbit_structure_discovery.py  # extended with v3_2_0, v3_2_1, v3_2_2 options
  results_v3_2_0.json + tree text
  results_v3_2_1.json + tree text
  results_v3_2_2.json + tree text

docs/specs/
  QA_ML_V3_2_EQUIVARIANT_PLAN.md   # this file
  QA_ML_V3_FINDINGS.md             # extended with v3.2 results
```

No new benchmark protocol JSON unless the v3 plan's protocol needs
restructuring (probably not).

## References

- v3.1 result (commit `7792e14`): rediscover_277 = 0.458 with m=150 in
  training; m=75 = 0.500. Established that k-quotient features help
  within-structure but not between-structure.
- Pepe (2025): the GA-equivariant pattern across rotations, proteins,
  pose, PDEs. v3.2 is the QA-modular analog: enforce algebraic
  invariance structurally rather than via training-set coverage.
- Cert [277] / [278]: the rediscovery targets.
- Wall (1960): the orbit-period framing; DOI: 10.1080/00029890.1960.11989541.
