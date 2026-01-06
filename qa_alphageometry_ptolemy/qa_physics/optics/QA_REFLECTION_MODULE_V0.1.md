# QA Reflection Module v0.1 - Projection Probe

**Date:** 2026-01-05
**Status:** ✅ **OPERATIONAL** - Demo running, protocol compliance verified

---

## Objective Achieved

Created a **minimal, exact-arithmetic reflection demonstrator** that serves as a **projection probe**:
- ✅ Non-reducing rational arithmetic (QARational)
- ✅ Exact geometric invariants (quadrances, not distances)
- ✅ QA-time compliant (k = path length)
- ✅ Observer protocol satisfaction
- ✅ Deterministic transition logs
- ✅ Bounded state space exploration

---

## Architecture

```
qa_physics/optics/
├── __init__.py                        # Module declaration
├── qa_reflection_failures.py         # Optics-specific failure types
├── qa_reflection_problem.py          # QARational, Point2D, LineABC, ReflectionProblem
├── qa_reflection_state.py            # ReflectionState (satisfies QAStateProtocol)
├── qa_reflection_search.py           # Bounded search + transition logs
└── run_reflection_demo.py            # Multi-observer demonstration
```

---

## Core Abstractions

### QARational (Non-Reducing Rational)

**Purpose:** Respect QA non-reduction discipline

```python
@dataclass(frozen=True)
class QARational:
    n: int  # Numerator (never simplified)
    d: int  # Denominator (never simplified)

    # Operations preserve exact representation
    def __add__(self, other) -> QARational:
        return QARational(self.n * other.d + other.n * self.d,
                         self.d * other.d)
```

**Guarantees:**
- No GCD simplification
- Exact arithmetic (no float approximation until projection)
- Frozen (immutable)
- JSON-serializable as (n,d) tuple

### ReflectionProblem

**Components:**
- `S: Point2D` - Source point
- `T: Point2D` - Target point
- `mirror: LineABC` - Mirror as A*x + B*y + C = 0
- `u_min, u_max: int` - Parameter bounds

**Parameterization:**
```python
mirror.point_from_u(u: int) -> Point2D | ReflectionFailure
```
- Deterministic: x = u, y = -(A*u + C)/B
- Returns failure for degenerate cases

### ReflectionState (QAStateProtocol)

**Satisfies projection interface:**
```python
def state_id(self) -> str
def to_invariants(self) -> Dict[str, Any]
```

**Invariants exposed:**
- Points: S, M(u), T (as (n,d) tuples)
- Quadrances: Q_SM, Q_MT, Q_ST (exact squared distances)
- Mirror coefficients: A, B, C
- Parameter: u, u_min, u_max

**Key insight:** Observers receive exact invariants, not "angles" or "distances"

### SearchConfig & Candidate Generation

**QA-time harness:**
- Time = path length (k = |path| - 1)
- Moves: sigma_plus (u → u+1), sigma_minus (u → u-1)
- Bounded: u ∈ [u_min, u_max]
- Deterministic: shortest paths in 1D parameter space

**Output:**
```python
@dataclass(frozen=True)
class ReflectionCandidate:
    u: int
    state: ReflectionState
    path_u: List[int]            # QA path in parameter space
    path_states: List[ReflectionState]
    logs: List[TransitionLogEvent]  # Deterministic transition logs
```

---

## Demo Results

**Problem instance:**
```
Mirror: y = 0 (horizontal line)
Source S: (-10, 5)
Target T: (10, 6)
u range: [-20, 20] → 41 candidate states
```

**Observer comparison (sample u ∈ {-10, 0, 10}):**

| Observer | u | k (QA time) | t_obs | Q_SM | Q_MT | Q_ST |
|----------|---|-------------|-------|------|------|------|
| NullObserver | -10 | 10 | 10.0 | 25 | 436 | 401 |
| NullObserver | 0 | 0 | 0.0 | 125 | 136 | 401 |
| NullObserver | 10 | 10 | 10.0 | 425 | 36 | 401 |
| AffineTime(1.0,0.0) | -10 | 10 | 10.0 | 25 | 436 | 401 |
| AffineTime(1.0,0.0) | 0 | 0 | 0.0 | 125 | 136 | 401 |
| AffineTime(1.0,0.0) | 10 | 10 | 10.0 | 425 | 36 | 401 |
| AffineTime(0.5,10.0) | -10 | 10 | 15.0 | 25 | 436 | 401 |
| AffineTime(0.5,10.0) | 0 | 0 | 10.0 | 125 | 136 | 401 |
| AffineTime(0.5,10.0) | 10 | 10 | 15.0 | 425 | 36 | 401 |

**Key observations:**
1. **QA time is observer-independent:** k = |path| - 1 (Axiom T1)
2. **Continuous time is observer-dependent:** Different t_obs for same k
3. **Invariants are observer-independent:** Quadrances identical across observers
4. **Topology preserved:** All 41 states yield 41 distinct observations

---

## Failure Algebra

**Optics-specific failure types:**
- `GEOM_DEGENERATE` - Mirror invalid (A=B=0, vertical line in x-param)
- `OOB` - Parameter u out of configured bounds
- `ILLEGAL` - Reserved for general legality failures
- `INVARIANT` - Reserved for conservation law violations

**Design:**
```python
@dataclass(frozen=True)
class ReflectionFailure:
    fail_type: str
    detail: str
    meta: Dict[str, Any]
```

**Integration:**
- Failures are first-class values (not exceptions)
- Encoded in invariant packets when present
- Observers can interpret failure states

---

## Protocol Compliance Verified

**QAStateProtocol satisfaction:**
```python
✅ state_id() - Stable identifier: "ReflectionState(u=5)"
✅ to_invariants() - Dict[str, Any] with 10 exact invariants
✅ Works with all locked observers (Null, AffineTime)
✅ JSON-serializable observation output
```

---

## What This Enables

### 1. **Projection as Physics Question**

Instead of "does QA do optics?", ask:
> "Which observer projection makes angle(incidence) = angle(reflection) hold?"

This is **testable**, **falsifiable**, and **observer-dependent**.

### 2. **Multiple Projection Comparison**

Current observers (Null, AffineTime) preserve:
- Exact invariants (quadrances)
- State topology (41/41 distinct)
- QA-time structure (k deterministic)

But they don't compute "angles" yet - that requires a GeometryObserver.

### 3. **Null Model Baseline**

NullObserver provides baseline: raw invariants, no geometry interpretation.

Any "better" observer must demonstrate improved law preservation.

### 4. **Exact Arithmetic Path**

No floats in QA layer:
- QARational for all coordinates
- Quadrance (squared distance) for all metrics
- Only observer projection introduces approximation

---

## What's NOT Locked (Intentionally)

❌ **Angle computation** - Need GeometryObserver with spread/rational-trig
❌ **Reflection law test** - Need angle equality check
❌ **Action functional** - Beyond path length (Fermat's principle)
❌ **Symmetry validation** - λ-scaling invariance not tested yet
❌ **Multi-path comparison** - Only shortest paths explored

---

## Next Steps

### Immediate: Add GeometryAngleObserver

**Requirements:**
```python
class GeometryAngleObserver(QAObserver):
    """
    Computes angle-like observables from quadrances.
    Preferably using rational trig (spread) not floats.
    """
    def compute_spread(self, Q_SM, Q_MT, mirror_normal) -> QARational:
        # Rational computation of spread = (sin θ)²
        pass

    def check_reflection_law(self, spread_incidence, spread_reflection) -> bool:
        # Test equality (within tolerance if needed)
        pass
```

### Then: Projection Law Tests

```python
qa_physics/optics/test_reflection_projection_laws.py

def test_reflection_law_across_observers():
    """
    For each observer, check if angle(incidence) = angle(reflection)
    holds for the optimal candidate.

    Expected:
    - NullObserver: Likely fails (no geometry interpretation)
    - GeometryAngleObserver: Should pass
    - AffineTime: Depends on invariant preservation
    """
```

### Future: Action Functional

**Fermat's principle:** Light takes path minimizing optical path length

**QA version:**
- Define action = sum over path of some invariant functional
- Minimal action → reflection law
- Test if QA-optimal path matches geometric-optimal path

---

## Validation Criteria

**This module is validated by:**
1. ✅ Demo runs without errors
2. ✅ Produces 41 distinct states for 41 parameter values
3. ✅ ReflectionState satisfies QAStateProtocol
4. ✅ Works with all locked observers
5. ✅ QARational never reduces (manual inspection)
6. ✅ Quadrances are exact (no float until to_float())
7. ✅ Topology preserved (41/41 unique observations)

**Falsification criteria:**
- If QARational auto-reduces → non-reduction axiom violated
- If observers collapse distinct states → topology destruction
- If k ≠ |path|-1 → Axiom T1 violated
- If float appears before projection → QA layer contaminated

**Status:** No violations detected.

---

## Key Design Decisions

### 1. **Non-Reducing Rationals**

**Why:** QA non-reduction discipline forbids 6/9 → 2/3

**How:** QARational stores (n,d) exactly as computed

**Tradeoff:** Denominators grow quickly, but exactness preserved

### 2. **Quadrance, Not Distance**

**Why:** √(dx² + dy²) is irrational for most cases

**How:** Keep Q = dx² + dy² as QARational

**Benefit:** Exact comparison, no float approximation

### 3. **Failure as First-Class Value**

**Why:** Exceptions break deterministic logging

**How:** Return `Point2D | ReflectionFailure`

**Benefit:** Failures are data, not control flow

### 4. **1D Parameter Space**

**Why:** Simplest non-trivial search problem

**How:** u ∈ ℤ, moves ±1

**Benefit:** Shortest paths trivial, focus on projection question

### 5. **Minimal Invariant Set**

**Why:** Observers should compute derived quantities

**How:** Expose S, M, T, Q_SM, Q_MT, Q_ST only

**Benefit:** No premature geometry interpretation

---

## Production Readiness

✅ **Type-safe** - Full type hints, Union types for failures
✅ **Immutable** - All dataclasses frozen
✅ **Exact** - QARational never reduces
✅ **Deterministic** - Shortest paths, stable parameter order
✅ **Protocol-compliant** - Satisfies QAStateProtocol
✅ **Failure-aware** - First-class failure values
✅ **Logged** - TransitionLogEvent infrastructure ready

---

## Constitutional Principle

> **Reflection is not a QA property - it's a projection property.**
> **QA provides exact invariants. Observer interprets them as "angles."**
> **The law of reflection is a statement about which projections preserve classical geometry.**

Any reflection claim must:
1. Specify which observer is used
2. Show how "angle" is computed from invariants
3. Test against null model baseline
4. Measure preservation of reflection law explicitly

---

## Status

**✅ REFLECTION MODULE OPERATIONAL - READY FOR GEOMETRY OBSERVER**

The optics module provides:
- Exact arithmetic substrate (QARational)
- Minimal geometric primitives (Point2D, LineABC)
- QA-time compliant state space (ReflectionState)
- Bounded search harness (deterministic paths + logs)
- Multi-observer demonstration (projection comparison)

**Next deliverable:** GeometryAngleObserver + reflection law tests

---

## Demo Output Summary

```
Problem: Mirror y=0, S=(-10,5), T=(10,6), u∈[-20,20]

Observer comparison:
- 3 observers tested
- 41 candidates generated
- 41/41 distinct observations (topology preserved)
- QA time: k ∈ [0,20] deterministic
- Observer time: t varies by projection

Invariants stable across observers:
- Q_SM, Q_MT, Q_ST exact
- Points S, M, T exact
- No float contamination

Ready for: angle computation + reflection law validation
```

**Commit ready.**
