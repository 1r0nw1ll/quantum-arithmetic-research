<!-- PRIMARY-SOURCE-EXEMPT: reason=internal QA axiom authority spec; primary source is the repo itself (QA_PHYSICS_PROJECTION_V0.1_LOCKED.md + QA_AXIOMS_BLOCK.md) -->
# QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1

**Status**: ACTIVE — mandatory for all empirical QA work
**Authority**: Theorem NT (QA_PHYSICS_PROJECTION_V0.1_LOCKED.md)
**Date**: 2026-03-28

---

## The Complete QA Axiom Set

These are the binding constraints on all QA computation. Theorem NT is one of seven axiom groups. All apply.

### A-group: Structural Axioms (from QA_AXIOMS_BLOCK.md)

| ID | Axiom | Rule |
|----|-------|------|
| A1 | No-Zero | State space is {1,...,N}, not {0,...,N-1}. b,e ∈ Z_{>0}. Zero is not a QA state. |
| A2 | Derived Coordinates | d = b+e, a = b+2e. These are NEVER independent. Never assign d or a directly. |
| A3 | Exact Rational L | L = (C·F)/12 is an exact rational (Python `Fraction`), never a float. |
| A4 | Deterministic Failures | Failures are deterministic. No stochastic relaxation, no continuous fault tolerance. |

### T-group: Time Axioms (from QA_PHYSICS_PROJECTION_V0.1_LOCKED.md)

| ID | Axiom | Rule |
|----|-------|------|
| T1 | Path Time | QA time = path length k ∈ ℕ. Continuous time is observer projection only. |
| T2 | Observer Firewall | All continuous functions are observer projections. Never causal inputs to QA logic. |

### S-group: Substrate Rules (implementation discipline)

| ID | Rule | Reason |
|----|------|--------|
| S1 | `x*x` not `x**2` or `pow(x,2)` | CPython `pow()` delegates to libm, which has ULP drift at ~0.24% rate on floats |
| S2 | `int` or `Fraction` only in QA arithmetic | No IEEE-754 float in any QA-layer computation |
| S3 | Canonical JSON: `sort_keys=True, separators=(',',':'), ensure_ascii=False` | Deterministic hash inputs |
| S4 | Hash: `sha256(domain.encode() + b'\x00' + payload)` | Domain separation in cert hashes |
| S5 | Manifest placeholder: 64 hex zeros, not string "placeholder" | Hash binding discipline |

### F-group: Fraction Axiom (exact geometry contexts)

| ID | Axiom | Rule |
|----|-------|------|
| F1 | Non-Reduction | Fractions are NOT automatically reduced. `{n:144,d:144}` ≠ `{n:1,d:1}`. Reduction is an explicit generator move (`RT_REDUCE_FRACTION`). Automatic GCD simplification = `ILLEGAL_NORMALIZATION` failure. |

---

## The Rule (Theorem NT)

> **Theorem NT**: Continuous time — and all continuous functions — only appear in observer projection.
> QA time = path length k ∈ ℕ (Axiom T1). The observer projection is where continuous physics enters.
> **QA dynamics remain discrete. Theorem NT is the firewall.**

**Consequence for empirical code**: No continuous quantity (float, r², FFT output, spectral density, correlation, amplitude, price return, neural activation) may serve as a causal input to QA orbit classification or generator selection. All such quantities are observer outputs — produced AFTER QA logic runs, never fed INTO it.

**Two distinct statuses — do not conflate them:**

| Status | Meaning |
|--------|---------|
| **Research-valid** | The underlying domain question is worth answering with QA |
| **QA-noncompliant** | The implementation violates one or more axioms and is not acceptable QA code |

"Research-valid" does not excuse noncompliance. A script can be research-valid and QA-noncompliant simultaneously. That is the honest description of every current Track D script. Recoverability means the research question can be re-implemented correctly — it does not mean the current implementation is acceptable.

---

## Causal Order (required in all compliant code)

```
[1] DISCRETE STATE DECLARATION
    Define the discrete state alphabet S (finite, integer-valued).
    S must not be derived from continuous signal values.
    S is declared a priori, independent of any physical measurement.

[2] OBSERVER DECLARATION
    Declare an explicit observer O: physical_signal → S
    State what O preserves and what it collapses.
    This is where the continuous measurement enters — as input to O only.
    O is one-way: physical → discrete. Never discrete → physical → discrete.

[3] QA CLASSIFICATION
    Given s ∈ S, compute (b, e) ∈ {1,...,N}² using the QA integer arithmetic.
    Compute f(b,e) = b²+b·e−e² (integer, in Z[φ]).
    Classify orbit (canonical rule — see qa_orbit_rules.py):
      singularity: b == m  AND  e == m   (unique fixed point)
      satellite:   (m//3)|b  AND  (m//3)|e   (e.g. 8|b AND 8|e for mod-24)
      cosmos:      everything else
    Note: v₃(f)=1 is algebraically impossible. b=e→singularity was incorrect.
    Authority: qa_orbit_rules.py::orbit_family() — import from there, do not reimplement.
    Apply generators σ, μ, λ, ν — all discrete, integer, no floats.

[4] PATH / TIME
    Record path length k ∈ ℕ. This is ALL QA time.
    No continuous time variable enters QA logic.

[5] OBSERVER PROJECTION (output)
    After QA logic is complete, project to continuous domain for measurement.
    Statistics (accuracy, r², p-value, t-test) computed here only.
    Results reported here are OBSERVATIONS of QA structure, not inputs to it.

FORBIDDEN PATTERN (Theorem NT violation):
    continuous_signal → discretize() → b, e → orbit_classify() → claim QA result

REQUIRED PATTERN:
    declare S → declare O → O(physical) → s ∈ S → QA classify → project → observe
```

---

## Compliance Gates (required for all empirical families and scripts)

### Structural Gates (A-group)

| Gate | Axiom | Check | Failure |
|------|-------|-------|---------|
| A1 | No-Zero | All b,e ∈ {1,...,N}. Zero never appears as a state value | `ZERO_STATE_VIOLATION` |
| A2 | Derived Coords | d and a computed as d=b+e, a=b+2e and never assigned independently | `DERIVED_COORD_VIOLATION` |
| A3 | Exact L | L=(C·F)/12 computed as Python `Fraction`, not float | `FLOAT_L_VIOLATION` |
| A4 | Deterministic | No random sampling inside QA-layer logic | `STOCHASTIC_QA_VIOLATION` |

### Time/Observer Gates (T-group)

| Gate | Axiom | Check | Failure |
|------|-------|-------|---------|
| T1 | Path Time | Discrete state alphabet S declared before physical data loaded | `UNDECLARED_STATE_ALPHABET` |
| T2-a | Observer | Observer O explicitly declared with preservation claims | `UNDECLARED_OBSERVER` |
| T2-b | Firewall | (b,e) come from S, not from continuous signal arithmetic | `CONTINUOUS_STATE_INJECTION` |
| T2-c | Firewall | No continuous quantity input to orbit_classify() or generator selection | `CONTINUOUS_CAUSAL_INJECTION` |
| T2-d | Firewall | Statistics/continuous metrics in projection layer only | `STATISTICS_IN_CAUSAL_LAYER` |
| T2-e | One-way | Observer direction is physical → discrete only | `OBSERVER_FEEDBACK_LOOP` |

### Substrate Gates (S-group)

| Gate | Rule | Check | Failure |
|------|------|-------|---------|
| S1 | No pow() | QA arithmetic uses `x*x` never `x**2` or `pow(x,2)` | `ULP_DRIFT_RISK` |
| S2 | No float | QA-layer values are `int` or `Fraction` only | `FLOAT_IN_QA_LAYER` |

### Fraction Gate (F-group, exact geometry only)

| Gate | Axiom | Check | Failure |
|------|-------|-------|---------|
| F1 | Non-Reduction | Fractions not automatically GCD-reduced; reduction = explicit move | `ILLEGAL_NORMALIZATION` |

---

## Exact Layer Boundaries

What is permitted in each layer — stated as hard rules, not suggestions:

### Observer Layer (O)
- **Permitted**: floats, numpy, scipy, STFT, filtering, rank computation, microstate clustering, k-means, any continuous signal processing needed to produce a discrete label
- **Required output**: `list[int]` where every element ∈ {1,...,N} (A1)
- **Forbidden**: QA orbit classification, generator application, any QA invariant computation

### QA Discrete Layer
- **Permitted**: `int` and `Fraction` arithmetic only (S2); `x*x` not `x**2` (S1); states ∈ {1,...,N} (A1); `d = b+e`, `a = b+2e` derived (A2)
- **Required**: orbit classification from `v₃(f(b,e))`; generator application (σ, μ, λ, ν); path length counting (T1)
- **Forbidden**: floats (S2); zero states (A1); independent assignment of d or a (A2); continuous time (T1)

### Projection / Validation Layer
- **Permitted**: floats, numpy, scipy, sklearn; accuracy, r², p-value, t-test, correlation; any statistics
- **Required**: comparison against declared null model
- **Forbidden**: feeding any result back as input to the QA discrete layer (T2-e)

### The Boundary in Code

```python
# OBSERVER LAYER: floats allowed, QA logic forbidden
discrete_states = observer.project(raw_signal)   # returns list[int] in {1,...,N}
observer._assert_states_valid(discrete_states, MODULUS)  # A1 enforced here

# QA LAYER: integers only, no floats
qa_results = run_qa_analysis(discrete_states, m=MODULUS)  # all int arithmetic

# PROJECTION LAYER: floats allowed again
observables = project_to_observables(qa_results)          # fractions → floats here
verdict = validate_against_null(observables, null_obs)    # statistics here
```

The boundary is a function call boundary. Cross it exactly twice: once going in (physical → discrete), once going out (discrete results → continuous observables). Never cross it a third time.

---

## Track D Script Compliance Ledger

### Finance Scripts

#### `qa_finance_joint_transition.py`
- **Discrete QA state**: ❌ NOT declared. b, e derived from `equalize_quantize(spy_ret, m)` — rank of continuous log-returns → integer bin
- **Generator path**: `qa_step(b, e, m) = (e, (b+e) % m)` — T applied to derived state ✓
- **Observer projection**: `orbit_family()` output correlated with VIX/returns — correct direction ✓
- **Continuous injection**: `b_states = equalize_quantize(log_returns, m)` at line 390-391 — **NT-3 VIOLATION**
- **Research question**: Do SPY/TLT co-movement states map to QA orbit families? Is stress concentrated in satellite/singularity?
- **Classification**: **SALVAGEABLE** — rank encoding is the closest to defensible; the structural questions are valid
- **Fix**: Declare S = {(spy_quintile, tlt_quintile)} as a 2D ordinal state; declare O = "5-day rolling rank maps return magnitude to ordinal position"; require O to be stable across instruments; run QA on the discrete quintile pair

#### `qa_finance_transition_structure.py`
- **Classification**: **SALVAGEABLE** — similar to above; check for same rank encoding pattern
- **Fix**: Same as above

#### `backtest_advanced_strategy.py`
- **Continuous injection**: HI computed from continuous prices treated as (b,e)
- **Classification**: **SALVAGEABLE** — if refactored so HI is computed from discrete regime labels (bull/bear/sideways) projected from price data, not from raw prices as QA state

---

### EEG Scripts

#### `eeg_hi2_0_experiment.py` (canonical EEG script)
- **Discrete QA state**: ❌ NOT declared. `b = int(features_7d[i, 0] * 23) + 1` at lines 76-78 — continuous EEG feature × 23 → integer
- **Generator path**: QA tuple computed from derived (b,e) ✓
- **Observer projection**: HI 2.0 used to classify seizure/non-seizure — correct direction for the classification task ✓
- **Continuous injection**: `features_7d` are continuous spectral/coherence EEG features scaled directly to b, e — **NT-3 VIOLATION**
- **Research question**: Does QA orbit structure in EEG signals predict seizure onset?
- **Classification**: **SALVAGEABLE** — the research question is valid and important
- **Fix**: Define S = {EEG microstate label} using a pre-trained microstate classifier (e.g., k-means on EEG topography → 4–8 discrete classes). Declare O = "EEG microstate segmentation maps continuous EEG to discrete microstate labels." Then (b=microstate_t, e=microstate_{t+1}) forms a 2D discrete state from LABELED microstates, not from continuous amplitudes. QA orbit then classifies the MICROSTATE TRANSITION PAIR.

#### `eeg_hi2_0_balanced_experiment.py`, `eeg_hi2_0_balanced_quick.py`, `eeg_hi2_0_experiment_fast.py`
- **Classification**: **SALVAGEABLE** — same violation, same fix as above
- All are variants of the same architecture

#### `eeg_brain_feature_extractor.py`, `eeg_brain_feature_extractor_enhanced.py`, `eeg_brain_feature_extractor_fixed.py`
- **Role**: Feature extraction pipeline feeding into HI computation
- **Classification**: **SALVAGEABLE** — the extractor should output discrete microstate labels, not continuous features for QA consumption

---

### Audio Scripts

#### `qa_audio_orbit_test.py`
- **Discrete QA state**: ❌ NOT declared. `quantize(samples, m)` = `(samples * m).astype(int)` — continuous amplitude → integer
- **Generator path**: T(b,e) = (e, (b+e) % m) applied to amplitude-derived states
- **Observer projection**: `orbit_follow_rate` — does next sample follow QA generator? This is the key metric
- **Continuous injection**: `states[t] = int(amplitude[t] * m)` — **NT-3 VIOLATION**
- **Research question**: Do authentic dynamical signals follow QA orbit trajectories more than noise?
- **Classification**: **SALVAGEABLE** — `orbit_follow_rate` as a concept is worth salvaging. The amplitude encoding is the problem.
- **Fix**: S = {frequency bin index} from STFT (short-time Fourier transform binned to m bins by ENERGY CONCENTRATION, not amplitude). Declare O = "dominant frequency bin at each window." Then (b=bin_t, e=bin_{t+1}) is a discrete frequency-state transition pair. orbit_follow_rate then asks: "does the frequency transition follow QA dynamics?" This is a legitimate structural question.

#### `qa_audio_ac_baseline.py`
- **Role**: Testing whether orbit_follow_rate is just lag-1 autocorrelation — explicitly a validity check
- **Classification**: **SALVAGEABLE** — the question it asks (is this a QA effect or just AC?) is correct scientific practice. But it tests the wrong orbit_follow_rate (from the noncompliant amplitude encoding). Fix the encoding first, then re-run this baseline.

#### `qa_audio_residual_control.py`
- **Classification**: **SALVAGEABLE** — same violation, same fix as orbit_test

#### `run_signal_experiments*.py` (all 6 variants)
- **Classification**: **SALVAGEABLE** — all use the same amplitude → quantize → (b,e) pattern
- The `_final` version is the canonical one; refactor that, the others become obsolete

---

### Seismic Scripts

#### `seismic_hi2_0_experiment.py`
- **Discrete QA state**: ❌ NOT declared. `b = int(normalized[i] * (self.modulus - 1)) + 1` at line 94 — continuous seismic amplitude normalized and scaled
- **Generator path**: QA tuple computed ✓
- **Observer projection**: Classification result compared with seismic wave type label
- **Continuous injection**: Continuous amplitude → (b,e) — **NT-3 VIOLATION**
- **Research question**: Does QA orbit structure predict seismic wave type (P/S/surface)?
- **Classification**: **SALVAGEABLE** — and importantly: seismic wave type IS already a discrete label in practice (seismologists label P-wave, S-wave, surface wave, coda). The correct encoding exists in the domain.
- **Fix**: Declare S = {quiet, p_wave, s_wave, surface_wave, coda} — these ARE the discrete seismic state labels already used in the [110] QA_SEISMIC_CONTROL_CERT.v1. O = "standard seismological wave-type identification." Then (b, e) encodes (wave_type_t, wave_type_{t+1}) using the integer label. QA orbit classifies the TRANSITION PAIR of wave types. This is the correct architecture and the [110] cert already has it. The experiment script just needs to match the cert.

#### `seismic_classifier_enhanced.py`, `seismic_data_generator.py`, `seismic_statistical_tests.py`
- **Classification**: **SALVAGEABLE** — support scripts, same fix applies

---

## Summary Ledger (All Axiom Violations)

Two columns per script: **Research** (is the question valid?) and **QA compliance** (is the code acceptable?). These are independent. Research-valid does not excuse noncompliance.

| Script | A1 No-Zero | T2-b Firewall | S1 x\*x | S2 No-float | Research status | QA compliance |
|--------|-----------|--------------|--------|-------------|-----------------|---------------|
| `qa_finance_joint_transition.py` | ❌ clip(0,m-1) | ❌ rank(return)→(b,e) | ✓ | ❌ numpy | research-valid | **QA-noncompliant** |
| `qa_finance_transition_structure.py` | ❌ same | ❌ same | ✓ | ❌ | research-valid | **QA-noncompliant** |
| `backtest_advanced_strategy.py` | ❌ | ❌ price→HI | ❌ | ❌ | research-valid | **QA-noncompliant** |
| `eeg_hi2_0_experiment.py` | ✓ (clamped) | ❌ feature×23→(b,e) | ✓ | ❌ floats | research-valid | **QA-noncompliant** |
| `eeg_hi2_0_balanced_experiment.py` | ✓ | ❌ same | ✓ | ❌ | research-valid | **QA-noncompliant** |
| `eeg_hi2_0_balanced_quick.py` | ✓ | ❌ same | ✓ | ❌ | research-valid | **QA-noncompliant** |
| `eeg_hi2_0_experiment_fast.py` | ✓ | ❌ same | ✓ | ❌ | research-valid | **QA-noncompliant** |
| `qa_audio_orbit_test.py` | ❌ clip(0,m-1) | ❌ amplitude→(b,e) | ❌ b_arr\*\*2 | ❌ numpy | research-valid | **QA-noncompliant** |
| `qa_audio_ac_baseline.py` | ❌ clip(0,m-1) | ❌ amplitude→(b,e) | check | ❌ | research-valid | **QA-noncompliant** |
| `qa_audio_residual_control.py` | ❌ clip(0,m-1) | ❌ amplitude→(b,e) | check | ❌ | research-valid | **QA-noncompliant** |
| `run_signal_experiments_final.py` | ❌ | ❌ amplitude→(b,e) | check | ❌ | research-valid | **QA-noncompliant** |
| `run_signal_experiments*.py` (5 old) | ❌ | ❌ | check | ❌ | superseded | **retire** |
| `seismic_hi2_0_experiment.py` | ✓ (+1 offset) | ❌ amplitude→(b,e) | check | ❌ | research-valid | **QA-noncompliant** |
| `seismic_classifier_enhanced.py` | ✓ (+1 offset) | ❌ same | check | ❌ | research-valid | **QA-noncompliant** |
| `seismic_statistical_tests.py` | ✓ | ❌ same | check | ❌ | research-valid | **QA-noncompliant** |

**Every current Track D script is QA-noncompliant. The research questions are research-valid. These are not the same statement.**

Violation notes:
- **A1**: Finance/audio `quantize`/`equalize_quantize` returns `clip(0, m-1)` — includes zero. EEG uses `+1` clamp — passes A1 but still fails T2-b.
- **T2-b**: Universal. Every script maps continuous data to (b,e). This is the primary violation.
- **S1**: `qa_audio_orbit_test.py` line 165 has `b_arr**2` — explicit S1 violation. Finance script correctly uses `b*b` and even comments "NEVER b**2".
- **S2**: Universal. Every script runs QA logic inside numpy float arrays. The QA layer is not separated from floating-point computation.
- **F1**: Not applicable to Track D (no exact rational geometry). Applies to cert families [44],[50],[55],[56] only.

---

## Canonical NT-Compliant Empirical Script Skeleton

{% raw %}
```python
# =============================================================================
# QA EMPIRICAL SCRIPT — Theorem NT Compliant Template
# Version: 1.0 | Spec: QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1
# =============================================================================

# ── [STEP 1] DECLARE DISCRETE STATE ALPHABET ─────────────────────────────────
# S must be defined BEFORE loading any physical data.
# S is a finite set of integer-valued states, declared a priori.

MODULUS = 9        # or 24 — must be declared, not inferred from data
# A1: states are {1,...,MODULUS}, never {0,...,MODULUS-1}
STATE_LABELS = {   # if states have physical meaning, declare it explicitly
    1: "label_A",  # A1: starts at 1, not 0
    2: "label_B",
    # ... up to MODULUS
}

# ── [STEP 2] DECLARE OBSERVER ─────────────────────────────────────────────────
# The observer maps continuous physical measurements → discrete states.
# Must declare: (a) what it preserves, (b) what it collapses.

class DeclaredObserver:
    """
    Observer: [physical signal type] → discrete states in {0,...,MODULUS-1}

    Preserves: [e.g., ordinal rank ordering / frequency bin identity / wave type label]
    Collapses: [e.g., absolute amplitude / phase / inter-sample variation]
    Direction: physical → discrete ONLY. No feedback from discrete to physical.
    """
    def project(self, physical_signal) -> list[int]:
        # All continuous→discrete logic lives here and ONLY here.
        # Return list of integer state indices in {1,...,MODULUS}. A1: never 0.
        raise NotImplementedError

    def _assert_states_valid(self, states: list[int], m: int) -> None:
        """A1 enforcement: all projected states must be in {1,...,m}."""
        for i, s in enumerate(states):
            assert 1 <= s <= m, f"A1 violation at index {i}: state={s} not in {{1,...,{m}}}"


# ── [STEP 3] QA DISCRETE LAYER ───────────────────────────────────────────────
# All functions below operate on integers only.
# No float arithmetic until Step 5.

def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    """Generator σ: T(b,e) = (e, d) where d = b+e in {1,...,m}
    A1: result is in {1,...,m}, never 0.
    The raw sum b+e can equal m (maps to m, not 0) or exceed m (wraps within {1,...,m}).
    Correct: ((b+e-1) % m) + 1  — maps Z_{>0} sums to {1,...,m}
    Wrong:   (b+e) % m          — produces 0 when b+e is a multiple of m
    """
    assert 1 <= b <= m and 1 <= e <= m, f"A1: b={b}, e={e} must be in {{1,...,{m}}}"
    d = ((b + e - 1) % m) + 1  # A1: result in {1,...,m}, never 0
    return e, d

def norm_f(b: int, e: int, m: int) -> int:
    """QA norm f(b,e) = (b*b + b*e - e*e) % m  — integer only, S1: x*x never x**2"""
    assert b >= 1 and e >= 1, f"A1 violation: b={b}, e={e} must be ≥ 1 (no zero)"
    return (b * b + b * e - e * e) % m  # S1: b*b not b**2

def derived_coords(b: int, e: int) -> tuple[int, int]:
    """A2: d and a are ALWAYS derived from (b,e). Never assign independently."""
    assert b >= 1 and e >= 1, f"A1 violation: b={b}, e={e}"
    d = b + e      # A2: d = b + e
    a = b + 2 * e  # A2: a = b + 2e
    return d, a

def three_adic_val(n: int) -> int:
    """v₃(n): 3-adic valuation"""
    if n == 0: return float('inf')
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v

def classify_orbit(b: int, e: int, m: int) -> str:
    """Classify (b,e) into orbit family. Integer arithmetic only.

    CANONICAL RULE (corrected 2026-03-28 — prior v₃/b==e rules were wrong):
      import from qa_orbit_rules instead of reimplementing here.
    """
    from qa_orbit_rules import orbit_family
    return orbit_family(b, e, m)

def path_length_to_target(b: int, e: int, target_orbit: str, m: int, max_k: int = 100) -> int | None:
    """Minimum path length k to reach target orbit. Returns None if not reached."""
    for k in range(max_k):
        if classify_orbit(b, e, m) == target_orbit:
            return k
        b, e = qa_step(b, e, m)
    return None


# ── [STEP 4] RUN QA LOGIC ─────────────────────────────────────────────────────
# Apply observer first, then QA logic on discrete states.
# Record path lengths (integers), orbit sequences (strings).

def run_qa_analysis(physical_signal, observer: DeclaredObserver, m: int = MODULUS) -> dict:
    """
    Main QA analysis function.
    Input: physical signal (continuous) + declared observer
    Output: dict of discrete QA results (orbit sequence, path lengths, etc.)
    All values in output are integers or strings — NO floats yet.
    """
    # Step 2: Observer projects continuous → discrete
    discrete_states = observer.project(physical_signal)

    # Step 3: QA discrete classification
    orbit_sequence = []
    path_records = []

    for i in range(len(discrete_states) - 1):
        b = discrete_states[i]
        e = discrete_states[i + 1]
        orbit = classify_orbit(b, e, m)
        orbit_sequence.append(orbit)

        # Does next state follow QA generator T?
        pred_b, pred_e = qa_step(b, e, m)
        follows_generator = (
            i + 2 < len(discrete_states) and
            discrete_states[i + 2] == pred_e  # T maps b→e as first component
        )
        path_records.append({
            "b": b, "e": e, "orbit": orbit,
            "follows_generator": follows_generator  # boolean, not float
        })

    return {
        "orbit_sequence": orbit_sequence,
        "path_records": path_records,
        "n_steps": len(path_records),
    }


# ── [STEP 5] OBSERVER PROJECTION (continuous measurements) ───────────────────
# ONLY HERE may continuous/statistical operations appear.
# Input: QA results (discrete). Output: measured observables.

def project_to_observables(qa_results: dict) -> dict:
    """
    Project discrete QA results to continuous observables for validation.
    All floats, fractions, and statistics live HERE only.
    """
    n = qa_results["n_steps"]
    orbit_seq = qa_results["orbit_sequence"]
    path_records = qa_results["path_records"]

    # Orbit distribution (fractions — continuous, observer layer)
    from collections import Counter
    counts = Counter(orbit_seq)
    orbit_fractions = {k: counts[k] / n for k in ["cosmos", "satellite", "singularity"]}

    # Generator follow rate (fraction — continuous, observer layer)
    follow_rate = sum(1 for r in path_records if r["follows_generator"]) / n

    return {
        "orbit_fractions": orbit_fractions,
        "generator_follow_rate": follow_rate,  # fraction: observer output only
        "n_steps": n,
    }


# ── [STEP 6] STATISTICAL VALIDATION ──────────────────────────────────────────
# Compare observed values against null model. Statistics here only.

def validate_against_null(observables: dict, null_observables: dict) -> dict:
    """
    Compare QA observables against null model.
    p-values, t-tests, effect sizes — all here, never earlier.
    """
    from scipy import stats
    # Example: is generator_follow_rate above null?
    obs_rate = observables["generator_follow_rate"]
    null_rate = null_observables["generator_follow_rate"]
    # ... statistical test ...
    return {"verdict": "CONSISTENT" if obs_rate > null_rate else "CONTRADICTS"}


# ── USAGE EXAMPLE ─────────────────────────────────────────────────────────────
# observer = MyDeclaredObserver()
# qa_results = run_qa_analysis(raw_signal, observer, m=MODULUS)
# observables = project_to_observables(qa_results)
# verdict = validate_against_null(observables, null_observables)
```
{% endraw %}

---

## Hard Gate: Required Declarations for New Empirical Families

Any new empirical QA family or script must include, as the FIRST artifact committed:

```json
{
  "schema": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
  "discrete_state_alphabet": {
    "source": "declared a priori, not derived from signal",
    "type": "finite integer set",
    "range": "[1, N] where N = modulus"
  },
  "observer": {
    "name": "...",
    "maps": "physical_signal_type → discrete_state_index",
    "preserves": "...",
    "collapses": "...",
    "direction": "physical → discrete ONLY"
  },
  "generator_set": ["sigma", "mu", "lambda", "nu"],
  "path_time_semantics": "k = integer path length, no continuous time",
  "prohibited_patterns": [
    "continuous_value → int() → b or e",
    "float * modulus → int → b or e",
    "statistics as input to orbit_classify()"
  ]
}
```

Any empirical artifact lacking this declaration is **NT-noncompliant by default**.

---

## Fix Priority

1. **EEG** — highest scientific value; microstate encoding is well-defined in neuroscience
2. **Seismic** — fix is trivial: use the wave-type labels already in [110] cert
3. **Audio** — fix requires frequency-bin encoding; orbit_follow_rate idea is worth preserving
4. **Finance** — fix requires explicit ordinal state declaration; rank encoding is closest to compliant
