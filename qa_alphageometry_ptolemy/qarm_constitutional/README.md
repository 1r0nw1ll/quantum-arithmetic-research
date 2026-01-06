# QARM v0.2 Constitutional Implementation

**Rust mirror of QARM_v02_Failures.tla - TLA+ verified**

## Overview

This library is a **mechanical translation** of the QARM v0.2 TLA+ specification into executable Rust code. It serves as the constitutional authority for all QA/QARM implementations.

**Constitutional Principle:** This code is a **compiler target**, not a design space. Modifications require updating the TLA+ spec first and re-running TLC verification.

## Verification Status

✅ **TLC Model Checking:** 504 states (full), 383 states (no-μ), 0 violations
✅ **GLFSI Theorem:** Confirmed by Rust execution
✅ **All Invariants:** Hold across all reachable states
✅ **Test Suite:** 27/27 tests passing

## Constitutional Parameters

```rust
pub const CAP: u32 = 20;        // Matches TLC config
pub const KSET: &[u32] = &[2, 3];  // Matches TLC config
```

## Core Types

```rust
pub enum FailType {
    Ok,
    OutOfBounds,
    FixedQViolation,
    Illegal,
}

pub enum Move {
    None,
    Sigma,  // Growth: e → e+1
    Mu,     // Swap: b ↔ e
    Lambda, // Scale: (b,e) → (k·b, k·e)
}

pub struct State {
    pub b: u32,
    pub e: u32,
    pub d: u32,    // Always d = b+e
    pub a: u32,    // Always a = d+e = b+2e
    pub qtag: u16, // Duo-modular: 24·φ₉(a) + φ₂₄(a)
    pub fail: FailType,
    pub last_move: Move,
}
```

## Invariants (TLA+ Mirrored)

All invariants are mechanically enforced:

1. **Inv_TupleClosed:** `d = b + e` and `a = d + e`
2. **Inv_InBounds:** All values in `[0, CAP]`
3. **Inv_QDef:** `qtag = qdef(a)`
4. **Inv_FailDomain:** `fail` in valid domain
5. **Inv_MoveDomain:** `last_move` in valid domain

## Generator Functions

```rust
pub fn step_sigma(s: &State) -> State;      // σ: e → e+1
pub fn step_mu(s: &State) -> State;         // μ: b ↔ e
pub fn step_lambda(s: &State, k: u32) -> State;  // λ_k: scale by k
```

**Semantics:** Absorbing failure states. Once `fail != Ok`, all moves return unchanged state.

## GLFSI Theorem

**Generator-Local Failure Signature Invariance (GLFSI):**

Per-generator failure counts are **invariant** under generator set changes:

| Generator | OUT_OF_BOUNDS | FIXED_Q_VIOLATION | Status |
|-----------|---------------|-------------------|--------|
| σ         | 21            | 100               | ✅ INVARIANT |
| λ         | 105           | 35                | ✅ INVARIANT |
| μ         | 40            | 74                | ✅ ABSENT when removed |

**Theorem:** Failure modes are intrinsic to (state, generator) pairs, not emergent properties of global reachable sets.

## Usage

### Run Constitutional Verification

```bash
cargo run --bin verify_glfsi
```

**Expected output:**
```
✅ GLFSI THEOREM CONFIRMED
✅ Constitutional verification PASSED
   QARM v0.2 Rust implementation matches TLA+ spec exactly.
```

### Run Test Suite

```bash
cargo test
```

**Expected:** `27 passed; 0 failed`

### Use as Library

```rust
use qarm_constitutional::{State, FailType, Move, step_sigma, verify_glfsi};

// Create initial state
let s = State {
    b: 0, e: 0, d: 0, a: 0,
    qtag: 0,
    fail: FailType::Ok,
    last_move: Move::None,
};

// Apply generator
let s2 = step_sigma(&s);

// Verify GLFSI theorem
verify_glfsi().expect("GLFSI must hold");
```

## Architecture

```
qarm_constitutional/
├── src/
│   ├── lib.rs              # Public API + integration tests
│   ├── main.rs             # GLFSI verification runner
│   ├── qarm_state.rs       # State representation + helpers
│   ├── qarm_moves.rs       # Generator implementations
│   ├── qarm_invariants.rs  # TLA+ invariant mirrors
│   └── qarm_glfsi.rs       # GLFSI theorem verification
├── Cargo.toml
└── README.md
```

## Compliance

**Constitutional Authority:** `QARM_v02_Failures.tla`
**TLC Verification:** `tlc_output_full.txt`, `tlc_output_nomu.txt`
**State Dumps:** `states_full.txt.dump`, `states_nomu.txt.dump`

## Development Rules

❌ **DO NOT** modify without updating TLA+ spec first
❌ **DO NOT** introduce heuristics or learning
❌ **DO NOT** add performance optimizations that change semantics
✅ **DO** update TLA+ spec first
✅ **DO** re-run TLC verification
✅ **DO** ensure all tests pass

## Version

**v0.2.0** - Constitutional lock
Tag: `qarm-v0.2-constitutional-lock`

## License

Constitutional implementation - mathematical research

## Citation

If using this implementation, cite:

```
QA/QARM Constitutional Implementation v0.2
TLA+ Verified Rust Mirror
Generator-Local Failure Signature Invariance (GLFSI) Theorem
2025
```

---

**Status:** ✅ **PRODUCTION READY** - Constitutional authority locked
