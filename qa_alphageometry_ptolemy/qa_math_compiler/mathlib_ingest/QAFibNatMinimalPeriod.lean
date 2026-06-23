import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic
import QAFibNatPeriodicity

open scoped Matrix

/-!
# Minimal Pisano period π(9) = 24 — minimality direction

Completes `π(9) = 24` by proving that 24 is the MINIMAL period.
`QAFibNatPeriodicity.lean` already established divisibility (period divides 24);
this file proves the minimality (24 divides every period).

The proof chain:
1. `fib_mat_pow_eq_one_iff`  : `fib_mat ^ m = 1 ↔ 24 ∣ m`
2. `fib_vec_period_iff`      : `fib_vec m = fib_vec 0 ↔ 24 ∣ m`
3. `pisano_period_9_exact`   : `(∀ n, (Nat.fib (n+m) : ZMod 9) = Nat.fib n) ↔ 24 ∣ m`

The key engine is `fib_mat_order_exact` (from `QAFibMatrix.lean`): for every
k ∈ {1,...,23}, `fib_mat ^ k ≠ 1`, which is verified by `native_decide`.

## Cert reference

- `[128]` Pisano period π(9) = 24 — minimality direction.
-/

-- ============================================================================
-- PRIVATE HELPERS
-- ============================================================================

-- For each k ∈ {1,...,23}, fib_vec k ≠ fib_vec 0 (verified by native_decide).
private theorem fib_vec_ne_fib_vec_zero :
    ∀ k : Fin 24, k.val ≠ 0 → fib_vec k.val ≠ fib_vec 0 := by native_decide

-- Reduce fib_vec m to fib_vec (m % 24) using fib_mat ^ 24 = 1.
private theorem fib_vec_mod24 (m : ℕ) : fib_vec m = fib_vec (m % 24) := by
  have hm : fib_vec m = (fib_mat ^ m) *ᵥ fib_vec 0 :=
    (fib_mat_pow_fib_vec m 0).symm.trans (by simp)
  have hmod : fib_vec (m % 24) = (fib_mat ^ (m % 24)) *ᵥ fib_vec 0 :=
    (fib_mat_pow_fib_vec (m % 24) 0).symm.trans (by simp)
  have hpow : fib_mat ^ m = fib_mat ^ (m % 24) := by
    conv_lhs => rw [show m = 24 * (m / 24) + m % 24 from by omega]
    rw [pow_add, pow_mul, fib_mat_pow_24, one_pow, one_mul]
  rw [hm, hpow, ← hmod]

-- ============================================================================
-- MATRIX POWER CHARACTERIZATION
-- ============================================================================

/-- **fib_mat ^ m = 1 iff 24 ∣ m** (cert [128]).

    The Fibonacci matrix `fib_mat` has exact multiplicative order 24 in
    M₂(ZMod 9): `fib_mat ^ m = 1` if and only if 24 divides m. -/
theorem fib_mat_pow_eq_one_iff (m : ℕ) : fib_mat ^ m = 1 ↔ 24 ∣ m := by
  constructor
  · intro h
    have hpow : fib_mat ^ (m % 24) = 1 := by
      have : fib_mat ^ m = fib_mat ^ (m % 24) := by
        conv_lhs => rw [show m = 24 * (m / 24) + m % 24 from by omega]
        rw [pow_add, pow_mul, fib_mat_pow_24, one_pow, one_mul]
      rw [← this]; exact h
    have hlt : m % 24 < 24 := Nat.mod_lt _ (by norm_num)
    rcases Nat.eq_zero_or_pos (m % 24) with h0 | hpos
    · exact Nat.dvd_of_mod_eq_zero h0
    · exact absurd hpow (fib_mat_order_exact ⟨m % 24, hlt⟩ (by omega))
  · rintro ⟨k, rfl⟩
    rw [pow_mul, fib_mat_pow_24, one_pow]

-- ============================================================================
-- FIBONACCI VECTOR PERIOD CHARACTERIZATION
-- ============================================================================

/-- **fib_vec m = fib_vec 0 iff 24 ∣ m** (cert [128]).

    The Fibonacci column vector returns to its initial state `fib_vec 0 = ![1, 0]`
    exactly when 24 divides m. -/
theorem fib_vec_period_iff (m : ℕ) : fib_vec m = fib_vec 0 ↔ 24 ∣ m := by
  constructor
  · intro h
    have hmod : fib_vec (m % 24) = fib_vec 0 := (fib_vec_mod24 m).symm.trans h
    have hlt : m % 24 < 24 := Nat.mod_lt _ (by norm_num)
    rcases Nat.eq_zero_or_pos (m % 24) with h0 | hpos
    · exact Nat.dvd_of_mod_eq_zero h0
    · exact absurd hmod (fib_vec_ne_fib_vec_zero ⟨m % 24, hlt⟩ (by omega))
  · rintro ⟨k, rfl⟩
    have h24k : 24 * k % 24 = 0 := by omega
    rw [fib_vec_mod24, h24k]

-- ============================================================================
-- MINIMAL PISANO PERIOD
-- ============================================================================

/-- **π(9) = 24: exact Pisano period characterization** (cert [128] SP2).

    The Fibonacci sequence mod 9 has m as a period if and only if 24 | m.
    Combined with `fib_nat_mod9_periodic`, this completely establishes π(9) = 24:
    the sequence repeats with period 24, and no smaller positive integer is a period. -/
theorem pisano_period_9_exact (m : ℕ) :
    (∀ n : ℕ, (Nat.fib (n + m) : ZMod 9) = Nat.fib n) ↔ 24 ∣ m := by
  rw [← fib_vec_period_iff]
  constructor
  · intro h
    -- Extract fib(m) ≡ 0 and fib(m+1) ≡ 1 from the periodicity hypothesis.
    have hm : (Nat.fib m : ZMod 9) = 0 := by
      have h0 := h 0
      simp only [Nat.zero_add] at h0
      simpa using h0
    have hm1 : (Nat.fib (m + 1) : ZMod 9) = 1 := by
      have h1 := h 1
      simp only [show 1 + m = m + 1 from by omega] at h1
      simpa using h1
    -- Reconstruct fib_vec m = fib_vec 0 from the two components.
    ext i
    fin_cases i
    · simp only [fib_vec, Matrix.cons_val_zero]
      rw [hm1]; native_decide
    · simp only [fib_vec, Matrix.cons_val_one, Matrix.head_cons]
      rw [hm]; native_decide
  · intro heq n
    -- Use the matrix-action chain: fib_vec(n+m) = F^n · fib_vec(m) = F^n · fib_vec(0) = fib_vec(n).
    have key : fib_vec (n + m) = fib_vec n :=
      calc fib_vec (n + m)
          = (fib_mat ^ n) *ᵥ fib_vec m := (fib_mat_pow_fib_vec n m).symm
        _ = (fib_mat ^ n) *ᵥ fib_vec 0 := by rw [heq]
        _ = fib_vec (n + 0) := fib_mat_pow_fib_vec n 0
        _ = fib_vec n := by simp
    simpa [fib_vec] using congr_fun key 1
