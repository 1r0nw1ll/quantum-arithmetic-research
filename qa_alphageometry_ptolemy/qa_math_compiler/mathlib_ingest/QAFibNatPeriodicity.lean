import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic
import QAFibMatrix

/-!
# Fibonacci sequence periodicity mod 9

Proves `Nat.fib (n + 24) ≡ Nat.fib n (mod 9)` for all `n : ℕ`, using the
matrix-power formulation from `QAFibMatrix.lean`: `fib_mat ^ 24 = 1`.

The proof chain:
1. `fib_mat_mul_fib_vec`   : `fib_mat.mulVec (fib_vec n) = fib_vec (n + 1)`
2. `fib_mat_pow_fib_vec`   : `(fib_mat ^ n).mulVec (fib_vec m) = fib_vec (n + m)`
3. `fib_vec_periodic`      : `fib_vec (n + 24) = fib_vec n`
4. `fib_nat_mod9_periodic` : `(Nat.fib (n + 24) : ZMod 9) = Nat.fib n`

## Cert reference

- `[128]` Pisano period π(9) = 24 — matrix order implies sequence periodicity.
-/

open scoped Matrix

-- ============================================================================
-- FIBONACCI COLUMN VECTOR
-- ============================================================================

/-- The Fibonacci column vector at step n: `(fib(n+1) mod 9, fib(n) mod 9)`. -/
def fib_vec (n : ℕ) : Fin 2 → ZMod 9 :=
  ![(Nat.fib (n + 1) : ZMod 9), (Nat.fib n : ZMod 9)]

-- ============================================================================
-- ONE-STEP MATRIX RECURRENCE
-- ============================================================================

/-- Multiplying by `fib_mat` advances `fib_vec` by one step. -/
theorem fib_mat_mul_fib_vec (n : ℕ) :
    fib_mat *ᵥ fib_vec n = fib_vec (n + 1) := by
  funext i
  fin_cases i
  · -- component 0: fib(n+1) + fib(n) = fib(n+2) (mod 9)
    simp [fib_mat, fib_vec, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    push_cast [Nat.fib_add_two]; ring
  · -- component 1: fib(n+1) = fib(n+1)
    simp [fib_mat, fib_vec, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

-- ============================================================================
-- ITERATED MATRIX ACTION
-- ============================================================================

/-- Applying `fib_mat ^ n` to `fib_vec m` yields `fib_vec (n + m)`. -/
theorem fib_mat_pow_fib_vec (n m : ℕ) :
    (fib_mat ^ n) *ᵥ fib_vec m = fib_vec (n + m) := by
  induction n generalizing m with
  | zero => simp [fib_vec]
  | succ n ih =>
    rw [pow_succ', ← Matrix.mulVec_mulVec, ih, fib_mat_mul_fib_vec]
    congr 1; omega

-- ============================================================================
-- PERIODICITY OF THE COLUMN VECTOR
-- ============================================================================

/-- The Fibonacci column vector `fib_vec` is periodic with period 24 (mod 9). -/
theorem fib_vec_periodic (n : ℕ) : fib_vec (n + 24) = fib_vec n := by
  have key : fib_vec (24 + n) = fib_vec n :=
    calc fib_vec (24 + n)
        = (fib_mat ^ 24) *ᵥ fib_vec n := (fib_mat_pow_fib_vec 24 n).symm
      _ = (1 : Matrix (Fin 2) (Fin 2) (ZMod 9)) *ᵥ fib_vec n := by rw [fib_mat_pow_24]
      _ = fib_vec n := Matrix.one_mulVec _
  rwa [Nat.add_comm] at key

-- ============================================================================
-- FIBONACCI PERIODICITY MOD 9
-- ============================================================================

/-- `Nat.fib` is periodic mod 9 with period 24:
    `(Nat.fib (n + 24) : ZMod 9) = Nat.fib n`. -/
theorem fib_nat_mod9_periodic (n : ℕ) :
    (Nat.fib (n + 24) : ZMod 9) = Nat.fib n := by
  have h := congr_fun (fib_vec_periodic n) 1
  simpa [fib_vec] using h
