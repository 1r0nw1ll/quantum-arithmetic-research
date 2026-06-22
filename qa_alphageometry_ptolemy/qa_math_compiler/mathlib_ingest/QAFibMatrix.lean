import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Tactic

/-!
# QA Fibonacci Matrix in GL₂(ZMod 9)

The QA T-step `(b, e) ↦ (b+e, b)` is the action of the Fibonacci matrix
```
F = [[1, 1],
     [1, 0]]
```
on column vectors in (ZMod 9)², via matrix-vector multiplication.

This file gives the group-theoretic characterisation of F:

- **F^24 = I** in M₂(ZMod 9) — the Pisano-period matrix identity.
- **Exact order 24**: F^k ≠ I for every k ∈ {1,...,23}.
  Together with F^24 = I, this establishes `ord(F) = 24` in GL₂(ZMod 9).
- **F is invertible**: det(F) = -1 ≡ 8 (mod 9), so det(F) ≠ 0 and F ∈ GL₂(ZMod 9).
- **T-step = matrix action**: Matrix.mulVec F ![b, e] = ![b+e, b] for all b e : ZMod 9.
- **Iteration = matrix power**: (F^k) * v = iterate(T, k, v) for all k : Fin 24.

These theorems form the algebraic bridge between the cert-layer QA orbit results
(QAOrbits.lean, QAOrbitPartition.lean) and standard GL₂ / Fibonacci-matrix theory.

## Cert references

- `[128]` Pisano period π(9) = 24 — F^24 = I and exact order
- `[126]` orbit structure — Fibonacci matrix generates the three orbit types
- `[153]` singularity dominance — det(F) ≠ 0 ensures F ∈ GL₂ (invertible)
-/

/-- The Fibonacci matrix F = [[1,1],[1,0]] over ZMod 9. -/
def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]

-- ============================================================================
-- GROUP-THEORETIC ORDER OF F IN GL₂(ZMod 9)
-- ============================================================================

/-- **F^24 = I** in M₂(ZMod 9) (cert [128] SP2, Pisano period).

    The Fibonacci matrix satisfies the Pisano-period matrix identity over ZMod 9.
    This is the matrix-ring lifting of `qa_t_period_divides_24` (QAOrbits.lean). -/
theorem fib_mat_pow_24 : fib_mat ^ 24 = 1 := by native_decide

/-- **Exact order 24** (cert [128] SP2).

    For every k ∈ {1,...,23}, F^k ≠ I in M₂(ZMod 9).
    Combined with `fib_mat_pow_24`, the exact multiplicative order of F is 24:
    ord(F) = 24 in the monoid M₂(ZMod 9) and in GL₂(ZMod 9). -/
theorem fib_mat_order_exact :
    ∀ k : Fin 24, k.val ≠ 0 → fib_mat ^ k.val ≠ 1 := by native_decide

-- ============================================================================
-- INVERTIBILITY: det(F) ≠ 0
-- ============================================================================

/-- **det(F) = 8** in ZMod 9 (cert [153]).

    det([[1,1],[1,0]]) = 1·0 - 1·1 = -1 ≡ 8 (mod 9).
    Since 8 ≠ 0 in ZMod 9, F is invertible: F ∈ GL₂(ZMod 9). -/
theorem fib_mat_det : Matrix.det fib_mat = 8 := by native_decide

/-- **F is invertible** (cert [153]).

    det(F) ≠ 0 in ZMod 9, so F lies in GL₂(ZMod 9). -/
theorem fib_mat_det_ne_zero : Matrix.det fib_mat ≠ 0 := by native_decide

-- ============================================================================
-- T-STEP = MATRIX ACTION
-- ============================================================================

/-- **T-step = matrix-vector multiplication by F** (cert [126]).

    The QA T-step `(b, e) ↦ (b+e, b)` is exactly the action of the Fibonacci
    matrix F on the column vector [b, e]ᵀ. This is the explicit algebraic
    bridge between the T-step and the Fibonacci matrix. -/
theorem fib_mat_action :
    ∀ b e : ZMod 9,
    Matrix.mulVec fib_mat ![b, e] = ![b + e, b] := by native_decide

-- ============================================================================
-- ITERATION = MATRIX POWER
-- ============================================================================

/-- **Iterating T k times = applying F^k** (cert [126] / [128]).

    For every k ∈ {0,...,23} and every starting state (b, e) ∈ (ZMod 9)²,
    the k-fold iterate of the T-step equals F^k applied to the column vector.
    This is the group-theoretic foundation for the orbit period theorems. -/
theorem fib_mat_iter :
    ∀ k : Fin 24, ∀ b e : ZMod 9,
    let qa_step := fun (s : ZMod 9 × ZMod 9) => (s.1 + s.2, s.1)
    let s := qa_step^[k.val] (b, e)
    Matrix.mulVec (fib_mat ^ k.val) ![b, e] = ![s.1, s.2] := by native_decide

-- ============================================================================
-- DERIVED: PERIODS FROM MATRIX ORDER
-- ============================================================================

/-- **Cosmos period 24 from matrix order** (cert [128] SP3).

    Because F^24 = I (fib_mat_pow_24) and the T-step is the matrix action,
    applying T 24 times to any state returns the original state.
    This is the matrix proof of `qa_t_period_divides_24` (QAOrbits.lean). -/
theorem fib_mat_cosmos_period :
    ∀ b e : ZMod 9,
    Matrix.mulVec (fib_mat ^ 24) ![b, e] = ![b, e] := by
  intro b e
  simp [fib_mat_pow_24]

/-- **Pisano π(9) = 24 from F^24 = I and exact order** (cert [128] SP2).

    The smallest k > 0 with F^k = I is k = 24.
    This is the matrix-algebraic proof of `qa_pisano_9_exact` (QAOrbitPartition.lean),
    now phrased purely in terms of matrix powers in M₂(ZMod 9). -/
theorem fib_mat_pisano_9 :
    (∀ k : Fin 24, k.val ≠ 0 → fib_mat ^ k.val ≠ 1) ∧ fib_mat ^ 24 = 1 :=
  ⟨fib_mat_order_exact, fib_mat_pow_24⟩
