import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]

theorem fib_mat_iter :
    ∀ k : Fin 24, ∀ b e : ZMod 9,
    let qa_step := fun (s : ZMod 9 × ZMod 9) => (s.1 + s.2, s.1)
    let s := qa_step^[k.val] (b, e)
    Matrix.mulVec (fib_mat ^ k.val) ![b, e] = ![s.1, s.2] := by native_decide
