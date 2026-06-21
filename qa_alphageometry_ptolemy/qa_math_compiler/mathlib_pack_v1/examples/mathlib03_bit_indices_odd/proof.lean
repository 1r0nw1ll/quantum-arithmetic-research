import Mathlib.Data.Nat.BitIndices

theorem qa_mathlib03_bit_indices_odd (n : Nat) : (2 * n + 1).bitIndices = 0 :: List.map (fun x => x + 1) n.bitIndices := by
  exact Nat.bitIndices_two_mul_add_one n
