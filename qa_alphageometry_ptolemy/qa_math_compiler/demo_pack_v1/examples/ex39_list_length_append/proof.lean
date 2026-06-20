theorem ex39_list_length_append {α : Type u} (xs ys : List α) : (xs ++ ys).length = xs.length + ys.length :=
  by
    induction xs with
    | nil => exact (Nat.zero_add ys.length).symm
    | cons x xs ih =>
        simp only [List.length, Nat.succ_add]
        exact congrArg Nat.succ ih
