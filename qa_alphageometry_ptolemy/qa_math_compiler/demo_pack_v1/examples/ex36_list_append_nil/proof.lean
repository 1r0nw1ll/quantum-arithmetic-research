theorem ex36_list_append_nil {α : Type u} (xs : List α) : xs ++ [] = xs :=
  by
    induction xs with
    | nil => rfl
    | cons x xs ih =>
        exact congrArg (List.cons x) ih
