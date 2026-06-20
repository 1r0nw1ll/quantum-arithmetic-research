theorem ex38_list_append_assoc {α : Type u} (xs ys zs : List α) : (xs ++ ys) ++ zs = xs ++ (ys ++ zs) :=
  by
    induction xs with
    | nil => rfl
    | cons x xs ih =>
        exact congrArg (List.cons x) ih
