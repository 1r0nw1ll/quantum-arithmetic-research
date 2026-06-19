theorem ex21_eq_symm {α : Sort u} (a b : α) : a = b → b = a :=
  by
    intro h
    exact Eq.symm h
