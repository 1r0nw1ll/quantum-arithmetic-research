theorem ex34_congr_arg {α β : Sort u} (f : α → β) (a b : α) : a = b → f a = f b :=
  by
    intro h
    exact congrArg f h
