theorem ex41_list_map_comp {α β γ : Type u} (f : β → γ) (g : α → β) (xs : List α) : List.map f (List.map g xs) = List.map (fun x => f (g x)) xs :=
  by
    induction xs with
    | nil => rfl
    | cons x xs ih =>
        simp only [List.map, ih]
