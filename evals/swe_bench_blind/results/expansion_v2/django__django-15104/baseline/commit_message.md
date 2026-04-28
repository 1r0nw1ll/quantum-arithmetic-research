Handle custom relation fields without a deconstructed `to` kwarg.

Use `pop("to", None)` in the migration autodetector's relation-agnostic
field normalization so custom `ForeignKey` subclasses that hardcode and omit
`to` during deconstruction don't raise `KeyError`.
