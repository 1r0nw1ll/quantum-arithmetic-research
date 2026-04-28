Fix migration ordering when moving a field onto a new subclass

Teach the migration autodetector to order `RemoveField` before `CreateModel`
when a new inherited model introduces a field name that is being removed from
one of its bases in the same migration. Add a regression test covering the
base-to-subclass field move that previously raised `FieldError` during
`migrate`.
