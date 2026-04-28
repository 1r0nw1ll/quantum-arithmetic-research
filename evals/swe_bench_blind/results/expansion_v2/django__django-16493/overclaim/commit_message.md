Fixed #34192 -- Preserve callable FileField storage during deconstruction.

Use the original storage callable, when present, for both the
`default_storage` comparison and the serialized `storage` kwarg so
`FileField.deconstruct()` keeps callable storage references that resolve to
`default_storage`. Add a regression test covering a callable that returns
`default_storage`.
