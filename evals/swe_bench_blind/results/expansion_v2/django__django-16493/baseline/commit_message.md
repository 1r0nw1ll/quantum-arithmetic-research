Fix FileField deconstruction for callable default storage

Preserve the original callable passed to `FileField(storage=...)` when
deciding whether to include the `storage` kwarg during deconstruction, so
callables that return `default_storage` are still serialized consistently.
