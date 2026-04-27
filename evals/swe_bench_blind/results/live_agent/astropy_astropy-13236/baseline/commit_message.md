table: stop auto-converting structured ndarrays to NdarrayMixin

Remove the special-case coercion that turned structured `numpy.ndarray`
inputs into `NdarrayMixin` when adding them to a `Table`. Structured arrays
now remain regular `Column` instances, and the existing regression test is
updated to cover that behavior while preserving explicit `NdarrayMixin`
inputs.
