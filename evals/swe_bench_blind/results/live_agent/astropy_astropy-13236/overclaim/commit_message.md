table: stop coercing structured ndarrays into NdarrayMixin

Remove the special-case conversion in `Table._convert_data_to_col` so
structured `numpy.ndarray` inputs are added through the normal `Column` /
`MaskedColumn` path. This aligns structured-array insertion with current table
column support and avoids the masked-structured-array mask loss caused by the
`NdarrayMixin` view conversion.
