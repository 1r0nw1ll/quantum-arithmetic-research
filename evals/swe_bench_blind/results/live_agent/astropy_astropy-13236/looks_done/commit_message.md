table: keep structured ndarrays as columns

Remove the automatic conversion of structured ndarrays to `NdarrayMixin`
during table column normalization. Update the structured-array regression
coverage to exercise both plain `Column` and explicit `NdarrayMixin`
inputs, and add a masked structured-array check for table assignment.
