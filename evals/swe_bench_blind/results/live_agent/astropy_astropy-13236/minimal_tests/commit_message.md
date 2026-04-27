Remove structured ndarray auto-conversion to NdarrayMixin

Stop converting structured `numpy.ndarray` inputs into `NdarrayMixin` when
adding them to a table, so they continue through the normal column path.
