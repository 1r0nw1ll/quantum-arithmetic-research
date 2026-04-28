Fix InheritDocstrings for properties

Extend `InheritDocstrings` to treat data descriptors like properties the same
way as functions when inheriting missing docstrings, and add a regression test
covering property docstring inheritance.
