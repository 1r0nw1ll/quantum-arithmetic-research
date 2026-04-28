Handle inherited docstrings for property descriptors

Extend `InheritDocstrings` so it copies docstrings for data descriptors,
including `property`, instead of limiting inheritance to plain functions.
Add a regression test covering a subclassed property with an omitted
docstring.
