Fixed `ModelChoiceIteratorValue` hashing for model choice widgets.

Add `__hash__()` based on the wrapped choice value so `ModelChoiceIteratorValue`
can be used in hashed lookups again, and add a regression test covering stable
hashes for equal values and distinct hashes for different values.
