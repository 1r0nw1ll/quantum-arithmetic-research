Fixed GenericForeignKey prefetch key normalization for UUID primary keys

Use the related model primary key's `to_python()` conversion when building
GenericForeignKey prefetch match keys so `CharField` object IDs can match
UUID primary keys consistently.
