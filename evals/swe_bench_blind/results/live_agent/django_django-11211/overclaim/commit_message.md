Fix GenericForeignKey prefetch key normalization for custom PK types

Normalize the prefetched generic foreign key value with the target model's
primary-key `get_prep_value()` before joining results in Python. This keeps
`prefetch_related()` working when a `GenericForeignKey` points at models whose
primary keys don't compare equal to the raw stored object ID value, such as
UUID-backed primary keys stored in a character field.
