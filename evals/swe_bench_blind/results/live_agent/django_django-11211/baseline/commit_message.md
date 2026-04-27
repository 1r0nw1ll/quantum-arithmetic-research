Fixed GenericForeignKey prefetch matching for UUID primary keys

Normalize prefetched related object primary keys with the model field's
`get_prep_value()` so GenericForeignKey joins match object ID values stored as
strings. Add a regression test covering `prefetch_related()` for a GFK pointing
to a model with a UUID primary key.
