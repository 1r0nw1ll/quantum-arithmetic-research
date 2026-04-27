Fixed #11211 -- Normalize GenericForeignKey prefetch keys for UUID PKs

Adjust `GenericForeignKey` prefetch matching to compare prepared primary key
values on both sides of the in-memory join, and add a regression test covering
`prefetch_related()` against a UUID-backed generic foreign key.
