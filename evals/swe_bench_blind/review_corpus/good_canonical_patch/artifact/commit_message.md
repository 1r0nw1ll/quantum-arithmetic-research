# Fix UUIDField.get_prep_value for GFK lookups

prefetch_related on a GenericForeignKey pointing at a model with a UUID
primary key was failing because UUIDField did not implement
get_prep_value. This adds get_prep_value, calling to_python on the
super-result so GFK queries can produce a properly typed comparison value.

Touches: django/db/models/fields/__init__.py.
