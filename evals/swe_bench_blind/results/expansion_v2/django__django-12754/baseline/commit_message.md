Fix migration ordering for inherited field moves

Teach the migration autodetector to make a newly created subclass depend on
removal of same-named fields from its concrete base models. Add a regression
test covering moving one base field onto multiple new subclasses in one
migration.
