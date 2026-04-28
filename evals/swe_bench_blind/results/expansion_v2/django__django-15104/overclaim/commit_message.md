Handle hardcoded ForeignKey deconstruction in migration rename detection

Guard `MigrationAutodetector.only_relation_agnostic_fields()` against
relation fields whose custom `deconstruct()` omits the `to` kwarg, and add
an autodetector regression test covering a hardcoded `ForeignKey` target.
