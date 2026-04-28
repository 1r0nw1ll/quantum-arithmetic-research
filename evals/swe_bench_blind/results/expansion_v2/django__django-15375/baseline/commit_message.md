Preserve summary state when wrapping aggregates with `default`

Keep `Aggregate.default`'s internal `Coalesce()` wrapper marked as a summary
expression so annotated querysets can still aggregate correctly, and add
regression coverage for annotate-then-aggregate cases with defaults.
