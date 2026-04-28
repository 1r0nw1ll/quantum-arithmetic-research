Fixed QuerySet.aggregate() after annotate() crash on aggregates with default

Preserve the aggregate summary flag when wrapping defaulted aggregates in
`Coalesce` so `aggregate()` continues to generate a summarized select after
`annotate()`. Add regression coverage for both aggregating the defaulted
annotation and aggregating a different field while the defaulted aggregate
remains only on the annotation.
