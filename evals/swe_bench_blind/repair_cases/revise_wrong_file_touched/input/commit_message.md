# Coerce GFK object ids to strings before prefetch

Adds a defensive cast in GenericForeignKey.get_prefetch_queryset so
non-int primary keys (UUID, etc.) are handled.
