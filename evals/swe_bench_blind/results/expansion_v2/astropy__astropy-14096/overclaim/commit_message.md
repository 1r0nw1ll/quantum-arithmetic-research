Preserve original AttributeError details in SkyCoord attribute fallback

Route unresolved `SkyCoord.__getattr__` lookups through `__getattribute__`
instead of synthesizing a new `AttributeError`, so subclass properties that
fail on a missing attribute now report the underlying attribute name. Add a
regression test covering subclassed `SkyCoord` property access.
