Preserve inner AttributeError from subclassed SkyCoord properties

Retry property descriptors in `SkyCoord.__getattr__` so that an
`AttributeError` raised inside a subclass property reports the missing
inner attribute instead of incorrectly reporting the property name.
