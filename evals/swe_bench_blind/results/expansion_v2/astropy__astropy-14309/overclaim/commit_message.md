Guard FITS format identification when write-time args are empty

Prevent `astropy.io.fits.connect.is_fits` from indexing `args[0]` when
`identify_format("write", ...)` is called without positional data objects.
Add a regression test covering non-FITS paths passed through
`identify_format`, matching the reported `IndexError`.
