Guard FITS format identification with no positional data object

Prevent `astropy.io.fits.connect.is_fits` from indexing into an empty
`args` tuple during `identify_format(..., origin="write", ...)`, and add
a regression test covering the non-FITS path case.
