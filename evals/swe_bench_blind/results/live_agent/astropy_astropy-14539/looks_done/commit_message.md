io.fits: handle `Q` variable-length columns in FITSDiff

Extend the table diff VLA branch to cover `Q` column formats in addition to
`P`, and add a regression test that compares a FITS file containing a `QD`
column against itself through `FITSDiff`.
