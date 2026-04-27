io.fits: handle Q-format VLA columns in FITSDiff

Treat `Q` variable-length array table columns the same as existing `P` VLA
handling in `TableDataDiff`, and add a regression test covering self-comparison
of a FITS file containing a `QD` column.
