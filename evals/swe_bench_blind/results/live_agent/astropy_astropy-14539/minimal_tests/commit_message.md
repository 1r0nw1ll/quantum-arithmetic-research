Handle Q-format VLA columns in FITSDiff

Teach `astropy.io.fits.TableDataDiff` to treat `Q` variable-length array
columns like `P` columns during comparison so identical FITS files do not
spuriously report table data differences.
