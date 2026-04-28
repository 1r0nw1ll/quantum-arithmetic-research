io.fits: preserve compact float string formatting in Card

Use Python's default float string representation in `Card._format_float()`
so FITS cards do not expand values unnecessarily and truncate valid comments.
Add a regression test covering HIERARCH cards with float values that previously
grew longer than needed.
