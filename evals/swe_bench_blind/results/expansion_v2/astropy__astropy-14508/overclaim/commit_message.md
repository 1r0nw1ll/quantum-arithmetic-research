io.fits: preserve compact float formatting in FITS cards

Use Python's shortest round-trippable float string in `Card._format_float()`
before applying the FITS width limit so HIERARCH cards keep valid comments
instead of expanding values unnecessarily. Add regression coverage for the
reported truncated-comment case and representative boundary float values.
