io.fits: preserve escaped quotes in CONTINUE string cards

Delay unescaping doubled single quotes until after `Card._split()` has
reassembled all `CONTINUE` fragments into one logical string. Add regression
coverage for long FITS string values that previously lost a trailing quote or
text following `''` during round-trips through `Card.fromstring()`.
