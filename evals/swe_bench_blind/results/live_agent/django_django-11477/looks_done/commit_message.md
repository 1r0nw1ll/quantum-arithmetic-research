Fixed #11477 -- Ignore `None` for optional named groups in URL reversing.

Discard `None` keyword arguments in `URLResolver._reverse_with_prefix()`
so optional named groups resolved as missing can be reversed back to the
shorter URL form. Added an i18n regression test covering `reverse()` and
`translate_url()` with an optional named group.
