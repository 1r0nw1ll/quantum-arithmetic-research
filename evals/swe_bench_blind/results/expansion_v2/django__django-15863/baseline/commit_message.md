Preserve `Decimal` precision in `floatformat`

Avoid converting `Decimal` inputs through `repr()` before parsing in
`floatformat`, and add a regression test for high-precision decimal values.
