Fixed #26431 -- Prevent missing optional URL parameters from being returned by resolve().

Filter out `None` values from named regex groups so `resolve()` and
`translate_url()` don't treat absent optional parameters as literal `"None"`.
