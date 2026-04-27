Fixed optional named group handling during URL resolution

Filter out uncaptured optional named groups from `RegexPattern.match()` so
`translate_url()` and other reverse-by-kwargs paths don't receive literal
`None` values for missing regex groups. Added regression coverage for
optional regex parameters in URL resolution and i18n URL translation.
