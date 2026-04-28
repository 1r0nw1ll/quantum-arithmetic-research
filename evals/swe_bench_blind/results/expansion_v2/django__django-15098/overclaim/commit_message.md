Fix i18n path parsing for script-and-region locale prefixes

Expand `get_language_from_path()`'s prefix matcher so locale-prefixed URLs
accept `lang-script-region` codes such as `en-latn-us` and `en-Latn-US`.
Add regression coverage for these path-based locale variants.
