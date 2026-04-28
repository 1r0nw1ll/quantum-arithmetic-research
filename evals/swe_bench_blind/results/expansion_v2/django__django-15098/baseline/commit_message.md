Fixed #15098 -- Accepted script+region language prefixes in i18n URL paths.

Allow `get_language_from_path()` to recognize configured `LANGUAGES` entries
with `lang-script-region` prefixes, including case-insensitive matches for
aliases without separate translation catalogs.
