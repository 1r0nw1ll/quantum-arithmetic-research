Preserve admin catch-all redirect query strings

Use `request.get_full_path(force_append_slash=True)` in
`AdminSite.catch_all_view()` so APPEND_SLASH redirects keep query strings.
Add regression tests for query-string redirects with and without script-name
prefixes.
