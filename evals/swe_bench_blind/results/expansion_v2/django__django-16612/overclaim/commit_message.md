Fixed preserving query strings in `AdminSite.catch_all_view()`.

Use `request.get_full_path(force_append_slash=True)` for append-slash
redirects so admin catch-all redirects keep existing query strings.
Added a regression test covering a missing-slash admin URL with a query
string when `APPEND_SLASH=True`.
