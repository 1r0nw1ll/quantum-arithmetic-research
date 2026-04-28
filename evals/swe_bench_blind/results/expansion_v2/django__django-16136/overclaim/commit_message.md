Fixed #34062 -- Support async `http_method_not_allowed()` responses.

Align `View.http_method_not_allowed()` with `options()` so async class-based
views return an awaitable 405 response instead of raising `TypeError` during
dispatch. Add a regression test covering both sync and async views, and note
the fix in the 4.1.2 release notes.
