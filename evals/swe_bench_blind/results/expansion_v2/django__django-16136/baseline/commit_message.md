Fixed #34062 -- Updated View.http_method_not_allowed() to support async.

Wrap `HttpResponseNotAllowed` in a coroutine for async class-based views,
matching the existing `options()` behavior. Add a regression test and the
4.1.2 release note entry.
