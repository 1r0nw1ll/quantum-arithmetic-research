# django/django: django__django-11477

**Repo:** `django/django` @ `e2867118`
**Difficulty:** 15 min - 1 hour

## Issue

translate_url() creates an incorrect URL when optional named groups are missing in the URL pattern
Description
	
There is a problem when translating urls with absent 'optional' arguments
(it's seen in test case of the patch)


## Maintainer hints

Could you please provide the patch as a pull request against master? Unless this is a regression, it probably doesn't qualify for a backport to 1.9 (see our ​supported versions policy). Please check the patch on Python 3 as well (dict.iteritems() doesn't exist there).
The issue here is that resolve() will return None for any optional capture groups that aren't captured, but reverse() will use force_text() and convert it to the literal string 'None'. This causes an asymmetry between resolve() and reverse(). I think it's a better idea to treat this asymmetry as a bug, and solve it within reverse(). We would discard any arguments that are None. Incidentally this will allow for cleaner code when you don't know if an optional parameter is set when you call reverse() or the {% url %} template tag: {% if obj.might_be_none != None %} {% url 'my_view' obj.might_be_none %} {% else %} {% url 'my_view' %} {% endif %} vs. {% url 'my_view' obj.might_be_none %} I think that's a reasonably common usecase, so it would be nice to support this cleaner syntax. Non-existent template variables will still pass an empty string rather than None, so this won't hide any typos in your template variable names.
This also affects the set_language function. When changing language on a page with an URL pattern containing an optional group, the empty optional part is set to none and translating the page leads to a 404. The bug on translation did not exist back in 1.8, not sure when it was introduced. I'm on 2.2. The best way to resolve both issues would be to patch directly _reverse_with_prefix in django.urls.resolvers.
​PR

## Deliverable

Produce a minimal unified-diff patch (`patch.diff`) that fixes the issue. Include a short `commit_message.md` describing the change scope. Do not claim more than the patch implements.
