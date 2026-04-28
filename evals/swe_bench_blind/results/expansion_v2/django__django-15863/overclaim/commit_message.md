Preserve Decimal precision in `floatformat`

Avoid converting `Decimal` inputs through `float()` in `django.template.defaultfilters.floatformat`, which could truncate high-precision values before rounding. Add a regression test covering a 20-decimal-place `Decimal` rendered through the filter.
