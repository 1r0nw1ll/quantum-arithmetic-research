Fixed #26431 -- Prevented resolve() from returning missing optional parameters.

Filter out optional named groups whose regex captures are absent so URL
resolution stays consistent when translated URLs are reversed.
