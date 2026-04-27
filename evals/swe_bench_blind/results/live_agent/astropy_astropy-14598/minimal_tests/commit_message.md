Fix FITS CONTINUE string round-tripping for doubled quotes

Tighten `Card._strg_comment_RE` so long-string subcard parsing stops at the end
of the fragment, and preserve doubled single quotes while reassembling CONTINUE
string values instead of collapsing them early.
