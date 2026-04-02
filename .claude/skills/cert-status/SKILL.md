---
name: cert-status
description: Show certificate ecosystem status — family count, recent additions, any gaps
user_invocable: true
---

Show the current state of the QA certificate ecosystem:

1. Run `cd qa_alphageometry_ptolemy && python qa_meta_validator.py` and report pass/fail count
2. List the 5 most recently modified family directories under `qa_alphageometry_ptolemy/` (by mtime)
3. Check Open Brain for any uncertified observations: `mcp__open-brain__search_thoughts` with query "cert gap" (limit 5)

Format as a compact status dashboard.
